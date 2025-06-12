# DIS DAT REAL SHIT
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
from astropy.wcs import WCS
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler, RobustScaler
from photutils.segmentation import detect_sources, deblend_sources
from photutils.background import Background2D, MedianBackground
from scipy.ndimage import gaussian_filter
import warnings
from astropy import units as u
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans


# Do sgas2111, middle third (crop fits and visualize)
class LensedGalaxyPipeline:
    def __init__(self, fits_files, base_path):
        """
        Initialize the pipeline for processing lensed galaxy FITS files.
        
        Parameters:
        -----------
        fits_files : list
            List of FITS filenames to process
        base_path : str
            Base directory path for the files
        """
        self.fits_files = fits_files
        self.base_path = base_path
        self.images = []
        self.flux_images = []  # Will store the calibrated flux images
        self.headers = []
        self.wavelengths = []
        self.pixel_areas = []  # Will store pixel area in square arcseconds
        self.wcs_list = []
        self.masks = []
        self.features = None
        self.embedding = None
        self.labels = None
        self.sf_regions = None
        
        # Extract approximate wavelengths from filenames (in microns)
        for filename in fits_files:
            if 'f150w' in filename:
                self.wavelengths.append(1.5)
            elif 'f200w' in filename:
                self.wavelengths.append(2.0)
            elif 'f277w' in filename:
                self.wavelengths.append(2.77)
            elif 'f356w' in filename:
                self.wavelengths.append(3.56)
            elif 'f444w' in filename:
                self.wavelengths.append(4.44)
            elif 'f480m' in filename:
                self.wavelengths.append(4.8)
            else:
                # Default fallback
                self.wavelengths.append(0.0)
    
    def load_fits_data(self):
        """Load all FITS files and extract images, headers, and WCS information."""
        print("Loading FITS data...")
        
        for i, filename in enumerate(self.fits_files):
            file_path = os.path.join(self.base_path, filename)
            
            try:
                with fits.open(file_path) as hdul:
                    # Assuming data is in the primary HDU
                    data = hdul[0].data
                    header = hdul[0].header
                    
                    # Handle different data dimensions
                    if data.ndim > 2:
                        # Take the first slice if more than 2D
                        data = data[0, 0, :, :] if data.ndim == 4 else data[0, :, :]
                    
                    # Store data and header
                    self.images.append(data)
                    self.headers.append(header)
                    
                    # Get the WCS information for coordinate transformations
                    wcs = WCS(header)
                    self.wcs_list.append(wcs)
                    
                    # Calculate pixel area in square arcseconds
                    # This is needed for flux density calculations
                    try:
                        # Get the pixel scale from the WCS
                        pixel_scale_x = np.abs(header.get('CDELT1', header.get('CD1_1', 0))) * 3600.0  # Convert degrees to arcsec
                        pixel_scale_y = np.abs(header.get('CDELT2', header.get('CD2_2', 0))) * 3600.0  # Convert degrees to arcsec
                        
                        # If CDELT/CD not available, try alternative keywords
                        if pixel_scale_x == 0 or pixel_scale_y == 0:
                            pixel_scale_x = np.abs(wcs.wcs.cdelt[0]) * 3600.0
                            pixel_scale_y = np.abs(wcs.wcs.cdelt[1]) * 3600.0
                        
                        # Calculate pixel area
                        pixel_area = pixel_scale_x * pixel_scale_y  # square arcseconds
                    except:
                        # Fallback to a default value if we can't get it from the header
                        print(f"  Warning: Could not determine pixel scale for {filename}, using default of 0.1 arcsec/pixel")
                        pixel_area = 0.01  # Default value of 0.1 arcsec × 0.1 arcsec
                        
                    self.pixel_areas.append(pixel_area)
                    
                    print(f"  Loaded {filename} - Shape: {data.shape}, Pixel area: {pixel_area:.6f} arcsec²")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                # Add placeholder data to maintain indexing
                self.images.append(np.zeros((100, 100)))
                self.headers.append(None)
                self.wcs_list.append(None)
                self.pixel_areas.append(0.01)  # Default placeholder
        from reproject import reproject_interp

        from scipy.ndimage import shift
        from skimage.registration import phase_cross_correlation

        # Use the first image as the reference
        ref_image = self.images[0]

        print("Aligning images using cross-correlation...")
        for i in range(1, len(self.images)):
            try:
                shift_yx, _, _ = phase_cross_correlation(ref_image, self.images[i], upsample_factor=10)
                print(f"  Image {i} shift = {shift_yx}")
                self.images[i] = shift(self.images[i], shift=shift_yx)
            except Exception as e:
                print(f"  Error aligning image {i}: {e}")


        # Make sure all images have the same shape by cropping or padding
        self._standardize_image_shapes()
        for i, img in enumerate(self.images):
            if img is not None:
                print(f"Image {i} shape after cropping: {img.shape}")

        # Convert image data to flux units
        self._convert_to_flux()
        
        return self
    
    def _standardize_image_shapes(self):
        """Ensure all images have the same shape by cropping to the smallest common dimensions."""
        if not self.images:
            return
            
        # Find the minimum dimensions across all images
        min_height = min(img.shape[0] for img in self.images if img is not None)
        min_width = min(img.shape[1] for img in self.images if img is not None)
        
        # Crop all images to the minimum dimensions
        for i, img in enumerate(self.images):
            if img is not None:
                self.images[i] = img[:min_height, :min_width]
                
        print(f"Standardized all images to shape: ({min_height}, {min_width})")

    def _convert_to_flux(self):
        """Convert image data from pixel values to physical flux units (if possible)."""
        print("Converting pixel values to flux...")

        self.flux_images = []

        for i, (image, header, wavelength, pixel_area) in enumerate(zip(self.images, self.headers, self.wavelengths, self.pixel_areas)):
            if image is None or header is None:
                print(f"  Image {i}: skipped (None)")
                self.flux_images.append(None)
                continue

            flux_image = None  # Start blank

            try:
                if 'PHOTFLAM' in header:
                    photflam = header['PHOTFLAM']
                    flux_image = image * photflam
                    print(f"  {self.fits_files[i]}: Using PHOTFLAM = {photflam:.3e}")

                elif 'PHOTFNU' in header:
                    photfnu = header['PHOTFNU']
                    flux_image = image * photfnu
                    print(f"  {self.fits_files[i]}: Using PHOTFNU = {photfnu:.3e}")

                elif 'FLUXCONV' in header:
                    fluxconv = header['FLUXCONV']
                    flux_image = image * fluxconv
                    print(f"  {self.fits_files[i]}: Using FLUXCONV = {fluxconv:.3e}")

                elif 'BUNIT' in header:
                    bunit = header['BUNIT'].lower()
                    if 'jy' in bunit:
                        flux_image = image  # Already in Jy
                        print(f"  {self.fits_files[i]}: Image already in flux units: {bunit}")
                    elif 'mjy/sr' in bunit:
                        sr_per_pixel = pixel_area * 2.35e-11  # arcsec² to sr
                        flux_image = image * 1e6 * sr_per_pixel  # MJy/sr → Jy/pixel
                        print(f"  {self.fits_files[i]}: Converting from {bunit} to Jy/pixel")

            except Exception as e:
                print(f"  Error processing {self.fits_files[i]}: {e}")

            # Fallback if no keyword was found
            if flux_image is None:
                print(f"  {self.fits_files[i]}: No flux keyword found — using raw pixel values as flux (unnormalized)")
                flux_image = image

            self.flux_images.append(flux_image)

        print("Conversion to flux units complete.")

    
    def preprocess_images(self):
        """Preprocess images by removing background and applying masks."""
        print("Preprocessing images...")
        
        # Create masks and background-subtracted images
        for i, image in enumerate(self.flux_images):
            # Skip if image is None or has very small dimensions
            if image is None or min(image.shape) < 10:
                self.masks.append(None)
                continue
                
            # Apply a Gaussian filter to reduce noise
            smoothed = gaussian_filter(image, sigma=0.8)
            
            # Estimate and subtract background
            bkg_estimator = MedianBackground()
            bkg = Background2D(smoothed, box_size=50, filter_size=3, 
                              bkg_estimator=bkg_estimator)
            image_nobkg = image - bkg.background
            
            # Replace original image with background-subtracted version
            self.flux_images[i] = image_nobkg
            
            # Create a mask for significant sources (signal above background)
            threshold = 3.0 * bkg.background_rms
            mask = image_nobkg > threshold
            
            # Apply morphological operations to clean up the mask
            from scipy import ndimage
            mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
            
            self.masks.append(mask)
            
            print(f"  Processed image {i+1}/{len(self.flux_images)}")
        
        return self
    
    def extract_features(self):
        """Extract pixel-level features for UMAP dimensionality reduction."""
        print("Extracting features...")
        
        # Check if we have valid images
        valid_indices = [i for i, img in enumerate(self.flux_images) if img is not None and np.any(img)]
        
        if not valid_indices:
            raise ValueError("No valid flux images to process")
            
        # Get shape from the first valid image
        h, w = self.flux_images[valid_indices[0]].shape
        
        # Create a multi-band feature array
        n_samples = h * w
        n_features = len(valid_indices) + 2  # Images + pixel coordinates
        
        feature_array = np.zeros((n_samples, n_features))
        
        # Add spatial coordinates (normalized)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        feature_array[:, 0] = (x_coords.flatten() / w)
        feature_array[:, 1] = (y_coords.flatten() / h)
        
        # Add flux data
        print("  Adding flux data to feature array...")
        for i, idx in enumerate(valid_indices):
            img = self.flux_images[idx]
            
            # Replace NaNs and infs with zeros
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize using robust scaling
            # This preserves relative flux ratios while making the data suitable for ML
            scaler = RobustScaler()
            try:
                normalized_img = scaler.fit_transform(img.reshape(-1, 1)).flatten()
            except:
                # Fallback if robust scaling fails (e.g., due to all zeros)
                print(f"    Warning: Robust scaling failed for image {idx}, using simple normalization")
                img_min = np.min(img)
                img_max = np.max(img)
                if img_max > img_min:
                    normalized_img = (img.flatten() - img_min) / (img_max - img_min)
                else:
                    normalized_img = np.zeros_like(img.flatten())
            
            # Add to feature array
            feature_array[:, i+2] = normalized_img
        
        # Create a composite mask from all individual masks
        composite_mask = np.zeros((h, w), dtype=bool)
        for i in valid_indices:
            if self.masks[i] is not None:
                composite_mask |= self.masks[i]
        
        # Only keep pixels that are part of detected sources in at least one band
        mask_flat = composite_mask.flatten()
        self.features = feature_array[mask_flat]
        
        # Save the mask shape for reconstructing the image later
        self.mask_shape = (h, w)
        self.mask_flat = mask_flat
        
        print(f"Extracted {self.features.shape[0]} features from {self.features.shape[1]} dimensions")
        
        return self
    
    def reduce_dimensions(self, n_components=2, n_neighbors=20, min_dist=0.1):
        """Apply UMAP dimensionality reduction to the feature space."""
        print("Applying UMAP dimensionality reduction...")
        
        if self.features is None or self.features.shape[0] == 0:
            raise ValueError("No features to reduce. Run extract_features first.")
        
        # UMAP Dimension Reduction
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='euclidean',
            random_state=42
        )
        
        self.embedding = reducer.fit_transform(self.features)
        
        print(f"Reduced dimensions to {n_components} components with UMAP")
        
        return self
    
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import KMeans
    from matplotlib.colors import ListedColormap

    def visualize_umap(self):
        """Display the UMAP embedding directly in the notebook with points colored by flux."""
        if self.embedding is None:
            print("No UMAP embedding available. Run reduce_dimensions first.")
            return None
        
        print("Displaying UMAP visualization with flux-based coloring...")

        # Create figure - always use 2D visualization even if embedding is 3D
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get first flux image for coloring points
        if len(self.flux_images) > 0 and self.flux_images[0] is not None:
            # Get the flux values for all points in the feature space
            full_flux = self.flux_images[0].flatten()
            feature_flux = full_flux[self.mask_flat]
            
            # Create colors based on flux percentiles
            percentiles = [50, 65, 75, 82, 88, 93, 97]
            thresholds = [np.percentile(feature_flux, p) for p in percentiles]
            
            # Create a categorical color array based on these thresholds
            color_categories = np.zeros(len(feature_flux), dtype=int)
            color_categories[:] = 1
            for i, threshold in enumerate(thresholds, 2):
                color_categories[feature_flux >= threshold] = i

            # Define the discrete colormap for flux levels
            flux_colors = [
                (0.2, 0.2, 0.7),  # Level 1: deep blue
                (0.3, 0.4, 0.8),  # Level 2: blue
                (0.0, 0.7, 0.7),  # Level 3: cyan
                (0.0, 0.8, 0.4),  # Level 4: teal
                (0.9, 0.9, 0.0),  # Level 5: yellow
                (0.9, 0.6, 0.0),  # Level 6: orange
                (0.9, 0.1, 0.1)   # Level 7: red (highest flux)
            ]
            flux_cmap = ListedColormap(flux_colors)
            
            scatter = ax.scatter(
                self.embedding[:, 0],
                self.embedding[:, 1],
                c=color_categories,
                cmap=flux_cmap,
                vmin=1,
                vmax=7,  # Adjusted to go from 1 to 7 for flux categories
                s=25,  # Adjust point size for visibility
                alpha=0.9
            )
            
            # Add colorbar with the same labels as the flux map
            cbar = plt.colorbar(scatter, ticks=range(1, 8))  # Adjusted ticks range
            cbar.set_label('UMAP Clusters', fontsize=14, fontweight='bold')
            
        else:
            # No flux image available, color by cluster
            if self.labels is not None:
                ax.scatter(
                    self.embedding[:, 0],
                    self.embedding[:, 1],
                    c=self.labels,
                    cmap='tab20',
                    s=20,
                    alpha=0.8
                )
            else:
                # No labels either, just show points
                ax.scatter(
                    self.embedding[:, 0],
                    self.embedding[:, 1],
                    s=10,
                    alpha=0.8
                )

        # Title and labels
        ax.set_title("UMAP 2D Embedding", fontsize=16, fontweight='bold')
        ax.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
        ax.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()  # Display in the notebook
        return fig  # Return the figure object for further customization


    def map_umap_clusters_to_image(self):
        """Map both UMAP clusters and flux percentiles back to the original image space."""
        print("Mapping both UMAP clusters and flux percentiles to image space...")
        
        if self.embedding is None:
            print("No UMAP embedding available. Run reduce_dimensions first.")
            return None
        
        h, w = self.mask_shape
        umap_rgb_image = np.zeros((h, w, 3))
        flux_rgb_image = np.zeros((h, w, 3))
        
        if len(self.flux_images) > 0 and self.flux_images[0] is not None:
            full_flux = self.flux_images[0].flatten()
            feature_flux = full_flux[self.mask_flat]
            
            # Cluster UMAP embedding using KMeans
            n_clusters = 7
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.embedding)
            
            # Shift cluster labels to range from 1 to 7
            cluster_labels += 1  # Now the labels range from 1 to 7
            
            # Flux percentiles and colors
            percentiles = [50, 65, 75, 82, 88, 93, 97]
            thresholds = [np.percentile(feature_flux, p) for p in percentiles]
            
            flux_categories = np.ones(len(feature_flux), dtype=int)
            for i, threshold in enumerate(thresholds, 2):
                flux_categories[feature_flux >= threshold] = i
            
            flux_colors = [
                (0.2, 0.2, 0.7),  # Level 1: deep blue
                (0.3, 0.4, 0.8),  # Level 2: blue
                (0.0, 0.7, 0.7),  # Level 3: cyan
                (0.0, 0.8, 0.4),  # Level 4: teal
                (0.9, 0.9, 0.0),  # Level 5: yellow
                (0.9, 0.6, 0.0),  # Level 6: orange
                (0.9, 0.1, 0.1)   # Level 7: red (highest flux)
            ]
            
            # Cluster colors
            cluster_colors = flux_colors[:n_clusters]
            
            # Map from feature space to image space
            y_indices = np.round(self.features[:, 1] * h).astype(int)
            x_indices = np.round(self.features[:, 0] * w).astype(int)
            
            y_indices = np.clip(y_indices, 0, h-1)
            x_indices = np.clip(x_indices, 0, w-1)
            
            # Map clusters to image space
            for i, cluster_id in enumerate(cluster_labels):
                y, x = y_indices[i], x_indices[i]
                if 0 <= y < h and 0 <= x < w and 1 <= cluster_id <= len(cluster_colors):
                    umap_rgb_image[y, x] = cluster_colors[cluster_id - 1]  # Subtract 1 to index color correctly
            
            # Map flux percentiles to image space
            for i, flux_cat in enumerate(flux_categories):
                y, x = y_indices[i], x_indices[i]
                if 0 <= y < h and 0 <= x < w and 1 <= flux_cat <= len(flux_colors):
                    flux_rgb_image[y, x] = flux_colors[flux_cat - 1]
            
            # Create the plot
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            umap_img = axes[0].imshow(umap_rgb_image, origin='lower')
            axes[0].set_title('UMAP Clusters Mapped to Image Space', fontsize=16, fontweight='bold')
            
            flux_img = axes[1].imshow(flux_rgb_image, origin='lower')
            axes[1].set_title('Flux Percentiles Mapped to Image Space', fontsize=16, fontweight='bold')
            
            # Colorbars
            cluster_cmap = ListedColormap(cluster_colors)
            cluster_mappable = plt.cm.ScalarMappable(cmap=cluster_cmap)
            cluster_mappable.set_array(np.arange(1, n_clusters + 1))  # Adjust range for colorbar (1 to 7)
            cbar_umap = plt.colorbar(cluster_mappable, ax=axes[0])
            cbar_umap.set_label('UMAP Clusters', fontsize=14, fontweight='bold')
            cbar_umap.set_ticks(np.arange(1, n_clusters + 1))  # Ensure colorbar has correct ticks
            
            flux_cmap = ListedColormap(flux_colors)
            flux_mappable = plt.cm.ScalarMappable(cmap=flux_cmap)
            flux_mappable.set_array(np.arange(1, len(flux_colors) + 1))
            cbar_flux = plt.colorbar(flux_mappable, ax=axes[1])
            cbar_flux.set_label('Flux Percentiles', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            return umap_rgb_image, flux_rgb_image
        else:
            print("No flux image available for coloring")
            return None, None



### PCA Implemetation ###

def run_pipeline(fits_files, base_path):
    """Run the full pipeline with the provided FITS files."""
    pipeline = LensedGalaxyPipeline(fits_files, base_path)
    
    # Run the full pipeline
    pipeline.load_fits_data()
    pipeline.preprocess_images()
    pipeline.extract_features()
    
    # Use parameters that worked well for medium sensitivity
    # Force 2D embedding with UMAP
    pipeline.reduce_dimensions(n_components=2, n_neighbors=20, min_dist=0.07)
  
    # so we can color by flux levels
    pipeline.visualize_umap()
    pipeline.map_umap_clusters_to_image()
    
    return pipeline

from sklearn.decomposition import PCA

class LensedGalaxyPipelineBaseline(LensedGalaxyPipeline):
    def reduce_dimensions(self, n_components=2):
        """Apply PCA dimensionality reduction to the feature space (Pipeline B baseline)."""
        print("Applying PCA dimensionality reduction...")

        if self.features is None or self.features.shape[0] == 0:
            raise ValueError("No features to reduce. Run extract_features first.")

        # PCA Dimension Reduction
        pca = PCA(n_components=n_components)
        self.embedding = pca.fit_transform(self.features)

        print(f"Reduced dimensions to {n_components} components with PCA")

        return self

def run_pipeline_baseline(fits_files, base_path):
    """Run the baseline pipeline using PCA instead of UMAP."""
    pipeline = LensedGalaxyPipelineBaseline(fits_files, base_path)

    # Run the full pipeline
    pipeline.load_fits_data()
    pipeline.preprocess_images()
    pipeline.extract_features()

    # Apply PCA instead of UMAP
    pipeline.reduce_dimensions(n_components=2)

    # Visualization and mapping (still works the same way)
    pipeline.visualize_umap()  # Still uses the same visual method
    pipeline.map_umap_clusters_to_image()

    return pipeline

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class LensedGalaxyPipelineBaseline(LensedGalaxyPipeline):
    def reduce_dimensions(self, n_components=2, method="pca"):
        """Apply dimensionality reduction to the feature space using PCA or t-SNE."""
        print(f"Applying {method.upper()} dimensionality reduction...")

        if self.features is None or self.features.shape[0] == 0:
            raise ValueError("No features to reduce. Run extract_features first.")

        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        else:
            raise ValueError("Unsupported reduction method. Use 'pca' or 'tsne'.")

        self.embedding = reducer.fit_transform(self.features)

        print(f"Reduced dimensions to {n_components} components with {method.upper()}")

        return self

def run_pipeline_baseline(fits_files, base_path, method="pca"):
    """Run the baseline pipeline using PCA or t-SNE instead of UMAP."""
    pipeline = LensedGalaxyPipelineBaseline(fits_files, base_path)

    # Run the full pipeline
    pipeline.load_fits_data()
    pipeline.preprocess_images()
    pipeline.extract_features()

    # Apply dimensionality reduction
    pipeline.reduce_dimensions(n_components=2, method=method)

    # Visualization and mapping
    pipeline.visualize_umap()
    pipeline.map_umap_clusters_to_image()

    return pipeline

# Example usage
if __name__ == "__main__":
    fits_files = [
        'reduced_NGC 4449_r_20250516_081456 (1).fits'
    ]

    base_path = "/Users/alr/Desktop/FoFARCSAT/"

    umap_pipeline = run_pipeline(fits_files, base_path)

    print("All baseline pipelines complete!")
