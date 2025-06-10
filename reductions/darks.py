from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np

def create_median_dark(dark_list, bias_filename, median_dark_filename):

    # Load bias frame
    with fits.open(bias_filename) as bias_hdul:
        bias = bias_hdul[0].data.astype(np.float32)  # Convert to float32 for memory efficiency

    dark_pixel_arrays = []  # List to store each processed dark frame

    # Load and process each dark frame ===
    for filename in dark_list:
        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header

            # Retrieve exposure time from FITS header
            exptime = header.get("EXPTIME")
            if exptime is None:
                raise ValueError(f"EXPTIME missing in header of {filename}")

            # Subtract bias and divide by exposure time to get dark current per second
            dark_corrected = (data - bias) / exptime
            dark_pixel_arrays.append(dark_corrected)

    # Stack into a 3D array and apply sigma clipping
    # Shape: (n_frames, height, width)
    dark_stack = np.stack(dark_pixel_arrays, axis=0)

    # Apply sigma clipping along the stack axis (axis=0)
    clipped = sigma_clip(dark_stack, sigma=3, axis=0)

    # Compute the median dark frame from the clipped stack
    median_dark = np.ma.median(clipped, axis=0).filled(np.nan)  # Fill masked values with NaN

    # === Save the master dark frame to a FITS file ===
    hdu = fits.PrimaryHDU(data=median_dark.astype(np.float32))
    hdu.writeto(median_dark_filename, overwrite=True)

    return median_dark
