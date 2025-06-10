import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

def create_median_bias(bias_list, median_bias_filename):

    bias_arrays = []  # List to store 2D arrays from each bias FITS file

    # Load each bias frame and convert it to float32 (saves a bit of memory for the frail jupyterhub)
    for filename in bias_list:
        with fits.open(filename) as hdul:
            bias_data = hdul[0].data.astype(np.float32)
            bias_arrays.append(bias_data)  # Store the array in the list

    # Stack all bias images into a 3D array
    bias_stack = np.stack(bias_arrays, axis=0)

    # Perform sigma clipping along the stack to remove outlier pixels
    # Clipping is applied pixel-wise across all frames
    bias_images_masked = sigma_clip(bias_stack, cenfunc='median', sigma=3, axis=0)

    # Compute the median of the sigma-clipped stack to produce the final bias
    median_bias = np.ma.median(bias_images_masked, axis=0)  # 2D masked array

    # Create a new FITS object
    hdu = fits.PrimaryHDU(median_bias.filled())  #Fill masked pixels with default (median) values
    hdu.writeto(median_bias_filename, overwrite=True)
    
    return median_bias.filled()

