

from astropy.io import fits
import numpy as np
import astroscrappy  # Library for detecting and removing cosmic rays

def reduce_science_frame(
    science_filename,
    median_bias_filename,
    median_flat_filename,
    median_dark_filename,
    reduced_science_filename="reduced_science.fits",
):
    """
    Reduce a raw science frame by performing bias subtraction, dark subtraction,
    flat-field correction, and cosmic ray removal.

    Parameters:
    - science_filename: path to the raw science FITS file
    - median_bias_filename: path to the median-combined bias frame
    - median_flat_filename: path to the median-combined and normalized flat frame
    - median_dark_filename: path to the median-combined dark frame (in units of e⁻/sec)
    - reduced_science_filename: filename to save the final reduced image (default: 'reduced_science.fits')

    Returns:
    - reduced_science: 2D numpy array of the reduced science frame
    """

    # Load the raw science image and its header
    with fits.open(science_filename) as sci_hdul:
        sci_data = sci_hdul[0].data.astype(float) # Convert data to float for precise arithmetic
        sci_header = sci_hdul[0].header # Read header for metadata
        exptime = sci_header.get("EXPTIME") # Exposure time fr scaling dark current

        # Ensure the exposure time actualy exists in the header
        if exptime is None:
            raise ValueError("Exposure time (EXPTIME) not found in science frame header.")

    # Load the calibration frames: bias, dark (per second), and flat
    with fits.open(median_bias_filename) as b:
        bias = b[0].data.astype(float)

    with fits.open(median_dark_filename) as d:
        dark = d[0].data.astype(float)

    with fits.open(median_flat_filename) as f:
        flat = f[0].data.astype(float)

    # === Cal Steps ===

    #Subtract the bias and the dark current (scaled by exposure time)
    calibrated = sci_data - bias - dark * exptime

    # Divide by the flat field to correct for pixel-to-pixel sensitivity
    # This can introduce NaNs or infs if flat contains zeros or near-zeros
    reduced_science = calibrated / flat

    # Clean up NaNs and infinite values
    reduced_science[np.isnan(reduced_science)] = 0
    reduced_science[np.isinf(reduced_science)] = 0

    # === Cosmic Ray Removal ===

    # astroscrappy to detect and remove cosmic rays from the reduced image
    cr_mask, cleaned = astroscrappy.detect_cosmics(reduced_science)
    reduced_science = cleaned  # Use the cleaned image for further analysis


    # Write the reduced science image to a new FITS file
    hdu = fits.PrimaryHDU(data=reduced_science, header=sci_header)
    hdu.writeto(reduced_science_filename, overwrite=True)

    return reduced_science
