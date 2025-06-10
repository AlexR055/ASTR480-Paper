from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np
import matplotlib.pyplot as plt

def create_median_flat(flat_list, bias_filename, median_flat_filename, dark_filename=None):

    # Load bias frame
    with fits.open(bias_filename) as bias_hdul:
        bias_frame = bias_hdul[0].data.astype(np.float32)

    # load dark frame
    if dark_filename:
        with fits.open(dark_filename) as dark_hdul:
            dark_frame = dark_hdul[0].data.astype(np.float32)

    flat_arrays = []  # To store correctd flat images

    # process flat frames
    for filename in flat_list:
        with fits.open(filename) as hdul:
            flat_data = hdul[0].data.astype(np.float32)
            header = hdul[0].header

            # Retrieve exposure time from header
            exptime = header.get("EXPTIME")
            if exptime is None:
                raise ValueError(f"Exposure time not found in header of {filename}")

            # Subtract bias
            corrected = flat_data - bias_frame

            # subtract dark frame
            if dark_filename:
                corrected -= dark_frame * exptime

            flat_arrays.append(corrected)

    # === Stack all corrected flats and apply sigma clipping ===
    flat_stack = np.stack(flat_arrays)
    clipped = sigma_clip(flat_stack, sigma=3, axis=0)  # Pixel-wise clipping across the stack
    median_flat = np.ma.median(clipped, axis=0).filled(np.nan)

    # Normalize flat field to 1
    normalized_flat = median_flat / np.nanmedian(median_flat)

    hdu = fits.PrimaryHDU(data=normalized_flat)
    hdu.writeto(median_flat_filename, overwrite=True)

    return median_flat


def plot_flat(median_flat_filename, ouput_filename="median_flat.png", profile_ouput_filename="median_flat_profile.png"):
   

    # Load flat field
    with fits.open(median_flat_filename) as hdul:
        flat_data = hdul[0].data.astype(np.float32)

    # === Plot the 2D normalized flat field image ===
    plt.figure()
    plt.imshow(flat_data, cmap="gray", origin="lower", vmin=0.9, vmax=1.1)
    plt.colorbar(label='Normalized Intensity')
    plt.title("Normalized Flat Field")
    plt.savefig(ouput_filename)
    plt.close()

    # === Compute and plot the 1D profile along the X-axis (median across Y) ===
    profile = np.nanmedian(flat_data, axis=0)
    plt.figure()
    plt.plot(profile)
    plt.title("Median Flat Field Profile (Y-axis)")
    plt.xlabel("X Pixel")
    plt.ylabel("Median Intensity")
    plt.grid(True)
    plt.savefig(profile_ouput_filename)
    plt.close()
