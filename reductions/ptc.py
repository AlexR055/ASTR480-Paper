#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: ptc.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from astropy.io import fits
import numpy as np

def calculate_gain(files):

    if len(files) != 2:
        raise ValueError("Two flat-field images are required to calculate gain.")

    #Load the two flat-field frames
    with fits.open(files[0]) as hdul1, fits.open(files[1]) as hdul2:
        flat1 = hdul1[0].data.astype(float)
        flat2 = hdul2[0].data.astype(float)

    # Compute mean signal levels in both flats
    mean1 = np.mean(flat1)
    mean2 = np.mean(flat2)

    # Compute variance of the difference image
    # The variance of (flat1 - flat2) = 2 * variance(signal), so divide by 2
    diff = flat1 - flat2
    variance = np.var(diff) / 2

    # Calculate gain: G = (mean1 + mean2) / variance(diff) 
    gain = (mean1 + mean2) / variance

    return gain  # Units: electrons per ADU (e-/ADU) I guess?


def calculate_readout_noise(files, gain):
    """
    Estimate the readout noise of a CCD (in electrons) using two bias frames.

    Parameters:
    - files: list of two bias frame FITS file paths
    - gain: CCD gain in e-/ADU, as computed from calculate_gain

    Returns:
    - readout_noise: estimated read noise in electrons (e-)
    """

    if len(files) != 2:
        raise ValueError("Two bias frames are required to calculate readout noise.")

    # Load the two bias frames
    with fits.open(files[0]) as hdul1, fits.open(files[1]) as hdul2:
        bias1 = hdul1[0].data.astype(float)
        bias2 = hdul2[0].data.astype(float)

    # Compute standard deviation of pixel-wise difference
    # For two images of the same distribution: var(diff) = 2 * read_noise^2
    # So std(diff) / sqrt(2) = read_noise (in ADU), then convert to electrons
    diff = bias1 - bias2
    stddev = np.std(diff) / np.sqrt(2)

    # Convert from ADU to electrons using the gain, might be overly complicating things
    readout_noise = gain * stddev

    return readout_noise
