#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Filename: reduction.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

def run_reduction():
    from bias import create_median_bias
    from darks import create_median_dark
    from flats import create_median_flat, plot_flat
    from science import reduce_science_frame
    from photometry import do_aperture_photometry, plot_radial_profile
    from astropy.io import fits
    import numpy as np

    print("Starting full CCD reduction...")

    # === Filenames ===
    bias_files = [f"Bias-S001-R001-C{str(i).zfill(3)}-NoFilt.fit" for i in range(1, 4)]
    dark_files = [f"Dark-S001-R001-C{str(i).zfill(3)}-NoFilt.fit" for i in range(1, 4)]
    flat_files = [f"AutoFlat-PANoRot-r-Bin1-{str(i).zfill(3)}.fit" for i in range(1, 4)]
    science_files = [
        "kelt-16-b-S001-R001-C084-r.fit",
        "kelt-16-b-S001-R001-C125-r.fit"
    ]

    # === Step 1: Create Median Bias Frame ===
    print("Creating median bias frame...")
    median_bias_filename = "median_bias.fits"
    create_median_bias(bias_files, median_bias_filename)

    # === Step 2: Create Median Dark Frame ===
    print("Creating median dark frame...")
    median_dark_filename = "median_dark.fits"
    create_median_dark(dark_files, median_bias_filename, median_dark_filename)

    # === Step 3: Create Median Flat Frame ===
    print("Creating median flat frame...")
    median_flat_filename = "median_flat.fits"
    create_median_flat(flat_files, median_bias_filename, median_flat_filename, median_dark_filename)

    print("Plotting flat frame...")
    plot_flat(median_flat_filename)

    # === Step 4: Reduce Science Frame ===
    for sci_file in science_files:
        print(f"Reducing science frame: {sci_file}")
        reduced_filename = f"reduced_{sci_file}"
        reduce_science_frame(
            sci_file,
            median_bias_filename,
            median_flat_filename,
            median_dark_filename,
            reduced_filename
        )

        # === Step 5: Aperture Photometry ===
        print(f"Performing aperture photometry on {reduced_filename}...")
        positions = [(150, 150), (300, 200)]  # You can change this
        radii = [2, 4, 6, 8, 10]
        sky_radius_in = 12
        sky_annulus_width = 4

        photometry_data = do_aperture_photometry(reduced_filename, positions, radii, sky_radius_in, sky_annulus_width)
        profile_filename = f"radial_profile_{sci_file.replace('.fit', '.png')}"
        plot_radial_profile(photometry_data, sky_radius_in, profile_filename)

    print("All reductions complete.")


