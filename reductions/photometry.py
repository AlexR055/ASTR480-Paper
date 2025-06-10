

from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

def do_aperture_photometry(image, positions, radii, sky_radius_in, sky_annulus_width):

    # Load science image
    with fits.open(image) as hdul:
        data = hdul[0].data.astype(float)

    photometry_results = []

    # Loop over target positions
    for pos in positions:
        aperture_sums = []

        # For each aperture radius, compute the net flux after sky subtraction
        for r in radii:
            # Define the circular source aperture
            aperture = CircularAperture(pos, r)

            # Define a surrounding annulus for sky background estimation
            annulus = CircularAnnulus(pos, r_in=sky_radius_in, r_out=sky_radius_in + sky_annulus_width)

            # Perform aperture photometry on both the aperture and annulus
            aper_flux = aperture_photometry(data, aperture)
            sky_flux = aperture_photometry(data, annulus)

            # Compute average sky background level per pixel
            annulus_area = annulus.area
            mean_sky = sky_flux['aperture_sum'][0] / annulus_area

            # Estimate total sky contribution within the aperture
            aperture_area = aperture.area
            total_sky = mean_sky * aperture_area

            # Subtract sky contribution to get net flux
            net_flux = aper_flux['aperture_sum'][0] - total_sky
            aperture_sums.append(net_flux)

        # Store the radial profile results in an Astropy table
        result = Table()
        result['radius'] = radii
        result['net_flux'] = aperture_sums
        result['x'] = [pos[0]] * len(radii)
        result['y'] = [pos[1]] * len(radii)

        photometry_results.append(result)

    return photometry_results  # List of tables


def plot_radial_profile(aperture_photometry_data, sky_radius_in, output_filename="radial_profile.png"):

    plt.figure()

    # Plot the radial profile for each star
    for i, table in enumerate(aperture_photometry_data):
        label = f"Target {i+1} at ({table['x'][0]}, {table['y'][0]})"
        plt.plot(table['radius'], table['net_flux'], marker='o', label=label)

    # Add a vertical dashed line indicating where sky background is measured
    plt.axvline(sky_radius_in, color='gray', linestyle='--', label=f'Sky radius = {sky_radius_in}')
    plt.xlabel("Aperture Radius (pixels)")
    plt.ylabel("Net Flux (ADU)")
    plt.title("Radial Profile")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()
