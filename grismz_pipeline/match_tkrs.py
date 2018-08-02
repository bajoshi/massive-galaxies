from __future__ import division

import numpy as np
from astropy.io import fits
import math

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

def convert_to_deg(coord_arr, coordtype):

    for i in range(coord_arr):

        if coordtype == 'ra':
            ra_hour_frac, ra_hour = math.modf(coord_arr[i])
            

    return conv_list

if __name__ == '__main__':

    # Read TKRS catalog
    cat = fits.open(home + '/Desktop/FIGS/tkrs_spectra/tkrs_by_ra.fits')

    # assign arrays
    tkrs_ra_ = cat[1].data['ra']
    tkrs_dec_ = cat[1].data['dec']

    # Convert to degrees
    convert_to_deg(tkrs_ra_, 'ra')

    sys.exit(0)