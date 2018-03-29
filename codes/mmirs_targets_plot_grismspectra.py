from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

if __name__ == '__main__':

    """
    Only doing this for the two galaxies that are selected for MMIRS 
    and they also have data from PEARS, FIGS, and 3DHST.
    """

    # Alll in GOODS-N
    # Define ids
    pears_id1 = 120467
    pears_id2 = 116296

    figs_id1 = 400456
    figs_id2 = 400602

    # read in spectra
    pears_data_path = home + "/Documents/PEARS/data_spectra_only/"

    pears_spec1 = fits.open(pears_data_path + 'h_pears_n_id' + str(pears_id1) + '.fits')
    pears_spec1 = fits.open(pears_data_path + 'h_pears_n_id' + str(pears_id2) + '.fits')

    

	sys.exit(0)