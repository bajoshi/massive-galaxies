from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys
import glob

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

def read_pears_broadband_cats():

    # Read PEARS broadband cat
    names_header = ['PID', 'ID', 'ALPHA_J2000', 'DELTA_J2000']
    # I can't tell if I should use ID or PID
    # reading in both for now

    pears_n_cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/n_biz_bviz_all.pid.txt', dtype=None, names=names_header, skip_header=365, usecols=(362, 279, 358, 248), delimiter=' ')
    pears_s_cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/s_biz_bviz_all.pid.txt', dtype=None, names=names_header, skip_header=365, usecols=(362, 279, 358, 248), delimiter=' ')

    return pears_n_cat, pears_s_cat

if __name__ == '__main__':
	
    lsf_path = massive_galaxies_dir + "north_lsfs/"

    pears_n_phot_cat, pears_s_phot_cat = read_pears_broadband_cats()

    print pears_n_phot_cat['ID']
    print pears_n_phot_cat['PID']

    sys.exit(0)