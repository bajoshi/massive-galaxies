from __future__ import division

import numpy as np

import os
import sys

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"

sys.path.append(massive_galaxies_dir)
import matching as mt

def read_pears_cats():

    # Read PEARS cats
    pears_master_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))
    pears_master_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['dec'] = pears_ncat['dec'] - dec_offset_goodsn_v19

    # Read PEARS broadband cat
    names_header = ['PID', 'ID', 'MAG_AUTO_b', 'MAG_AUTO_v', 'MAG_AUTO_i', 'MAG_AUTO_z', 'MAGERR_AUTO_b',\
     'MAGERR_AUTO_v', 'MAGERR_AUTO_i', 'MAGERR_AUTO_z', 'ALPHA_J2000', 'DELTA_J2000']
    # I can't tell if I should use ID or PID
    
    north_pears_cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/n_biz_bviz_all.pid.txt',\
     dtype=None, names=names_header, skip_header=365, usecols=(362, 279, 158, 164, 162, 166, 93, 91, 92, 89, 358, 248), delimiter=' ')
    south_pears_cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/s_biz_bviz_all.pid.txt',\
     dtype=None, names=names_header, skip_header=365, usecols=(362, 279, 158, 164, 162, 166, 93, 91, 92, 89, 358, 248), delimiter=' ')

    north_pears_cat['DELTA_J2000'] = north_pears_cat['DELTA_J2000'] - dec_offset_goodsn_v19
    # I'm assuming there is the same offset in this broadband photometry catalog as well. Applying the correction for now.

    return north_pears_cat, south_pears_cat, pears_master_ncat, pears_master_scat

def match_pears_broadband_master():

    return None

def k_corr():
    """
    This func converts broadband observed frame mags to rest frame mags.
    """

    return None

if __name__ == '__main__':
    
    pears_broadband_ncat, pears_broadband_scat, pears_ncat, pears_scat = read_pears_cats()

    

    sys.exit(0)

