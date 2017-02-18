from __future__ import division

from astropy.convolution import convolve, convolve_fft
import numpy as np
import numpy.ma as ma
from astropy.io import fits

import os
import sys
import time
import glob

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/fits_comp_spectra/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
new_codes_dir = home + "/Desktop/FIGS/new_codes/"
lsf_dir = new_codes_dir

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
import grid_coadd as gd
import fast_chi2_jackknife as fcj
from fast_chi2_jackknife_massive_galaxies import create_bc03_lib_main

if __name__ == '__main__':
	
	# Read the matched galaxies catalog between 3DHST and PEARS
    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt', dtype=None, names=True, skip_header=2)
    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    # Find the indices for massive galaxies
    massive_galaxies_indices = np.where(stellarmass >= 11)[0]

    # Only use the galaxy ids that are unique
    pears_unique_ids, pears_unique_ids_indices = np.unique(pears_id[massive_galaxies_indices], return_index=True)
 
    # Pick a random galaxy
    current_pears_index = pears_unique_ids[4]
    count = pears_unique_ids_indices[4]
    
    # Find the redshift and other rest frame quants for the chosen galaxy
    redshift = photz[massive_galaxies_indices][count]
    lam_em, flam_em, ferr, specname = gd.fileprep(current_pears_index, redshift)

    lsf_south_path = new_codes_dir + 'pears_lsfs/south_lsfs/'

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for lsffile in glob.glob(lsf_south_path + '*.txt'):

        lsf = np.loadtxt(lsffile)

        if 'avg' in lsffile:
            ax.plot(np.arange(len(lsf)), lsf, color ='k', linewidth=1, zorder=10)
        else:
            ax.plot(np.arange(len(lsf)), lsf, color ='b')

    plt.show()

    sys.exit(0)

