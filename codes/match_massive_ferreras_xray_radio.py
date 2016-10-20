from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys
import time
import datetime

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

sys.path.append(massive_galaxies_dir)
import matching as mt

def get_pears_fitsfile(pearsid):

	data_path = home + "/Documents/PEARS/data_spectra_only/"
    # Get the correct filename and the number of extensions
    filename = data_path + 'h_pears_n_id' + str(pearsid) + '.fits'
    if not os.path.isfile(filename):
        filename = data_path + 'h_pears_s_id' + str(pearsid) + '.fits'

    fitsfile = fits.open(filename)

    return fitsfile

def find_matches_in_ferreras2009(pears_cat, ferreras_prop_cat, ferreras_cat, indices):

    pears_ids = pears_cat['pearsid']
    ferreras_ids = ferreras_cat['id']

    count_matched = 0
    count_notmatched = 0
    for i in range(len(pears_ids[indices])):
        if pears_ids[indices][i] in ferreras_ids:
            ferreras_ind = np.where(ferreras_ids == pears_ids[indices][i])[0]
            print ferreras_cat['ra'][ferreras_ind], ',', ferreras_cat['dec'][ferreras_ind], ferreras_ids[ferreras_ind], pears_ids[indices][i], "matched."
            #print pears_cat['mstar'][indices][i], ferreras_prop_cat['mstar'][ferreras_ind]  # compare stellarmass
            #print pears_cat['threedzphot'][indices][i], ferreras_cat['z'][ferreras_ind]  # compare redshifts
            count_matched += 1
        else:
            #print fitsfile[0].header['ra'], ',', fitsfile[0].header['dec'], "for", pears_ids[indices][i], "did not match."
            count_notmatched += 1

    print count_matched, "galaxies matched"
    print count_notmatched, "galaxies did not match"

    return None

if __name__ == '__main__':
	
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # Read pears + 3dhst catalog
    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt', dtype=None, names=True, skip_header=2)

    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    # Find indices for massive galaxies
    massive_galaxies_indices = np.where(stellarmass >= 10.5)[0]

    # Get the unique ids
    pears_unique_ids, pears_unique_ids_indices = np.unique(pears_id[massive_galaxies_indices], return_index=True)

    # Match with Ferreras et al. 2009
    ferreras_cat = np.genfromtxt(massive_galaxies_dir + 'ferreras_2009_ETG_cat.txt', dtype=None,\
                                 names=['id', 'ra', 'dec', 'z'], usecols=(0,1,2,5), skip_header=23)
    ferreras_prop_cat = np.genfromtxt(massive_galaxies_dir + 'ferreras_2009_ETG_prop_cat.txt', dtype=None,\
                                 names=['id', 'mstar'], usecols=(0,1), skip_header=23)    
    
    find_matches_in_ferreras2009(cat, ferreras_prop_cat, ferreras_cat, massive_galaxies_indices)

    sys.exit(0)
