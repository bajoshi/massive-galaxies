from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

sys.path.append(massive_galaxies_dir)
import matching as mt

def get_pears_fitsfile(pearsid, field):

    data_path = home + "/Documents/PEARS/data_spectra_only/"
    # Get the correct filename
    if field == 'GOODS-N':
        filename = data_path + 'h_pears_n_id' + str(pearsid) + '.fits'
    elif field == 'GOODS-S':
        filename = data_path + 'h_pears_s_id' + str(pearsid) + '.fits'

    fitsfile = fits.open(filename)

    return fitsfile

def find_matches_in_ferreras2009(pears_cat, ferreras_prop_cat, ferreras_cat, massiveindices, massive_unique_indices):

    pears_ids = pears_cat['pearsid']
    ferreras_ids = ferreras_cat['id']
    field = pears_cat['field']

    indices_match = []

    count_matched = 0
    count_notmatched = 0
    for i in range(len(pears_ids[massiveindices][massive_unique_indices])):
        if pears_ids[massiveindices][massive_unique_indices][i] in ferreras_ids:
            ferreras_ind = np.where(ferreras_ids == pears_ids[massiveindices][massive_unique_indices][i])[0]
            #print ferreras_cat['ra'][ferreras_ind], ',', ferreras_cat['dec'][ferreras_ind], ferreras_ids[ferreras_ind], pears_ids[massiveindices][massive_unique_indices][i], "matched."
            #print pears_cat['mstar'][massiveindices][massive_unique_indices][i], ferreras_prop_cat['mstar'][ferreras_ind]  # compare stellarmass
            #print pears_cat['threedzphot'][massiveindices][massive_unique_indices][i], ferreras_cat['z'][ferreras_ind]  # compare redshifts
            indices_match.append(massiveindices[massive_unique_indices][i])
            count_matched += 1
        else:
            fitsfile = get_pears_fitsfile(pears_ids[massiveindices][massive_unique_indices][i], field[massiveindices][massive_unique_indices][i])
            #print fitsfile[0].header['ra'], ',', fitsfile[0].header['dec'], "for", pears_ids[massiveindices][massive_unique_indices][i], "did not match."
            count_notmatched += 1
            fitsfile.close()

    print count_matched, "galaxies matched"
    print count_notmatched, "galaxies did not match"

    return indices_match

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # Read pears + 3dhst catalog
    cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/color_stellarmass.txt', dtype=None, names=True, skip_header=2)

    pears_id = cat['pearsid']
    field = cat['field']
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
    
    indices_match = find_matches_in_ferreras2009(cat, ferreras_prop_cat, ferreras_cat, massive_galaxies_indices, pears_unique_ids_indices)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ax.plot(stellarmass, ur_color, '.', markersize=5, color='k')
    ax.plot(stellarmass[massive_galaxies_indices], ur_color[massive_galaxies_indices], '.', markersize=5, color='k')
    ax.plot(stellarmass[indices_match], ur_color[indices_match], 'o', markersize=10, mfc='None', mec='r')

    ax.set_ylim(1.4,2.8)
    #ax.set_xlim(7,12)

    plt.show()
    plt.close()

    # Read in Xray and radio catalogs
    # GOODS-S
    goodss_chandra = np.genfromtxt(home + '/Desktop/FIGS/x_ray_radio_sources_figs/goodss_chandra.txt',\
                                   dtype=None, usecols=(2,3), names=['ra','dec'], delimiter='  ')
    goodss_radio_miller = np.genfromtxt(home + '/Desktop/FIGS/x_ray_radio_sources_figs/miller2013ApJS205.13_GOODSS_radio.txt',\
                                        dtype=None, names=['ra','dec'], usecols=(1,2), skip_header=11, skip_footer=2, delimiter='  ')
    goodss_radio_afonso = np.genfromtxt(home + '/Desktop/FIGS/x_ray_radio_sources_figs/goodss_radio_afonso.txt',\
                                        dtype=None, names=['ra','dec'], usecols=(1,2))
    # GOODS-N
    goodsn_chandra = np.genfromtxt(home + '/Desktop/FIGS/x_ray_radio_sources_figs/alexander2003AJ126.539_GOODSN_chandra.txt',\
                                   dtype=None, names=['ra','dec'], skip_header=72, usecols=(0,1), delimiter='  ')
    goodsn_radio = np.genfromtxt(home + '/Desktop/FIGS/x_ray_radio_sources_figs/morrison2010ApJS188.178_GOODSN_radio.txt',\
                                 dtype=None, names=['hr','min','sec','deg','arcmin','arcsec'], usecols=(1,2,3,5,6,7), skip_header=33)


    # make ra dec lists for all catalogs
    

    sys.exit(0)
