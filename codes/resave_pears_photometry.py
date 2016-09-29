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
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/fits_comp_spectra/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd

if __name__ == '__main__':
    
    # Open the PEARS broadband data files
    # I can't tell if I should use ID or PID
    # reading in both for now
    names_header = ['PID', 'ID', 'MAG_AUTO_b', 'MAG_AUTO_v', 'MAG_AUTO_i', 'MAG_AUTO_z',\
     'MAGERR_AUTO_b', 'MAGERR_AUTO_v', 'MAGERR_AUTO_i', 'MAGERR_AUTO_z', 'ALPHA_J2000', 'DELTA_J2000']
    
    north_pears_cat = np.genfromtxt('/Volumes/Bhavins_backup/PEARS/n_biz_bviz_all.pid.txt',\
     dtype=None, names=names_header, skip_header=365, usecols=(362, 279, 158, 164, 162, 166, 93, 91, 92, 89, 358, 248), delimiter=' ')
    south_pears_cat = np.genfromtxt('/Volumes/Bhavins_backup/PEARS/s_biz_bviz_all.pid.txt',\
     dtype=None, names=names_header, skip_header=365, usecols=(362, 279, 158, 164, 162, 166, 93, 91, 92, 89, 358, 248), delimiter=' ')

    # Read pears catalog
    data_path = home + "/Documents/PEARS/data_spectra_only/"
    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt',
                       dtype=None, names=True, skip_header=2)

    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    # Find indices for massive galaxies
    massive_galaxies_indices = np.where(stellarmass >= 10.5)[0] 

    # Read PEARS spectrum and the galaxy's coordinates
    for u in range(len(pears_id[massive_galaxies_indices])):

        pearsid = pears_id[massive_galaxies_indices][u]

        # Get the correct filename and the number of extensions
        skip_ids = [41913, 88679, 103918]
        # Fix the 3dhst and pears matching/saving before the follwing condition is takem out.
        # These ids exist in both the south and north catalogs so I'm discarding them for now.
        if pearsid in skip_ids:
            continue

        filename = data_path + 'h_pears_n_id' + str(pearsid) + '.fits'
        if not os.path.isfile(filename):
            filename = data_path + 'h_pears_s_id' + str(pearsid) + '.fits'
            field = 'south'
        else:
            field = 'north'

        fitsfile = fits.open(filename)

        ra = fitsfile[0].header['RA']
        dec = fitsfile[0].header['DEC']

        if field == 'north':
            print ra, dec
            print np.where(np.isclose(north_pears_cat['ALPHA_J2000'], ra, rtol=1e-7, atol=1e-8))[0]
            sys.exit(0)



        #    idx = np.where((north_pears_cat['ALPHA_J2000'] == ra) & (north_pears_cat['DELTA_J2000'] == dec))[0]
        #    print pearsid, idx, ra, north_pears_cat['ALPHA_J2000'][idx], dec, north_pears_cat['DELTA_J2000'][idx]
        #elif field == 'south':
        #    idx = np.where((south_pears_cat['ALPHA_J2000'] == ra) & (south_pears_cat['DELTA_J2000'] == dec))[0]
        #    print pearsid, idx, ra, south_pears_cat['ALPHA_J2000'][idx], dec, south_pears_cat['DELTA_J2000'][idx]









