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
save_dir = home + "/Desktop/FIGS/new_codes/"
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

    # Create empty lists to be containers for holding data on the matched pairs
    ids = []
    master_ra = []
    master_dec = []
    broadband_cat_ra = []
    broadband_cat_dec = []
    broadband_cat_bmag = []
    broadband_cat_bmagerr = []
    broadband_cat_vmag = []
    broadband_cat_vmagerr = []
    broadband_cat_imag = []
    broadband_cat_imagerr = []
    broadband_cat_zmag = []
    broadband_cat_zmagerr = []

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
            idx = np.where(np.isclose(north_pears_cat['ALPHA_J2000'], ra, rtol=1e-5, atol=1e-5) & np.isclose(north_pears_cat['DELTA_J2000'], dec, rtol=1e-5, atol=1e-5))[0]
            ang_dist = np.sqrt((north_pears_cat['ALPHA_J2000'][idx] - ra)**2 + (north_pears_cat['DELTA_J2000'][idx] - dec)**2)
            match_idx = np.argmin(ang_dist)
            #print pearsid, match_idx, ra, north_pears_cat['ALPHA_J2000'][idx][match_idx], dec, north_pears_cat['DELTA_J2000'][idx][match_idx]

            ids.append(pearsid)
            master_ra.append(ra)
            master_dec.append(dec)
            broadband_cat_ra.append(north_pears_cat['ALPHA_J2000'][idx][match_idx])
            broadband_cat_dec.append(north_pears_cat['DELTA_J2000'][idx][match_idx])
            broadband_cat_bmag.append(north_pears_cat['MAG_AUTO_b'][idx][match_idx])
            broadband_cat_bmagerr.append(north_pears_cat['MAGERR_AUTO_b'][idx][match_idx])
            broadband_cat_vmag.append(north_pears_cat['MAG_AUTO_v'][idx][match_idx])
            broadband_cat_vmagerr.append(north_pears_cat['MAGERR_AUTO_v'][idx][match_idx])
            broadband_cat_imag.append(north_pears_cat['MAG_AUTO_i'][idx][match_idx])
            broadband_cat_imagerr.append(north_pears_cat['MAGERR_AUTO_i'][idx][match_idx])
            broadband_cat_zmag.append(north_pears_cat['MAG_AUTO_z'][idx][match_idx])
            broadband_cat_zmagerr.append(north_pears_cat['MAGERR_AUTO_z'][idx][match_idx])

        elif field == 'south':
            idx = np.where(np.isclose(south_pears_cat['ALPHA_J2000'], ra, rtol=1e-5, atol=1e-5) & np.isclose(south_pears_cat['DELTA_J2000'], dec, rtol=1e-5, atol=1e-5))[0]
            ang_dist = np.sqrt((south_pears_cat['ALPHA_J2000'][idx] - ra)**2 + (south_pears_cat['DELTA_J2000'][idx] - dec)**2)
            match_idx = np.argmin(ang_dist)            
            #print pearsid, match_idx, ra, south_pears_cat['ALPHA_J2000'][idx][match_idx], dec, south_pears_cat['DELTA_J2000'][idx][match_idx]

            ids.append(pearsid)
            master_ra.append(ra)
            master_dec.append(dec)
            broadband_cat_ra.append(south_pears_cat['ALPHA_J2000'][idx][match_idx])
            broadband_cat_dec.append(south_pears_cat['DELTA_J2000'][idx][match_idx])
            broadband_cat_bmag.append(south_pears_cat['MAG_AUTO_b'][idx][match_idx])
            broadband_cat_bmagerr.append(south_pears_cat['MAGERR_AUTO_b'][idx][match_idx])
            broadband_cat_vmag.append(south_pears_cat['MAG_AUTO_v'][idx][match_idx])
            broadband_cat_vmagerr.append(south_pears_cat['MAGERR_AUTO_v'][idx][match_idx])
            broadband_cat_imag.append(south_pears_cat['MAG_AUTO_i'][idx][match_idx])
            broadband_cat_imagerr.append(south_pears_cat['MAGERR_AUTO_i'][idx][match_idx])
            broadband_cat_zmag.append(south_pears_cat['MAG_AUTO_z'][idx][match_idx])
            broadband_cat_zmagerr.append(south_pears_cat['MAGERR_AUTO_z'][idx][match_idx])
    
    # Convert lists to arrays for saving to txt file
    ids = np.asarray(ids)
    master_ra = np.asarray(master_ra)
    master_dec = np.asarray(master_dec)
    broadband_cat_ra = np.asarray(broadband_cat_ra)
    broadband_cat_dec = np.asarray(broadband_cat_dec)
    broadband_cat_bmag = np.asarray(broadband_cat_bmag)
    broadband_cat_bmagerr = np.asarray(broadband_cat_bmagerr)
    broadband_cat_vmag = np.asarray(broadband_cat_vmag)
    broadband_cat_vmagerr = np.asarray(broadband_cat_vmagerr)
    broadband_cat_imag = np.asarray(broadband_cat_imag)
    broadband_cat_imagerr = np.asarray(broadband_cat_imagerr)
    broadband_cat_zmag = np.asarray(broadband_cat_zmag)
    broadband_cat_zmagerr = np.asarray(broadband_cat_zmagerr)

    print len(ids)

    # Save to a txt file
    data = np.array(zip(ids, master_ra, master_dec, broadband_cat_ra, broadband_cat_dec,\
     broadband_cat_bmag, broadband_cat_bmagerr, broadband_cat_vmag, broadband_cat_vmagerr,\
      broadband_cat_imag, broadband_cat_imagerr, broadband_cat_zmag, broadband_cat_zmagerr),\
                    dtype=[('ids', int), ('master_ra', float), ('master_dec', float),\
                     ('broadband_cat_ra', float), ('broadband_cat_dec', float), ('broadband_cat_bmag', float), ('broadband_cat_bmagerr', float),\
                     ('broadband_cat_vmag', float), ('broadband_cat_vmagerr', float), ('broadband_cat_imag', float), ('broadband_cat_imagerr', float),\
                     ('broadband_cat_zmag', float), ('broadband_cat_zmagerr', float)])
    np.savetxt(save_dir + 'pears_broadband_massive_galaxies.txt', data, fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f'], delimiter=' ',\
               header='Broadband data from matches with master broadband cat. This is for galaxies with M > 10^10.5 M_sol.' + '\n' + \
               'ids, master_ra, master_dec, broadband_cat_ra, broadband_cat_dec, broadband_cat_bmag, broadband_cat_bmagerr, broadband_cat_vmag, broadband_cat_vmagerr, broadband_cat_imag, broadband_cat_imagerr, broadband_cat_zmag, broadband_cat_zmagerr')

    # Make regions file from matches and confirm by eye







