from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

sys.path.append(massive_galaxies_dir)
import matching as mt
from pears_and_3dhst import read_3dhst_cats

if __name__ == '__main__':
    
    # Read 3dhst cats
    threed_ncat, threed_scat, threed_v41_phot = read_3dhst_cats()

    # Read PEARS cats
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))
    
    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['dec'] = pears_ncat['dec'] - dec_offset_goodsn_v19

    # read santini et al. 2015 cat
    # this is only in GOODS-S
    names_header=['id', 'ra', 'dec', 'zbest', 'zphot', 'zphot_l68', 'zphot_u68']
    santini = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/santini_candels_cat.txt',\
     names=names_header, usecols=(0,1,2,9,13,14,15), skip_header=187)
    
    # Create arrays for id, ra, and dec for all catalogs
    threed_n_ra = threed_ncat[1].data['ra']
    threed_n_dec = threed_ncat[1].data['dec']
    threed_n_id = threed_ncat[1].data['id']

    threed_s_ra = threed_scat[1].data['ra']
    threed_s_dec = threed_scat[1].data['dec']
    threed_s_id = threed_scat[1].data['id']

    pears_n_ra = pears_ncat['ra']
    pears_n_dec = pears_ncat['dec']
    pears_n_id = pears_ncat['id']

    pears_s_ra = pears_scat['ra']
    pears_s_dec = pears_scat['dec']
    pears_s_id = pears_scat['id']

    santini_s_ra = santini['ra']
    santini_s_dec = santini['dec']
    santini_s_id = santini['id']
    santini_s_z = santini['zphot']

    # Run the matching function
    # PEARS North
    deltaRA, deltaDEC, threed_n_ra_matches, threed_n_dec_matches, pears_n_ra_matches, pears_n_dec_matches, threed_n_ind, pears_n_ind, num_single_matches = \
    mt.match(threed_n_ra, threed_n_dec, pears_n_ra, pears_n_dec, lim=0.3*1/3600)

    print "There were", num_single_matches, "single matches found out of", len(pears_ncat), "objects in the PEARS NORTH catalog with 3DHST."
    mt.plot_diff(deltaRA, deltaDEC, name='pears_threed_goodsn')

    # PEARS South
    # with 3DHST
    deltaRA, deltaDEC, threed_s_ra_matches, threed_s_dec_matches, pears_s_ra_matches_3d, pears_s_dec_matches_3d, threed_s_ind, pears_s_ind_3d, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, pears_s_ra, pears_s_dec, lim=0.3*1/3600)

    print "There were", num_single_matches, "single matches found out of", len(pears_scat), "objects in the PEARS SOUTH catalog with 3DHST."
    mt.plot_diff(deltaRA, deltaDEC, name='pears_threed_goodss')

    # with Santini
    deltaRA, deltaDEC, santini_s_ra_matches, santini_s_dec_matches, pears_s_ra_matches_san, pears_s_dec_matches_san, santini_s_ind, pears_s_ind_san, num_single_matches = \
    mt.match(santini_s_ra, santini_s_dec, pears_s_ra, pears_s_dec, lim=0.3*1/3600)

    print "There were", num_single_matches, "single matches found out of", len(pears_scat), "objects in the PEARS SOUTH catalog with Santini et al. 2015."
    mt.plot_diff(deltaRA, deltaDEC, name='pears_santini_goodss')

    plt.show()
    sys.exit(0)

    # save the matched file
    # North
    # get 3dhst north redshift
    threed_n_z = []
    for i in threed_n_ind:
        threed_n_z_indx = np.where((threed_v41_phot[1].data['id'] == threed_n_id[i]) & (threed_v41_phot[1].data['field'] == 'GOODS-N'))[0]
        threed_n_z.append(threed_v41_phot[1].data['z_peak'][threed_n_z_indx])

    pears_n_id_towrite = np.asarray(pears_n_id[pears_n_ind])
    pears_n_ra_towrite = np.asarray(pears_n_ra_matches)
    pears_n_dec_towrite = np.asarray(pears_n_dec_matches)

    match_id_towrite = np.asarray(threed_n_id[threed_n_ind])
    match_ra_towrite = np.asarray(threed_n_ra_matches)
    match_dec_towrite = np.asarray(threed_n_dec_matches)

    # i thought this was the easiest/cleanest way of creating a numpy array with all elements filled in with the same string
    match_source = np.empty(len(threed_n_ra_matches), dtype=object)
    for i in range(len(match_source)):
        match_source[i] = ' 3DHST '
    match_source_towrite = np.asarray(match_source)

    redshift_towrite = np.asarray(threed_n_z)
    redshift_towrite = redshift_towrite.flatten()
    redshift_lerr_towrite = np.ones(len(threed_n_ra_matches)) * -99.0
    redshift_uerr_towrite = np.ones(len(threed_n_ra_matches)) * -99.0

    data = np.array(zip(pears_n_id_towrite, pears_n_ra_towrite, pears_n_dec_towrite, match_id_towrite,\
     match_ra_towrite, match_dec_towrite, match_source_towrite, redshift_towrite, redshift_lerr_towrite, redshift_uerr_towrite),\
                    dtype=[('pears_n_id_towrite', int), ('pears_n_ra_towrite', float), ('pears_n_dec_towrite', float),\
                     ('match_id_towrite', int), ('match_ra_towrite', float), ('match_dec_towrite', float), ('match_source_towrite', '|S7'),\
                     ('redshift_towrite', float), ('redshift_lerr_towrite', float), ('redshift_uerr_towrite', float)])

    np.savetxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', data, fmt=['%d', '%.6f', '%.6f', '%d', '%.6f', '%.6f', '%s', '%.4f', '%.4f', '%.4f'],\
     delimiter=' ', header='These galaxies are in the GOODS-N field.' + '\n' + 'pearsid pearsra pearsdec matchid matchra matchdec source zphot zphot_l68 zphot_u68')

    # match south separately again to get one file to write
    pears_s_id_towrite = []
    pears_s_ra_towrite = []
    pears_s_dec_towrite = []
    match_id_towrite = []
    match_ra_towrite = []
    match_dec_towrite = []
    match_source_towrite = []
    redshift_towrite = []
    redshift_lerr_towrite = []
    redshift_uerr_towrite = []

    lim = 0.3*1/3600
    for i in range(len(pears_s_ra)):

        current_pears_ra = pears_s_ra[i]
        current_pears_dec = pears_s_dec[i]
        current_pears_id = pears_s_id[i]
 
        # check catalogs for matches
        index_array_santini = np.where((santini_s_ra <= current_pears_ra + lim) & (santini_s_ra >= current_pears_ra - lim)\
         & (santini_s_dec <= current_pears_dec + lim) & (santini_s_dec >= current_pears_dec - lim))[0]

        index_array_3d = np.where((threed_s_ra <= current_pears_ra + lim) & (threed_s_ra >= current_pears_ra - lim)\
         & (threed_s_dec <= current_pears_dec + lim) & (threed_s_dec >= current_pears_dec - lim))[0]

        # the next if block checks if there are matches in santini first and then in 3dhst
        # if there is a match in santini then it uses that by default. ONly uses 3dhst if 
        # no match is found in santini
        # if no match is found in either catalog then it moves to the next object.
        if index_array_santini.size:
            if len(index_array_santini) == 1:
                pears_s_id_towrite.append(current_pears_id)
                pears_s_ra_towrite.append(current_pears_ra)
                pears_s_dec_towrite.append(current_pears_dec)
                match_id_towrite.append(santini_s_id[index_array_santini])
                match_ra_towrite.append(santini_s_ra[index_array_santini])
                match_dec_towrite.append(santini_s_dec[index_array_santini])
                match_source_towrite.append('CANDELS')
                redshift_towrite.append(santini_s_z[index_array_santini])
                redshift_lerr_towrite.append(santini['zphot_l68'][index_array_santini])
                redshift_uerr_towrite.append(santini['zphot_u68'][index_array_santini])
            else:
                # Not dealing with multiple matches for now
                continue
        elif index_array_3d.size:
            if len(index_array_3d) == 1:
                pears_s_id_towrite.append(current_pears_id)
                pears_s_ra_towrite.append(current_pears_ra)
                pears_s_dec_towrite.append(current_pears_dec)
                match_id_towrite.append(threed_s_id[index_array_3d])
                match_ra_towrite.append(threed_s_ra[index_array_3d])
                match_dec_towrite.append(threed_s_dec[index_array_3d])
                match_source_towrite.append(' 3DHST ')
                threed_z_indx = np.where((threed_v41_phot[1].data['id'] == threed_s_id[index_array_3d]) & (threed_v41_phot[1].data['field'] == 'GOODS-S'))[0]
                threed_z = threed_v41_phot[1].data['z_peak'][threed_z_indx]
                redshift_towrite.append(threed_z)
                redshift_lerr_towrite.append(-99.0)
                redshift_uerr_towrite.append(-99.0)
            else:
                # Not dealing with multiple matches for now
                continue
        else:
            continue    

    # save matched file
    # South
    pears_s_id_towrite = np.asarray(pears_s_id_towrite)
    pears_s_ra_towrite = np.asarray(pears_s_ra_towrite)
    pears_s_dec_towrite = np.asarray(pears_s_dec_towrite)
    match_id_towrite = np.asarray(match_id_towrite)
    match_ra_towrite = np.asarray(match_ra_towrite)
    match_dec_towrite = np.asarray(match_dec_towrite)
    match_source_towrite = np.asarray(match_source_towrite)
    redshift_towrite = np.asarray(redshift_towrite)
    redshift_lerr_towrite = np.asarray(redshift_lerr_towrite)
    redshift_uerr_towrite = np.asarray(redshift_uerr_towrite)

    data = np.array(zip(pears_s_id_towrite, pears_s_ra_towrite, pears_s_dec_towrite, match_id_towrite,\
     match_ra_towrite, match_dec_towrite, match_source_towrite, redshift_towrite, redshift_lerr_towrite, redshift_uerr_towrite),\
                    dtype=[('pears_s_id_towrite', int), ('pears_s_ra_towrite', float), ('pears_s_dec_towrite', float),\
                     ('match_id_towrite', int), ('match_ra_towrite', float), ('match_dec_towrite', float), ('match_source_towrite', '|S7'),\
                     ('redshift_towrite', float), ('redshift_lerr_towrite', float), ('redshift_uerr_towrite', float)])

    np.savetxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', data, fmt=['%d', '%.6f', '%.6f', '%d', '%.6f', '%.6f', '%s', '%.4f', '%.4f', '%.4f'],\
     delimiter=' ', header='These galaxies are in the GOODS-S field.' + '\n' + 'pearsid pearsra pearsdec matchid matchra matchdec source zphot zphot_l68 zphot_u68')

    sys.exit(0)


