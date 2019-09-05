"""
The purpose of this code is to get the final sample for running the 
redshift pipeline on all of PEARS.

It will match the PEARS master catalog with the 3DHST photometry catalog.
A match has to be found in the photometry catalog otherwise it will skip
the object. 
It will also attempt to provide a ground-based spectroscopic redshift 
by matching with the catalog from Nimish. Other zspec, zspec quality, 
and zspec source are set to -99.0 if no match is found.

No cut on redshift is applied.
Only a NetSig cut of 10 is applied.
"""

from __future__ import division

import numpy as np

import os
import sys

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import dn4000_catalog as dc
import new_refine_grismz_gridsearch_parallel as ngp

def main():

    # Get correct directories 
    threedhst_datadir = "/Volumes/Bhavins_backup/3dhst_data/"
    # This is if working on the laptop. 
    # Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(threedhst_datadir):
        threedhst_datadir = home + "/Desktop/3dhst_data/"  # this path only exists on firstlight

    # ------------------------------- Read PEARS cats ------------------------------- #
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    
    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['pearsdec'] = pears_ncat['pearsdec'] - dec_offset_goodsn_v19

    # ------------------------------- Read in 3DHST photomtery catalog ------------------------------- #
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
    'f_U', 'e_U', 'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
    'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), \
        skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), \
        skip_header=3)

    # ------------------------------- Read in ground-based spectroscopic redshift compilation ------------------------------- #
    # read Nimish's specz catalogs
    goods_n_specz_cat = np.genfromtxt(massive_galaxies_dir + 'goods_n_specz_0117.txt', \
        dtype=None, names=['ra','dec','z_spec','z_qual','catname','duplicate'], skip_header=13)
    goods_s_specz_cat = np.genfromtxt(massive_galaxies_dir + 'cdfs_specz_0117.txt', \
        dtype=None, names=['ra','dec','z_spec','z_qual','catname','duplicate'], skip_header=13)

    # ------------------------------- Now match the three ------------------------------- #
    # Preliminary prep before matching
    all_pears_cats = [pears_ncat, pears_scat]

    # Lists for saving to file
    pears_id_list = []
    pears_field_list = []
    pears_ra_list = []
    pears_dec_list = []
    zspec_list = []
    zspec_source_list = []
    zspec_qual_list = []
    netsig_list = []
    imag_list = []

    catcount = 0
    match_count = 0
    for cat in all_pears_cats:

        if catcount == 0:
            cat = pears_ncat
            phot_cat_3dhst = goodsn_phot_cat_3dhst
            spec_cat = goods_n_specz_cat
            current_field = 'GOODS-N'
        if catcount == 1:
            cat = pears_scat
            phot_cat_3dhst = goodss_phot_cat_3dhst
            spec_cat = goods_s_specz_cat
            current_field = 'GOODS-S'

        threed_ra = phot_cat_3dhst['ra']
        threed_dec = phot_cat_3dhst['dec']
        spec_ra = spec_cat['ra']
        spec_dec = spec_cat['dec']

        for i in range(len(cat)):

            current_id = cat['id'][i]

            # find grism obj ra,dec
            current_ra = float(cat['pearsra'][i])
            current_dec = float(cat['pearsdec'][i])

            # Matching radius
            ra_lim = 0.3/3600  # arcseconds in degrees
            dec_lim = 0.3/3600

            # ------------------------- Spec match ------------------------- #
            spec_idx = np.where((spec_ra >= current_ra - ra_lim) & (spec_ra <= current_ra + ra_lim) & \
                (spec_dec >= current_dec - dec_lim) & (spec_dec <= current_dec + dec_lim))[0]

            """
            If there are multiple matches within 0.5 arseconds then choose the closest one.
            """
            if len(spec_idx) > 1:
                print "Found multiple matches in ground-based spectroscopic catalog. Picking closest one."

                ra_two = current_ra
                dec_two = current_dec

                dist_list = []
                for v in range(len(spec_idx)):

                    ra_one = spec_ra[spec_idx][v]
                    dec_one = spec_dec[spec_idx][v]

                    dist = np.arccos(np.cos(dec_one*np.pi/180) * np.cos(dec_two*np.pi/180) * \
                        np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + \
                        np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
                    dist_list.append(dist)

                dist_list = np.asarray(dist_list)
                dist_idx = np.argmin(dist_list)
                spec_idx = spec_idx[dist_idx]
                
                current_zspec = float(spec_cat['z_spec'][spec_idx])
                current_zspec_source = spec_cat['catname'][spec_idx]
                current_zspec_qual = spec_cat['z_qual'][spec_idx]

            else:
                current_zspec = -99.0
                current_zspec_source = -99.0
                current_zspec_qual = -99.0

            # ------------------------- Now match with photometry ------------------------- #
            # Uses exact same procedure as above
            threed_phot_idx = np.where((threed_ra >= current_ra - ra_lim) & (threed_ra <= current_ra + ra_lim) & \
                (threed_dec >= current_dec - dec_lim) & (threed_dec <= current_dec + dec_lim))[0]

            """
            If there are multiple matches within 0.5 arseconds then choose the closest one.
            """
            if len(threed_phot_idx) > 1:
                print "Found multiple matches in photometry catalog. Picking closest one."

                ra_two = current_ra
                dec_two = current_dec

                dist_list = []
                for v in range(len(threed_phot_idx)):

                    ra_one = threed_ra[threed_phot_idx][v]
                    dec_one = threed_dec[threed_phot_idx][v]

                    dist = np.arccos(np.cos(dec_one*np.pi/180) * np.cos(dec_two*np.pi/180) * \
                        np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + \
                        np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
                    dist_list.append(dist)

                dist_list = np.asarray(dist_list)
                dist_idx = np.argmin(dist_list)
                threed_phot_idx = threed_phot_idx[dist_idx]

            elif len(threed_phot_idx) == 0:
                print "Match not found in Photmetry catalog. Skipping."
                continue

            # Get current i-band magnitude
            current_imag = float(cat['imag'][i])

            # Magnitude cut
            #if current_imag > 24.0:
            #    print "Skipping due to magnitude cut. Current galaxy magnitude (i_AB):", current_imag
            #    continue

            print "PEARS ID and Field:", current_id, current_field

            # Get NetSig
            grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = \
            ngp.get_data(current_id, current_field)

            if return_code == 0:
                print current_field, current_id,
                print "Skipping due to an error with the obs data. See the error message just above this one.",
                print "Moving to the next galaxy."
                continue

            if netsig_chosen < 10:
                print current_field, current_id,
                print "Skipping due to low NetSig:", netsig_chosen
                continue

            pears_id_list.append(current_id)
            pears_field_list.append(current_field)
            pears_ra_list.append(current_ra)
            pears_dec_list.append(current_dec)
            zspec_list.append(current_zspec)
            zspec_source_list.append(current_zspec_source)
            zspec_qual_list.append(current_zspec_qual)
            netsig_list.append(netsig_chosen)
            imag_list.append(current_imag)

            match_count += 1

        catcount += 1

    print "Total galaxies in sample:", match_count

    # Save final sample
    # Convertt to numpy arrays and save
    pears_id = np.asarray(pears_id_list)
    pears_field = np.asarray(pears_field_list)
    pears_ra = np.asarray(pears_ra_list)
    pears_dec = np.asarray(pears_dec_list)
    zspec = np.asarray(zspec_list)
    zspec_source = np.asarray(zspec_source_list)
    zspec_qual = np.asarray(zspec_qual_list)
    netsig = np.asarray(netsig_list)
    imag = np.asarray(imag_list)

    # Save to ASCII file
    data = np.array(zip(pears_id, pears_field, pears_ra, pears_dec, zspec, zspec_source, zspec_qual, netsig, imag),\
        dtype=[('pears_id', int), ('pears_field', '|S7'), ('pears_ra', float), ('pears_dec', float), \
        ('zspec', float), ('zspec_source', '|S10'), ('zspec_qual', '|S1'), ('netsig', float), ('imag', float)])
    np.savetxt(massive_galaxies_dir + 'pears_full_sample.txt', data, \
        fmt=['%d', '%s', '%.7f', '%.6f', '%.4f', '%s', '%s', '%.2f', '%.2f'],\
        delimiter=' ', header='pearsid  field  ra  dec  zspec  zspec_source  zspec_qual  netsig  imag')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
