from __future__ import division

import numpy as np

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

sys.path.append(massive_galaxies_dir)
import matching as mt

if __name__ == '__main__':
    """
    This code matches the PEARS master catalogs with
    Nimish's spectroscopic redshift catalogs.
    """

    # read PEARS master catalogs 
    pears_master_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag', 'netsig_corr'], usecols=(0,1,2,3,6))
    pears_master_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag', 'netsig_corr'], usecols=(0,1,2,3,6))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_master_ncat['dec'] = pears_master_ncat['dec'] - dec_offset_goodsn_v19

    # read Nimish's specz catalogs
    goods_n_specz_cat = np.genfromtxt(massive_galaxies_dir + 'goods_n_specz_0117.txt', \
        dtype=None, names=['ra','dec','z_spec','z_qual','catname','duplicate'], skip_header=13)
    goods_s_specz_cat = np.genfromtxt(massive_galaxies_dir + 'cdfs_specz_0117.txt', \
        dtype=None, names=['ra','dec','z_spec','z_qual','catname','duplicate'], skip_header=13)
    
    # Matching
    # create arrays
    specz_n_ra = goods_n_specz_cat['ra']
    specz_n_dec = goods_n_specz_cat['dec']

    specz_s_ra = goods_s_specz_cat['ra']
    specz_s_dec = goods_s_specz_cat['dec']

    pears_n_ra = pears_master_ncat['ra']
    pears_n_dec = pears_master_ncat['dec']

    pears_s_ra = pears_master_scat['ra']
    pears_s_dec = pears_master_scat['dec']

    # run matching code
    # north
    deltaRA, deltaDEC, specz_n_ra_matches, specz_n_dec_matches, pears_n_ra_matches, pears_n_dec_matches, \
    specz_n_ind, pears_n_ind, num_single_matches = \
    mt.match(specz_n_ra, specz_n_dec, pears_n_ra, pears_n_dec, lim=0.1*1/3600)

    print "There were", num_single_matches, "single matches found out of", len(pears_master_ncat), 
    print "objects in the PEARS NORTH catalog with spec z catalog from N. Hathi."
    mt.plot_diff(deltaRA, deltaDEC, name='pears_specz_goodsn')

    # south
    deltaRA, deltaDEC, specz_s_ra_matches, specz_s_dec_matches, pears_s_ra_matches, pears_s_dec_matches, \
    specz_s_ind, pears_s_ind, num_single_matches = \
    mt.match(specz_s_ra, specz_s_dec, pears_s_ra, pears_s_dec, lim=0.1*1/3600)

    print "There were", num_single_matches, "single matches found out of", len(pears_master_scat), 
    print "objects in the PEARS SOUTH catalog with spec z catalog from N. Hathi."
    mt.plot_diff(deltaRA, deltaDEC, name='pears_specz_goodss')

    #plt.show()

    # Now find how many are in your redshift range
    all_spec_cats = [goods_n_specz_cat, goods_s_specz_cat]

    spec_count = 0
    catcount = 0
    for cat in all_spec_cats:

        if catcount == 0:
            spec_cat = goods_n_specz_cat
            spec_ind = specz_n_ind
            pears_cat = pears_master_ncat
            pears_ind = pears_n_ind
            current_field = 'GOODS-N'

        elif catcount == 1:
            spec_cat = goods_s_specz_cat
            spec_ind = specz_s_ind
            pears_cat = pears_master_scat
            pears_ind = pears_s_ind
            current_field = 'GOODS-S'

        total_objects = len(spec_ind)
        print "In field", current_field, "with", total_objects, "objects."

        for i in range(total_objects):

            current_id = pears_cat['id'][pears_ind][i]
            #current_photz = pears_cat['old_z'][pears_ind][i]

            current_specz = float(spec_cat['z_spec'][spec_ind][i])
            current_specz_source = spec_cat['catname'][spec_ind][i]
            current_specz_qual = spec_cat['z_qual'][spec_ind][i]

            #print current_field, current_id, current_specz, current_specz_source, current_specz_qual

            # get i band mag
            if current_field == 'GOODS-N':
                idarg = np.where(pears_master_ncat['id'] == current_id)[0]
                imag = pears_master_ncat['imag'][idarg]
                netsig_corr = pears_master_ncat['netsig_corr'][idarg]
            elif current_field == 'GOODS-S':
                idarg = np.where(pears_master_scat['id'] == current_id)[0]
                imag = pears_master_scat['imag'][idarg]
                netsig_corr = pears_master_scat['netsig_corr'][idarg]

            if (current_specz >= 0.6) & (current_specz <= 1.235):
                if current_specz_qual == 'A' or current_specz_qual == '4':
                    if current_specz_source == '3D_HST':
                        continue
                    else:
                        spec_count += 1
                        print "\n", "At id:", current_id, "in", current_field,
                        print "Corrected NetSig:", netsig_corr, "  i-band mag:", imag
                        print "Spec-z is", current_specz, "from", current_specz_source, "with quality", current_specz_qual
                        #print "Photo-z is", pears_cat['old_z'][pears_ind][i]

        catcount += 1

    print "Total", spec_count, "spec redshifts in z range with highest quality"
    sys.exit(0)