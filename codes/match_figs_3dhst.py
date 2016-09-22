from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

sys.path.append(massive_galaxies_dir)
import matching as mt

def match_candels_3dhst(threed_s_ra, threed_s_dec):

    candels_goodss_cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/candels_goodss_photometry.txt', dtype=None, names=['id', 'ra', 'dec'], usecols=(0, 2, 3), skip_header=73)

    candels_goodss_ra = candels_goodss_cat['ra']
    candels_goodss_dec = candels_goodss_cat['dec']

    # Using only goodss because there isn't a CANDELS GOODS-N catalog

    deltaRA, deltaDEC, threed_goodss_ra_matches, threed_goodss_dec_matches, candels_goodss_ra_matches, candels_goodss_dec_matches, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, candels_goodss_ra, candels_goodss_dec, lim=0.3*1/3600)

    print "Single matches for CANDELS and 3DHST - ", num_single_matches
    mt.plot_diff(deltaRA, deltaDEC, name='candels_goodss_3dhst')

    return None

def match_figs_candels(figs_ra, figs_dec, figs_field):

    candels_goodss_cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/candels_goodss_photometry.txt', dtype=None, names=['id', 'ra', 'dec'], usecols=(0, 2, 3), skip_header=73)

    candels_goodss_ra = candels_goodss_cat['ra']
    candels_goodss_dec = candels_goodss_cat['dec']

    if (figs_field == 'gs1') or (figs_field == 'gs2'):
        figs_s_ra = figs_ra
        figs_s_dec = figs_dec 

        deltaRA, deltaDEC, figs_goodss_ra_matches, figs_goodss_dec_matches, candels_goodss_ra_matches, candels_goodss_dec_matches, num_single_matches = \
        mt.match(figs_s_ra, figs_s_dec, candels_goodss_ra, candels_goodss_dec, lim=0.3*1/3600)

        print "Single matches for FIGS " + figs_field + " and CANDELS - ", num_single_matches, "out of", len(figs_ra), "objects in the FIGS catalog."
        mt.plot_diff(deltaRA, deltaDEC, name='candels_figs_' + figs_field)

    # use the block below whenever a CANDELS GOODS-N catalog comes around
    #elif (figs_field == 'gn1') or (figs_field == 'gn2'):
    #    figs_n_ra = figs_ra
    #    figs_n_dec = figs_dec 

    #    deltaRA, deltaDEC, figs_goodsn_ra_matches, figs_goodsn_dec_matches, candels_goodsn_ra_matches, candels_goodsn_dec_matches, num_single_matches = \
    #    mt.match(figs_n_ra, figs_n_dec, candels_goodsn_ra, candels_goodsn_dec, lim=0.3*1/3600)

    #    print "Single matches for FIGS " + figs_field + " and CANDELS - ", num_single_matches, "out of", len(figs_ra)
    #    mt.plot_diff(deltaRA, deltaDEC, name='candels_figs_' + figs_field)

    return None

def match_candels_goods():

    candels_goodss_cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/candels_goodss_photometry.txt', dtype=None, names=['id', 'ra', 'dec'], usecols=(0, 2, 3), skip_header=73)

    candels_goodss_ra = candels_goodss_cat['ra']
    candels_goodss_dec = candels_goodss_cat['dec']

    goodss_cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/h_goods_sz_r2.0z_cat.txt', dtype=None, names=['id', 'ra', 'dec'], usecols=(0, 1, 2), skip_header=104)

    goodss_ra = goodss_cat['ra']
    goodss_dec = goodss_cat['dec']

    deltaRA, deltaDEC, goodss_ra_matches, goodss_dec_matches, candels_goodss_ra_matches, candels_goodss_dec_matches, num_single_matches = \
    mt.match(goodss_ra, goodss_dec, candels_goodss_ra, candels_goodss_dec, lim=0.3*1/3600)

    print "Single matches for GOODS-S and CANDELS-GOODS-S - ", num_single_matches
    mt.plot_diff(deltaRA, deltaDEC, name='candels_goodss')
    
    return None

def match_figs_goods(figs_ra, figs_dec, figs_field):

    goodss_cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/h_goods_sz_r2.0z_cat.txt', dtype=None, names=['id', 'ra', 'dec'], usecols=(0, 1, 2), skip_header=104)

    goodss_ra = goodss_cat['ra']
    goodss_dec = goodss_cat['dec']

    goodsn_cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/h_goods_nz_r2.0z_cat.txt', dtype=None, names=['id', 'ra', 'dec'], usecols=(0, 1, 2), skip_header=104)

    goodsn_ra = goodsn_cat['ra']
    goodsn_dec = goodsn_cat['dec']

    if (figs_field == 'gs1') or (figs_field == 'gs2'):
        figs_s_ra = figs_ra
        figs_s_dec = figs_dec 

        deltaRA, deltaDEC, figs_goodss_ra_matches, figs_goodss_dec_matches, goodss_ra_matches, goodss_dec_matches, num_single_matches = \
        mt.match(figs_s_ra, figs_s_dec, goodss_ra, goodss_dec, lim=0.3*1/3600)

        print "Single matches for FIGS " + figs_field + " and GOODS-S - ", num_single_matches, "out of", len(figs_ra), "objects in the FIGS catalog."
        mt.plot_diff(deltaRA, deltaDEC, name='goodss_figs_' + figs_field)

    elif (figs_field == 'gn1') or (figs_field == 'gn2'):
        figs_n_ra = figs_ra
        figs_n_dec = figs_dec 

        deltaRA, deltaDEC, figs_goodsn_ra_matches, figs_goodsn_dec_matches, goodsn_ra_matches, goodsn_dec_matches, num_single_matches = \
        mt.match(figs_n_ra, figs_n_dec, goodsn_ra, goodsn_dec, lim=0.3*1/3600)

        print "Single matches for FIGS " + figs_field + " and GOODS-N - ", num_single_matches, "out of", len(figs_ra), "objects in the FIGS catalog."
        mt.plot_diff(deltaRA, deltaDEC, name='goodsn_figs_' + figs_field)

    return None

def match_goods_3dhst(threed_n_ra, threed_n_dec, threed_s_ra, threed_s_dec):

    # read in goods cats and make arrays
    goodss_cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/h_goods_sz_r2.0z_cat.txt', dtype=None, names=['id', 'ra', 'dec'], usecols=(0, 1, 2), skip_header=104)

    goodss_ra = goodss_cat['ra']
    goodss_dec = goodss_cat['dec']

    goodsn_cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/h_goods_nz_r2.0z_cat.txt', dtype=None, names=['id', 'ra', 'dec'], usecols=(0, 1, 2), skip_header=104)

    goodsn_ra = goodsn_cat['ra']
    goodsn_dec = goodsn_cat['dec']

    # match south
    deltaRA, deltaDEC, threed_goodss_ra_matches, threed_goodss_dec_matches, goodss_ra_matches, goodss_dec_matches, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, goodss_ra, goodss_dec, lim=0.3*1/3600)

    print "Single matches for 3DHST and GOODS-S - ", num_single_matches
    mt.plot_diff(deltaRA, deltaDEC, name='goodss_3dhst')

    # match north
    deltaRA, deltaDEC, threed_goodsn_ra_matches, threed_goodsn_dec_matches, goodsn_ra_matches, goodsn_dec_matches, num_single_matches = \
    mt.match(threed_n_ra, threed_n_dec, goodsn_ra, goodsn_dec, lim=0.3*1/3600)

    print "Single matches for 3DHST and GOODS-N - ", num_single_matches
    mt.plot_diff(deltaRA, deltaDEC, name='goodsn_3dhst')

    return None

def read_3dhst_cats():

    # Read 3D-HST catalog
    # Using v4.1 instead of the master v4.1.5 catalog
    threed_ncat = fits.open(home + '/Desktop/FIGS/new_codes/goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat.FITS')
    threed_scat = fits.open(home + '/Desktop/FIGS/new_codes/goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat.FITS')

    #threed_v415 = fits.open(home + '/Documents/3D-HST/3dhst.v4.1.5.master.fits')

    return threed_ncat, threed_scat

if __name__ == '__main__':
    
    # Read 3dhst cats
    threed_ncat, threed_scat = read_3dhst_cats()

    # Read in FIGS catalogs # latest version v1.2
    gn1cat = np.genfromtxt(massive_galaxies_dir + 'GN1_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec'], usecols=([2,3,4]), skip_header=25)
    gn2cat = np.genfromtxt(massive_galaxies_dir + 'GN2_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec'], usecols=([2,3,4]), skip_header=25)
    gs1cat = np.genfromtxt(massive_galaxies_dir + 'GS1_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec'], usecols=([2,3,4]), skip_header=25)
    gs2cat = np.genfromtxt(massive_galaxies_dir + 'GS2_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec'], usecols=([2,3,4]), skip_header=25)

    # Number of objects in each FIGS field
    # GN1: 7271
    # GN2: 3810
    # GS1: 8193
    # GS2: 6354
    # Total: 25628

    # Create arrays for id, ra, and dec for both catalogs
    threed_n_ra = threed_ncat[1].data['ra']
    threed_n_dec = threed_ncat[1].data['dec']
    threed_s_ra = threed_scat[1].data['ra']
    threed_s_dec = threed_scat[1].data['dec']
    threed_n_id = threed_ncat[1].data['id']
    threed_s_id = threed_scat[1].data['id']

    gn1_ra = gn1cat['ra']
    gn1_dec = gn1cat['dec']
    gn1_id = gn1cat['id']

    gn2_ra = gn2cat['ra']
    gn2_dec = gn2cat['dec']
    gn2_id = gn2cat['id']

    gs1_ra = gs1cat['ra']
    gs1_dec = gs1cat['dec']
    gs1_id = gs1cat['id']

    gs2_ra = gs2cat['ra']
    gs2_dec = gs2cat['dec']
    gs2_id = gs2cat['id']

    # Match 3DHST and FIGS to CANDELS and the GOODS catalogs
    #match_candels_3dhst(threed_s_ra, threed_s_dec)

    #match_goods_3dhst(threed_n_ra, threed_n_dec, threed_s_ra, threed_s_dec)

    #match_figs_candels(gs1_ra, gs1_dec, 'gs1')
    #match_figs_candels(gs2_ra, gs2_dec, 'gs2')

    #match_figs_goods(gn1_ra, gn1_dec, 'gn1')
    #match_figs_goods(gn2_ra, gn2_dec, 'gn2')
    #match_figs_goods(gs1_ra, gs1_dec, 'gs1')
    #match_figs_goods(gs2_ra, gs2_dec, 'gs2')

    #match_candels_goods()

    #sys.exit(0)

    # Run the matching function
    # GN1 --------------------------------------------------
    deltaRA, deltaDEC, threed_gn1_ra_matches, threed_gn1_dec_matches, figs_gn1_ra_matches, figs_gn1_dec_matches, threed_n_gn1_ind, figs_gn1_ind, num_single_matches = \
    mt.match(threed_n_ra, threed_n_dec, gn1_ra, gn1_dec, lim=0.3*1/3600)

    print "Single matches GN1 - ", num_single_matches, "out of", len(gn1cat), "objects in the FIGS catalog."
    mt.plot_diff(deltaRA, deltaDEC, name='threedhst_gn1')

    # GN2 --------------------------------------------------
    deltaRA, deltaDEC, threed_gn2_ra_matches, threed_gn2_dec_matches, figs_gn2_ra_matches, figs_gn2_dec_matches, threed_n_gn2_ind, figs_gn2_ind, num_single_matches = \
    mt.match(threed_n_ra, threed_n_dec, gn2_ra, gn2_dec, lim=0.3*1/3600)

    print "Single matches GN2 - ", num_single_matches, "out of", len(gn2cat), "objects in the FIGS catalog."
    mt.plot_diff(deltaRA, deltaDEC, name='threedhst_gn2')

    # find offsets and take them out
    raoff, decoff = mt.findoffset(deltaRA, deltaDEC)
    gn2_ra += raoff

    # match again
    deltaRA, deltaDEC, threed_gn2_ra_matches, threed_gn2_dec_matches, figs_gn2_ra_matches, figs_gn2_dec_matches, threed_n_gn2_ind, figs_gn2_ind, num_single_matches = \
    mt.match(threed_n_ra, threed_n_dec, gn2_ra, gn2_dec, lim=0.3*1/3600)
    print "Single matches GN2 -", num_single_matches, "out of", len(gn2cat), "objects in the FIGS catalog."

    # check that offset is gone
    raoff, decoff = mt.findoffset(deltaRA, deltaDEC)
    mt.plot_diff(deltaRA, deltaDEC, name='threedhst_gn2_afteroffsetremoved')

    ## GS1 --------------------------------------------------
    deltaRA, deltaDEC, threed_gs1_ra_matches, threed_gs1_dec_matches, figs_gs1_ra_matches, figs_gs1_dec_matches, threed_n_gs1_ind, figs_gs1_ind, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, gs1_ra, gs1_dec, lim=0.3*1/3600)

    print "Single matches GS1 - ", num_single_matches, "out of", len(gs1cat), "objects in the FIGS catalog."
    mt.plot_diff(deltaRA, deltaDEC, name='threedhst_gs1')

    # find offsets and take them out
    raoff, decoff = mt.findoffset(deltaRA, deltaDEC)
    gs1_ra += raoff
    gs1_dec += decoff

    # match again
    deltaRA, deltaDEC, threed_gs1_ra_matches, threed_gs1_dec_matches, figs_gs1_ra_matches, figs_gs1_dec_matches, threed_n_gs1_ind, figs_gs1_ind, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, gs1_ra, gs1_dec, lim=0.3*1/3600)

    print "Single matches GS1 - ", num_single_matches, "out of", len(gs1cat), "objects in the FIGS catalog."

    # check that offset is gone
    raoff, decoff = mt.findoffset(deltaRA, deltaDEC)
    mt.plot_diff(deltaRA, deltaDEC, name='threedhst_gs1_afteroffsetremoved')    

    # GS2 --------------------------------------------------
    deltaRA, deltaDEC, threed_gs2_ra_matches, threed_gs2_dec_matches, figs_gs2_ra_matches, figs_gs2_dec_matches, threed_n_gs2_ind, figs_gs2_ind, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, gs2_ra, gs2_dec, lim=0.3*1/3600)

    print "Single matches GS2 - ", num_single_matches, "out of", len(gs2cat), "out of", len(gs2_ra), "objects in the FIGS catalog."
    mt.plot_diff(deltaRA, deltaDEC, name='threedhst_gs2')

    # find offsets and take them out
    raoff, decoff = mt.findoffset(deltaRA, deltaDEC)
    gs2_ra += raoff

    # match again
    deltaRA, deltaDEC, threed_gs2_ra_matches, threed_gs2_dec_matches, figs_gs2_ra_matches, figs_gs2_dec_matches, threed_n_gs2_ind, figs_gs2_ind, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, gs2_ra, gs2_dec, lim=0.3*1/3600)
    print "Single matches GS2 -", num_single_matches, "out of", len(gs2cat), "out of", len(gs2_ra), "objects in the FIGS catalog."

    # check that offset is gone
    raoff, decoff = mt.findoffset(deltaRA, deltaDEC)
    mt.plot_diff(deltaRA, deltaDEC, name='threedhst_gs2_afteroffsetremoved')

    # SAve matches info to a separte file
    # the original figs and 3dhst catalogs used for the matching have only 6 digits after the decimal point for both ra and dec
    # this is why I'm only saving 6 digits after the decimal point.
    # The code by default will save the offset corrected coordinates.
    # GN1 --------------------------------------------------
    gn1_id_matched = gn1_id[figs_gn1_ind]
    threed_n_gn1_id_matched = threed_n_id[threed_n_gn1_ind]
    data = np.array(zip(gn1_id_matched, threed_n_gn1_id_matched, figs_gn1_ra_matches, figs_gn1_dec_matches, threed_gn1_ra_matches, threed_gn1_dec_matches),\
                dtype=[('gn1_id_matched', int), ('threed_n_gn1_id_matched', int), ('figs_gn1_ra_matches', float), ('figs_gn1_dec_matches', float), ('threed_gn1_ra_matches', float), ('threed_gn1_dec_matches', float)])
    np.savetxt(massive_galaxies_dir + 'gn1_threedhst_matches.txt', data, fmt=['%d', '%d', '%.6f', '%.6f', '%.6f', '%.6f'], delimiter=' ',\
               header= 'The 3DHST ID is specific to its north or south catalog. Currently using 3DHST v4.1 catalog.' + '\n' + 'figs_id threed_north_idv41 figs_ra figs_dec threed_ra threed_dec')

    # GN2 --------------------------------------------------
    gn2_id_matched = gn2_id[figs_gn2_ind]
    threed_n_gn2_id_matched = threed_n_id[threed_n_gn2_ind]
    data = np.array(zip(gn2_id_matched, threed_n_gn2_id_matched, figs_gn2_ra_matches, figs_gn2_dec_matches, threed_gn2_ra_matches, threed_gn2_dec_matches),\
                dtype=[('gn2_id_matched', int), ('threed_n_gn2_id_matched', int), ('figs_gn2_ra_matches', float), ('figs_gn2_dec_matches', float), ('threed_gn2_ra_matches', float), ('threed_gn2_dec_matches', float)])
    np.savetxt(massive_galaxies_dir + 'gn2_threedhst_matches.txt', data, fmt=['%d', '%d', '%.6f', '%.6f', '%.6f', '%.6f'], delimiter=' ',\
               header= 'The 3DHST ID is specific to its north or south catalog. Currently using 3DHST v4.1 catalog.' + '\n' + 'figs_id threed_north_idv41 figs_ra figs_dec threed_ra threed_dec')

    # GS1 --------------------------------------------------
    gs1_id_matched = gs1_id[figs_gn2_ind]
    threed_n_gs1_id_matched = threed_n_id[threed_n_gs1_ind]
    data = np.array(zip(gs1_id_matched, threed_n_gs1_id_matched, figs_gs1_ra_matches, figs_gs1_dec_matches, threed_gs1_ra_matches, threed_gs1_dec_matches),\
                dtype=[('gs1_id_matched', int), ('threed_n_gs1_id_matched', int), ('figs_gs1_ra_matches', float), ('figs_gs1_dec_matches', float), ('threed_gs1_ra_matches', float), ('threed_gs1_dec_matches', float)])
    np.savetxt(massive_galaxies_dir + 'gs1_threedhst_matches.txt', data, fmt=['%d', '%d', '%.6f', '%.6f', '%.6f', '%.6f'], delimiter=' ',\
               header= 'The 3DHST ID is specific to its north or south catalog. Currently using 3DHST v4.1 catalog.' + '\n' + 'figs_id threed_south_idv41 figs_ra figs_dec threed_ra threed_dec')


    # GS2 --------------------------------------------------
    gs2_id_matched = gs2_id[figs_gs2_ind]
    threed_n_gs2_id_matched = threed_n_id[threed_n_gs2_ind]
    data = np.array(zip(gs2_id_matched, threed_n_gs2_id_matched, figs_gs2_ra_matches, figs_gs2_dec_matches, threed_gs2_ra_matches, threed_gs2_dec_matches),\
                dtype=[('gs2_id_matched', int), ('threed_n_gs2_id_matched', int), ('figs_gs2_ra_matches', float), ('figs_gs2_dec_matches', float), ('threed_gs2_ra_matches', float), ('threed_gs2_dec_matches', float)])
    np.savetxt(massive_galaxies_dir + 'gs2_threedhst_matches.txt', data, fmt=['%d', '%d', '%.6f', '%.6f', '%.6f', '%.6f'], delimiter=' ',\
               header= 'The 3DHST ID is specific to its north or south catalog. Currently using 3DHST v4.1 catalog.' + '\n' + 'figs_id threed_south_idv41 figs_ra figs_dec threed_ra threed_dec')

    sys.exit(0)