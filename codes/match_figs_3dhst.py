from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

sys.path.append(home)
import matching as mt

if __name__ == '__main__':
    
    # Read 3D-HST catalog
    threed_ncat = fits.open(home + '/Desktop/FIGS/new_codes/goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat.FITS')
    threed_scat = fits.open(home + '/Desktop/FIGS/new_codes/goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat.FITS')

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

    # Create arrays for ra and dec for both catalogs
    threed_n_ra = threed_ncat[1].data['ra']
    threed_n_dec = threed_ncat[1].data['dec']
    threed_s_ra = threed_scat[1].data['ra']
    threed_s_dec = threed_scat[1].data['dec']

    gn1_ra = gn1cat['ra']
    gn1_dec = gn1cat['dec']

    gn2_ra = gn2cat['ra']
    gn2_dec = gn2cat['dec']

    gs1_ra = gs1cat['ra']
    gs1_dec = gs1cat['dec']

    gs2_ra = gs2cat['ra']
    gs2_dec = gs2cat['dec']

    # Run the matching function
    # GN1
    deltaRA, deltaDEC, threed_gn1_ra_matches, threed_gn1_dec_matches, figs_gn1_ra_matches, figs_gn1_dec_matches, num_single_matches = \
    mt.match(threed_n_ra, threed_n_dec, gn1_ra, gn1_dec, lim=0.3*1/3600)

    print "Single matches GN1 - ", num_single_matches, "out of", len(gn1cat)
    mt.plot_diff(deltaRA, deltaDEC, name='gn1')

    # GN2
    deltaRA, deltaDEC, threed_gn2_ra_matches, threed_gn2_dec_matches, figs_gn2_ra_matches, figs_gn2_dec_matches, num_single_matches = \
    mt.match(threed_n_ra, threed_n_dec, gn2_ra, gn2_dec, lim=0.3*1/3600)

    print "Single matches GN2 - ", num_single_matches, "out of", len(gn2cat)
    mt.plot_diff(deltaRA, deltaDEC, name='gn2')

    # GS1
    deltaRA, deltaDEC, threed_gs1_ra_matches, threed_gs1_dec_matches, figs_gs1_ra_matches, figs_gs1_dec_matches, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, gs1_ra, gs1_dec, lim=0.3*1/3600)

    print "Single matches GS1 - ", num_single_matches, "out of", len(gs1cat)
    mt.plot_diff(deltaRA, deltaDEC, name='gs1')

    # GS2
    deltaRA, deltaDEC, threed_gs2_ra_matches, threed_gs2_dec_matches, figs_gs2_ra_matches, figs_gs2_dec_matches, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, gs2_ra, gs2_dec, lim=0.3*1/3600)

    print "Single matches GS2 - ", num_single_matches, "out of", len(gs2cat)

    # find offset and take it out
    raoff, decoff = mt.findoffset(deltaRA, deltaDEC)
    gs2_ra += raoff

    # match again
    deltaRA, deltaDEC, threed_gs2_ra_matches, threed_gs2_dec_matches, figs_gs2_ra_matches, figs_gs2_dec_matches, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, gs2_ra, gs2_dec, lim=0.3*1/3600)
    print "Single matches GS2 - ", num_single_matches, "out of", len(gs2cat)

    # check that offset is gone
    raoff, decoff = mt.findoffset(deltaRA, deltaDEC)
    mt.plot_diff(deltaRA, deltaDEC, name='gs2')

    sys.exit(0)