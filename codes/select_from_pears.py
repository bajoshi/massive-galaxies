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

def do_pears_threed_matching():

    # First re do the matching 
    # I don't remember if the prvious matching was done with 3DHST cat 4.1 or 4.1.5
    # So I'm doing it again.
    # Read 3dhst 4.1.5 cat
    threedcat = fits.open(home + '/Documents/3D-HST/3dhst.v4.1.5.master.fits')  # v4.1.5 catalog 
    
    # Read PEARS cats
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS v2.0 readme
    pears_ncat['dec'] = pears_ncat['dec'] - dec_offset_goodsn_v19

    # Create arrays for id, ra, and dec for both catalogs
    threed_n_idx = np.where(threedcat[1].data['field'] == 'goodsn')[0]
    threed_s_idx = np.where(threedcat[1].data['field'] == 'goodss')[0]

    threed_n_ra = threedcat[1].data['ra'][threed_n_idx]
    threed_n_dec = threedcat[1].data['dec'][threed_n_idx]
    threed_n_id = threedcat[1].data['phot_id'][threed_n_idx]

    threed_s_ra = threedcat[1].data['ra'][threed_s_idx]
    threed_s_dec = threedcat[1].data['dec'][threed_s_idx]
    threed_s_id = threedcat[1].data['phot_id'][threed_s_idx]

    pears_n_ra = pears_ncat['ra']
    pears_n_dec = pears_ncat['dec']
    pears_n_id = pears_ncat['id']

    pears_s_ra = pears_scat['ra']
    pears_s_dec = pears_scat['dec']
    pears_s_id = pears_scat['id']

    # Match
    # North     
    deltaRA, deltaDEC, threed_n_ra_matches, threed_n_dec_matches, pears_n_ra_matches, pears_n_dec_matches, threed_n_ind, pears_n_ind, num_single_matches = \
    mt.match(threed_n_ra, threed_n_dec, pears_n_ra, pears_n_dec, lim=0.3*1/3600)

    print "Single matches PEARS North - ", num_single_matches, "out of", len(pears_ncat), "objects in the PEARS North master catalog."
    mt.plot_diff(deltaRA, deltaDEC, name='threedhst_pears_n')

    # South   
    deltaRA, deltaDEC, threed_s_ra_matches, threed_s_dec_matches, pears_s_ra_matches, pears_s_dec_matches, threed_s_ind, pears_s_ind, num_single_matches = \
    mt.match(threed_s_ra, threed_s_dec, pears_s_ra, pears_s_dec, lim=0.3*1/3600)

    print "Single matches PEARS South - ", num_single_matches, "out of", len(pears_scat), "objects in the PEARS South master catalog."
    mt.plot_diff(deltaRA, deltaDEC, name='threedhst_pears_s')

    # find offsets and take them out if needed
    raoff, decoff = mt.findoffset(deltaRA, deltaDEC)

    # Save results
    pears_n_id_matched = pears_n_id[pears_n_ind]
    threed_n_id_matched = threed_n_id[threed_n_ind]
    data = np.array(zip(pears_n_id_matched, threed_n_id_matched, pears_n_ra_matches, pears_n_dec_matches, threed_n_ra_matches, threed_n_dec_matches),\
                dtype=[('pears_n_id_matched', int), ('threed_n_id_matched', int), ('pears_n_ra_matches', float), ('pears_n_dec_matches', float), ('threed_n_ra_matches', float), ('threed_n_dec_matches', float)])
    np.savetxt(massive_galaxies_dir + 'pears_n_threedhst_matches.txt', data, fmt=['%d', '%d', '%.6f', '%.6f', '%.6f', '%.6f'], delimiter=' ',\
               header= 'The 3DHST ID is specific to its north or south catalog. Currently using 3DHST v4.1.5 catalog.' + '\n' + 'pears_id threed_north_idv415 pears_ra pears_dec threed_ra threed_dec')

    pears_s_id_matched = pears_s_id[pears_s_ind]
    threed_s_id_matched = threed_s_id[threed_s_ind]
    data = np.array(zip(pears_s_id_matched, threed_s_id_matched, pears_s_ra_matches, pears_s_dec_matches, threed_s_ra_matches, threed_s_dec_matches),\
                dtype=[('pears_s_id_matched', int), ('threed_s_id_matched', int), ('pears_s_ra_matches', float), ('pears_s_dec_matches', float), ('threed_s_ra_matches', float), ('threed_s_dec_matches', float)])
    np.savetxt(massive_galaxies_dir + 'pears_s_threedhst_matches.txt', data, fmt=['%d', '%d', '%.6f', '%.6f', '%.6f', '%.6f'], delimiter=' ',\
               header= 'The 3DHST ID is specific to its north or south catalog. Currently using 3DHST v4.1.5 catalog.' + '\n' + 'pears_id threed_south_idv415 pears_ra pears_dec threed_ra threed_dec')

    return None

def get_props(matchedfile, threedcat, pearscat, ra_list, dec_list):

    for i in range(len(matchedfile)):

        # first find object in 3DHST catlaog
        ra = matchedfile['threed_ra'][i]
        dec = matchedfile['threed_dec'][i]

        lim = 0.05 * 1/3600

        threed_idx = np.where((threedcat[1].data['ra'] >= ra - lim) & (threedcat[1].data['ra'] <= ra + lim) \
            & (threedcat[1].data['dec'] >= dec - lim) & (threedcat[1].data['dec'] <= dec + lim))[0]
        # see description in select_from_figs.py code

        # get stellar pop properties from 3DHST 
        mstar = threedcat[1].data['lmass'][threed_idx[0]]
        zphot = threedcat[1].data['z_peak_phot'][threed_idx[0]]

        # get magnitudes from FIGS
        current_id = matchedfile['pears_id'][i]
        pears_idx = np.where(pearscat['id'] == current_id)[0]

        # convert the fluxes to magnitudes
        # the fluxes are given in nJy
        imag = float(pearscat['imag'][pears_idx])

        # check if object can be selected
        if (zphot >= 0.5) and (mstar >= 10.5) and (imag < 22):
            print current_id, ra, dec, mstar, zphot, '{:.2f}'.format(imag)
            ra_list.append(ra)
            dec_list.append(dec)

    return ra_list, dec_list

if __name__ == '__main__':

    #do_pears_threed_matching()

    # Read 3dhst 4.1.5 cat
    threedcat = fits.open(home + '/Documents/3D-HST/3dhst.v4.1.5.master.fits')  # v4.1.5 catalog 

    # read in catalogs for matched objects between 3DHST and PEARS
    pears_n_matches = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/pears_n_threedhst_matches.txt', dtype=None, names=True, skip_header=1)
    pears_s_matches = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/pears_s_threedhst_matches.txt', dtype=None, names=True, skip_header=1)

    # Read PEARS cats
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))

    # loop over all matches in all fields and select an object if it passes all criteria
    ra_list = []
    dec_list = []

    ra_list, dec_list = get_props(pears_n_matches, threedcat, pears_ncat, ra_list, dec_list)

    # show objects on sky
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(ra_list, dec_list, 'o', markersize=2)

    plt.show()

    sys.exit(0)