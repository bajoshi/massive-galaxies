from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

def get_props(matchedfile, threedcat, figscat, ra_list, dec_list):

    for i in range(len(matchedfile)):

        # first find object in 3DHST catlaog
        ra = matchedfile['threed_ra'][i]
        dec = matchedfile['threed_dec'][i]

        lim = 0.05 * 1/3600

        threed_idx = np.where((threedcat[1].data['ra'] >= ra - lim) & (threedcat[1].data['ra'] <= ra + lim) \
            & (threedcat[1].data['dec'] >= dec - lim) & (threedcat[1].data['dec'] <= dec + lim))[0]
        # I'm using a search box because searching for exact ra and dec 
        # values doesn't work due to the way in which floats are compared. 
        # Therefore, I've chosen a very restrictive search limit.
        # This above line with the lim of 0.05 arcseconds has been tested 
        # for GN1 and GN2 and is working.
        # i.e. it will ALWAYS give you exactly one match in the 3DHST catalog.
        # i.e. used the following lines to test
        # if len(threed_idx[0]) != 1:
        #     print i, ra, dec, threed_idx
        # There is another way of doing this. See the code stacking-analysis-pears/cmd_pears.py
        # where i've used abs(threed[1].data['ra'] - threedra) < 1e-3)

        # get stellar pop properties from 3DHST 
        mstar = threedcat[1].data['lmass'][threed_idx[0]]
        zphot = threedcat[1].data['z_peak_phot'][threed_idx[0]]

        # get magnitudes from FIGS
        current_id = matchedfile['figs_id'][i]
        figs_idx = np.where(figscat['id'] == current_id)[0]

        # convert the fluxes to magnitudes
        # the fluxes are given in nJy
        f105w_flux = figscat['f105w_flux'][figs_idx[0]] * 1e-9
        f125w_flux = figscat['f125w_flux'][figs_idx[0]] * 1e-9
        f140w_flux = figscat['f140w_flux'][figs_idx[0]] * 1e-9
        f160w_flux = figscat['f160w_flux'][figs_idx[0]] * 1e-9

        f105w_mag = -2.5 * np.log10(f105w_flux / 3631)
        f125w_mag = -2.5 * np.log10(f125w_flux / 3631)
        f140w_mag = -2.5 * np.log10(f140w_flux / 3631)
        f160w_mag = -2.5 * np.log10(f160w_flux / 3631)

        # check if object can be selected
        if (zphot >= 0.5) and (zphot <= 2.0) and (mstar >= 10.5):
            if (f105w_mag <= 20.0) and np.isfinite(f105w_mag):
                print ra, dec, mstar, zphot, '{:.2f}'.format(f105w_mag), '{:.2f}'.format(f125w_mag), '{:.2f}'.format(f140w_mag), '{:.2f}'.format(f160w_mag)
                ra_list.append(ra)
                dec_list.append(dec)

    return ra_list, dec_list

if __name__ == '__main__':

    # Read in FIGS catalogs # latest version v1.2
    gn1cat = np.genfromtxt(massive_galaxies_dir + 'GN1_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec','f105w_flux','f125w_flux','f140w_flux','f160w_flux'], usecols=([2,3,4,17,19,21,23]), skip_header=25)
    gn2cat = np.genfromtxt(massive_galaxies_dir + 'GN2_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec','f105w_flux','f125w_flux','f140w_flux','f160w_flux'], usecols=([2,3,4,17,19,21,23]), skip_header=25)
    gs1cat = np.genfromtxt(massive_galaxies_dir + 'GS1_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec','f105w_flux','f125w_flux','f140w_flux','f160w_flux'], usecols=([2,3,4,17,19,21,23]), skip_header=25)
    gs2cat = np.genfromtxt(massive_galaxies_dir + 'GS2_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec','f105w_flux','f125w_flux','f160w_flux'], usecols=([2,3,4,13,15,17]), skip_header=19)
    
    # read in catalogs for matched objects between 3DHST and FIGS
    gn1_matches = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/gn1_threedhst_matches.txt', dtype=None, names=True, skip_header=1)
    gn2_matches = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/gn2_threedhst_matches.txt', dtype=None, names=True, skip_header=1)
    gs1_matches = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/gs1_threedhst_matches.txt', dtype=None, names=True, skip_header=1)
    gs2_matches = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/gs2_threedhst_matches.txt', dtype=None, names=True, skip_header=1)

    # read in 3DHST master catalog
    threedcat = fits.open(home + '/Documents/3D-HST/3dhst.v4.1.5.master.fits')

    # loop over all matches in all fields and select an object if it passes all criteria
    ra_list = []
    dec_list = []

    ra_list, dec_list = get_props(gn1_matches, threedcat, gn1cat, ra_list, dec_list)
    ra_list, dec_list = get_props(gn2_matches, threedcat, gn2cat, ra_list, dec_list)

    # show objects on sky
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(ra_list, dec_list, 'o', markersize=2)

    plt.show()

    sys.exit(0)