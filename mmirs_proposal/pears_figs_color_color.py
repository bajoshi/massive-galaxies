from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

if __name__ == '__main__':
    """
    Only selecting using FIGS for now.
    """

    # Read in FIGS catalogs # latest version v1.2
    # All fluxes are in nJy
    gn1cat = np.genfromtxt(massive_galaxies_dir + 'GN1_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec','f814w_flux','f105w_flux','f125w_flux','f140w_flux','f160w_flux'], \
                           usecols=([2,3,4,13,17,19,21,23]), skip_header=25)
    gn2cat = np.genfromtxt(massive_galaxies_dir + 'GN2_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec','f814w_flux','f105w_flux','f125w_flux','f140w_flux','f160w_flux'], \
                           usecols=([2,3,4,13,17,19,21,23]), skip_header=25)

    # Convert fluxes to magnitudes # all AB magnitudes
    gn1_f814w_mag = -2.5 * np.log10(gn1cat['f814w_flux'] * 1e-9) + 8.9
    gn1_f105w_mag = -2.5 * np.log10(gn1cat['f105w_flux'] * 1e-9) + 8.9
    gn1_f125w_mag = -2.5 * np.log10(gn1cat['f125w_flux'] * 1e-9) + 8.9
    gn1_f140w_mag = -2.5 * np.log10(gn1cat['f140w_flux'] * 1e-9) + 8.9
    gn1_f160w_mag = -2.5 * np.log10(gn1cat['f160w_flux'] * 1e-9) + 8.9

    gn2_f814w_mag = -2.5 * np.log10(gn2cat['f814w_flux'] * 1e-9) + 8.9
    gn2_f105w_mag = -2.5 * np.log10(gn2cat['f105w_flux'] * 1e-9) + 8.9
    gn2_f125w_mag = -2.5 * np.log10(gn2cat['f125w_flux'] * 1e-9) + 8.9
    gn2_f140w_mag = -2.5 * np.log10(gn2cat['f140w_flux'] * 1e-9) + 8.9
    gn2_f160w_mag = -2.5 * np.log10(gn2cat['f160w_flux'] * 1e-9) + 8.9

    # Apply color cuts
    color_cut_idx_gn1 = np.where((gn1_f160w_mag < 24) & (gn1_f105w_mag - gn1_f125w_mag >= 1.0))[0]
    color_cut_idx_gn2 = np.where((gn2_f160w_mag < 24) & (gn2_f105w_mag - gn2_f125w_mag >= 1.0))[0]

    # make arrays of selected objects which passed the cuts
    obj_figsid = []
    obj_ra = []
    obj_dec = []
    obj_f160w_mag = []
    obj_105_125_color = []

    for i in range(len(color_cut_idx_gn1)):
        f105_125_color = gn1_f105w_mag[color_cut_idx_gn1[i]] - gn1_f125w_mag[color_cut_idx_gn1[i]]
        if not np.isfinite(f105_125_color):
            continue
        else:
            #print gn1cat['id'][color_cut_idx_gn1[i]], gn1cat['ra'][color_cut_idx_gn1[i]], \
            # gn1cat['dec'][color_cut_idx_gn1[i]], gn1_f160w_mag[color_cut_idx_gn1[i]]

            obj_figsid.append(gn1cat['id'][color_cut_idx_gn1[i]])
            obj_ra.append(gn1cat['ra'][color_cut_idx_gn1[i]])
            obj_dec.append(gn1cat['dec'][color_cut_idx_gn1[i]])
            obj_f160w_mag.append(gn1_f160w_mag[color_cut_idx_gn1[i]])
            obj_105_125_color.append(f105_125_color)

    for i in range(len(color_cut_idx_gn2)):
        f105_125_color = gn2_f105w_mag[color_cut_idx_gn2[i]] - gn2_f125w_mag[color_cut_idx_gn2[i]]
        if not np.isfinite(f105_125_color):
            continue
        else:
            #print gn2cat['id'][color_cut_idx_gn2[i]], gn2cat['ra'][color_cut_idx_gn2[i]], \
            # gn2cat['dec'][color_cut_idx_gn2[i]], gn2_f160w_mag[color_cut_idx_gn2[i]]

            obj_figsid.append(gn2cat['id'][color_cut_idx_gn2[i]])
            obj_ra.append(gn2cat['ra'][color_cut_idx_gn2[i]])
            obj_dec.append(gn2cat['dec'][color_cut_idx_gn2[i]])
            obj_f160w_mag.append(gn2_f160w_mag[color_cut_idx_gn2[i]])
            obj_105_125_color.append(f105_125_color)

    # Match with MOSDEF catalog to make sure that they havent already taken data on this object
    # Match with Barger+ 2008 catalog to get Ks magnitudes whcih should be <21 for you to be able to obseve with MMIRS
    # Read in MOSDEF and Barger catalogs
    mosdef_cat = np.genfromtxt(home + '/Desktop/MOSDEF_survey_final_redshift_release.txt', \
        dtype=None, names=True, skip_header=1)
    
    # the Barger catalog has to be read line by line because it has gaps and can't be read with genfromtxt
    with open(home + '/Desktop/barger_2008_specz.cat') as f:
        lines = f.readlines()

        # initialize arrays
        barger_ra = []
        barger_dec = []
        barger_zspec = []
        barger_kflux = []
        barger_kmag = []
        count = 0

        # loop line by line
        for line in lines[32:]:  # skipe 32 lines of header
            a = line.split()
            if len(a) > 11:
                if a[11] == 's' or a[11] == 'n':
                    continue
                else:
                    barger_ra.append(float(a[1]))
                    barger_dec.append(float(a[2]))
                    barger_zspec.append(float(a[11]))
                    barger_kflux.append(float(a[3]))
                    barger_kmag.append(float(a[5]))

    # convert to numpy arrays
    barger_ra = np.asarray(barger_ra)
    barger_dec = np.asarray(barger_dec)
    barger_zspec = np.asarray(barger_zspec)
    barger_kflux = np.asarray(barger_kflux)
    barger_kmag = np.asarray(barger_kmag)

    # Read in 3DHST catalog as well to check what the stellar masses are
    threed_v41_phot = fits.open(home + '/Desktop/FIGS/new_codes/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat.fits')

    # Now match and take out the matches
    # ONLY if they are in the MOSDEF catalog 
    # and append to final arrays
    # Don't take out any matches with 3DHST and Barger et al.
    final_obj_figsid = []
    final_obj_ra = []
    final_obj_dec = []
    final_obj_f160w_mag = []
    final_obj_105_125_color = []
    final_obj_stellarmass = []
    final_obj_zphot = []

    for i in range(len(obj_ra)):

        current_ra = obj_ra[i]
        current_dec = obj_dec[i]
        current_id = obj_figsid[i]
        current_f160w_mag = obj_f160w_mag[i]

        # checked with a tolerance of 1.0 arcseconds as well
        # got the exact same result
        mosdef_idx = np.where((abs(mosdef_cat['RA'] - current_ra) < 0.5/3600) & \
            (abs(mosdef_cat['DEC'] - current_dec) < 0.5/3600))[0]
        barger_idx = np.where((abs(barger_ra - current_ra) < 0.5/3600) & (abs(barger_dec - current_dec) < 0.5/3600))[0]
        threed_idx = np.where((abs(threed_v41_phot[1].data['ra'] - current_ra) < 0.5/3600) & \
            (abs(threed_v41_phot[1].data['dec'] - current_dec) < 0.5/3600))[0]

        # In here, if I use the 3DHST z_spec column instead of z_peak then I get -1 for z_spec for all of them
        # i.e. 3DHST does not have a spec_z for these galaxies
        if threed_idx.size:
            print "3DHST match", threed_v41_phot[1].data['id'][threed_idx], threed_v41_phot[1].data['ra'][threed_idx], \
            current_ra, threed_v41_phot[1].data['dec'][threed_idx], current_dec, \
            threed_v41_phot[1].data['z_peak'][threed_idx], \
            threed_v41_phot[1].data['lmass'][threed_idx], current_f160w_mag, current_id

        if barger_idx.size:
            print "BARGER match", barger_ra[barger_idx], current_ra, \
            barger_dec[barger_idx], current_dec, barger_zspec[barger_idx], \
            barger_kmag[barger_idx], current_f160w_mag, current_id

        if mosdef_idx.size:
            #if float(mosdef_cat['Z_MOSFIRE'][mosdef_idx]) != -1.0:
            print "MOSDEF match", mosdef_cat['RA'][mosdef_idx], current_ra, \
            mosdef_cat['DEC'][mosdef_idx], current_dec, mosdef_cat['Z_MOSFIRE'][mosdef_idx], current_f160w_mag, current_id
            continue

        final_obj_figsid.append(current_id)
        final_obj_ra.append(current_ra)
        final_obj_dec.append(current_dec)
        final_obj_f160w_mag.append(current_f160w_mag)
        final_obj_105_125_color.append(obj_105_125_color[i])
        final_obj_stellarmass.append(threed_v41_phot[1].data['lmass'][threed_idx])
        final_obj_zphot.append(threed_v41_phot[1].data['z_peak'][threed_idx])

    # Reprint object list to terminal
    for i in range(len(final_obj_ra)):
        print final_obj_figsid[i], final_obj_ra[i], final_obj_dec[i], \
        final_obj_105_125_color[i], final_obj_f160w_mag[i], final_obj_stellarmass[i], final_obj_zphot[i]
    
    # Find which ones match with PEARS to plot their spectra
    # i.e. PEARS + FIGS + 3DHST side by side
    # Read PEARS north cat
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS v2.0 readme
    pears_ncat['dec'] = pears_ncat['dec'] - dec_offset_goodsn_v19

    data_path = home + "/Documents/PEARS/data_spectra_only/"

    ra_plotlist = []
    dec_plotlist = []

    for i in range(len(final_obj_ra)):

        current_ra = final_obj_ra[i]
        current_dec = final_obj_dec[i]
        current_id = final_obj_figsid[i]
        current_f160w_mag = final_obj_f160w_mag[i]

        pears_idx = np.where((abs(pears_ncat['ra'] - current_ra) < 0.5/3600) & \
            (abs(pears_ncat['dec'] - current_dec) < 0.5/3600))[0]

        if current_f160w_mag <= 23.4:
            ra_plotlist.append(current_ra)
            dec_plotlist.append(current_dec)

            #print pears_ncat['id'][pears_idx], current_id, current_ra, current_dec, current_f160w_mag
            print current_id, current_ra, current_dec, current_f160w_mag

    # Plot RA dec on sky along with GOODS-N field and MMIRS MOS FoV overlaid
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{RA}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{DEC}$', fontsize=15)   

    ax.plot(ra_plotlist, dec_plotlist, 'o', markersize=2, color='k')
    ax.scatter(final_obj_ra, final_obj_dec, s=50, edgecolors='r', facecolor='None')

    ax.set_xlim(189.45,189.05)

    plt.show()

    sys.exit(0)

    # make color-color plots
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{m_{AB}(F160W)}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{m_{AB}(F105W) - m_{AB}(F125W)}$', fontsize=15)

    # concatenate full gn1 and gn2 mags
    all_f105w_mag = np.concatenate((gn1_f105w_mag, gn2_f105w_mag))
    all_f125w_mag = np.concatenate((gn1_f125w_mag, gn2_f125w_mag))
    all_f160w_mag = np.concatenate((gn1_f160w_mag, gn2_f160w_mag))

    ax.plot(all_f160w_mag, all_f105w_mag - all_f125w_mag, 'o', color='gray', markersize=1.5, alpha=0.4)
    ax.scatter(final_obj_f160w_mag, final_obj_105_125_color, s=50, edgecolors='r', facecolor='None')

    ax.axhline(y=1.0, ls='--', lw=2, xmin=0, xmax=0.71)
    ax.axvline(x=24.0, ls='--', lw=2, ymin=0.42, ymax=1.0)

    ax.set_xlim(12,29)
    ax.set_ylim(-0.5,3.0)

    fig.savefig(home + '/Desktop/FIGS/massive-galaxies-figures/figs_105_125_160_colormag.png', dpi=150, bbox_inches='tight')

    sys.exit(0)
