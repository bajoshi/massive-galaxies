from __future__ import division

import numpy as np
from astropy.io import fits
import math
from astropy.convolution import convolve, Gaussian1DKernel

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
tkrs_dir = home + "/Desktop/FIGS/tkrs_spectra/"

sys.path.append(massive_galaxies_dir)
import matching as mt

def convert_tkrs_ra_to_deg(coord_arr):

    # Create empty list
    conv_list = []

    # Loop and convert
    for i in range(len(coord_arr)):

        ra_hour_frac, ra_hour = math.modf(coord_arr[i])
        ra_hour_frac *= 60
        ra_min_frac, ra_min = math.modf(ra_hour_frac)
        ra_sec = ra_min_frac * 60

        ra_deg = mt.convert_sex2deg(ra_hour, ra_min, ra_sec)

        conv_list.append(ra_deg)

    # Convert to numpy array
    conv_list = np.asarray(conv_list)

    return conv_list

if __name__ == '__main__':

    # Read TKRS catalog
    cat = fits.open(home + '/Desktop/FIGS/tkrs_spectra/tkrs_by_ra.fits')

    # assign arrays
    tkrs_ra = convert_tkrs_ra_to_deg(cat[1].data['ra'])  # Convert to degrees
    tkrs_dec = cat[1].data['dec']

    # Read in catalog for large differences
    specz_sample_cat = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/large_diff_specz.txt', dtype=None, names=True)

    # Define emission line wavelengths (in air; angstroms)
    lya = 1216
    nv = 1240
    siiv = 1400
    civ = 1549
    heii_1 = 1640
    ciii = 1909
    cii = 2326
    mgii = 2798
    nev = 3426
    ni_1 = 3467
    oii = 3727
    neiii_1 = 3869
    neiii_2 = 3967
    h_epsilon = 3970
    h_delta = 4102
    h_gamma = 4340
    oiii_0 = 4363
    heii_2 = 4686
    h_beta = 4861
    oiii_1 = 4959
    oiii_2 = 5007
    ni_2 = 5198
    hei = 5876
    oi_1 = 6300
    oi_2 = 6364
    nii_1 = 6548
    h_alpha = 6563
    nii_2 = 6583
    sii_1 = 6716
    sii_2 = 6731

    # Define absorption line wavelengths
    ca_h = 3968
    ca_k = 3933
    gband = 4308
    mgb = 5175
    fe_1 = 5270
    fe_2 = 5335

    # Loop over specz sample outliers and check their TKRS spectra
    tkrs_compare_count = 0
    for i in range(len(specz_sample_cat)):

        ra = specz_sample_cat['ra'][i]
        dec = specz_sample_cat['dec'][i]

        zgrism = specz_sample_cat['zgrism'][i]
        zspec = specz_sample_cat['zspec'][i]
        zphot = specz_sample_cat['zphot'][i]
        zspec_source = specz_sample_cat['zsource'][i]
        zspec_qual = specz_sample_cat['zqual'][i]

        current_id = specz_sample_cat['pearsid'][i]
        current_field = specz_sample_cat['field'][i]

        # match
        ra_tol = 1/3600  # in arcseconds
        dec_tol = 1/3600

        idx = np.where((ra >= tkrs_ra - ra_tol) & (ra <= tkrs_ra + ra_tol) & \
            (dec >= tkrs_dec - dec_tol) & (dec <= tkrs_dec + dec_tol))[0]

        # If a match is found 
        # Read in TKRS spectrum file and plot
        if idx.size:
            if cat[1].data['mask'][idx] != -2147483647:

                # Get filename
                maskname = '0' + str(cat[1].data['mask'][idx][0])
                if len(maskname) > 2:
                    maskname = maskname.lstrip('0')

                slitname = '0' + str(cat[1].data['slit'][idx][0])
                idname = str(cat[1].data['ID'][idx][0])

                maskdir = tkrs_dir + 'KTRS' + maskname + '/'
                filebasename = 'spec1d.KTRS' + maskname + '.' + slitname + '.' + idname + '.fits'
                filename = maskdir + filebasename

                # Get data
                spec_hdu = fits.open(filename)
                lam = np.concatenate((spec_hdu[1].data['LAMBDA'].ravel(), spec_hdu[2].data['LAMBDA'].ravel()))
                spec = np.concatenate((spec_hdu[1].data['SPEC'].ravel(), spec_hdu[2].data['SPEC'].ravel()))

                tkrs_z = float(cat[1].data['z'][idx])
                tkrs_zqual = int(cat[1].data['zq'][idx])
                alt_z = float(cat[1].data['z_alt'][idx])

                print current_id, current_field, zgrism, zspec, zphot, zspec_source, zspec_qual#, "    TKRS stats:", tkrs_z, tkrs_zqual, alt_z
                tkrs_compare_count += 1

                # PLot TKRS spectrum
                """
                fig = plt.figure()
                ax = fig.add_subplot(111)

                # smooth and plot
                gauss = Gaussian1DKernel(stddev=5)
                spec = convolve(spec, gauss)

                # put in rest frame using tkrs_z
                if np.isfinite(tkrs_z):
                    lam_rest = lam / (1 + tkrs_z)
                    spec_rest = spec * (1 + tkrs_z)
                else:
                    lam_rest = lam / (1 + zspec)
                    spec_rest = spec * (1 + zspec)

                ax.plot(lam_rest, spec_rest, '-', color='k')

                # plot emission line wavelengths
                ax.axvline(x=mgii, ymin=0.65, ymax=0.85, lw='1.5', ls='--', color='r')
                ax.axvline(x=ni_1, ymin=0.65, ymax=0.85, lw='1.5', ls='--', color='r')
                ax.axvline(x=oii, ymin=0.65, ymax=0.85, lw='1.5', ls='--', color='r')
                ax.axvline(x=h_beta, ymin=0.65, ymax=0.85, lw='1.5', ls='--', color='r')
                ax.axvline(x=oiii_1, ymin=0.65, ymax=0.85, lw='1.5', ls='--', color='r')
                ax.axvline(x=oiii_2, ymin=0.65, ymax=0.85, lw='1.5', ls='--', color='r')

                plt.show()

                plt.clf()
                plt.cla()
                plt.close()
                """

    print "Total to compare:", tkrs_compare_count

    sys.exit(0)