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
    specz_sample_cat = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/large_diff_specz.txt', skip_header=1, dtype=None, names=True)
    # large_diff_specz_short.txt -- is the file containgin info for all galaxies that actually had a large diff between grism and spec z.
    # This file contains even those galaxies that don't have a match within TKRS.
    # The info in this file is copy-pasted from the result printed to the terminal by the code specz_results.py
    # large_diff_specz_short.txt -- is the file containgin info for all galaxies that had the fitting code rerun on them. 
    # The results are saved in massive-galaxies-figures/large_diff_specz_sample/
    # i.e. this file only has the matches with TKRS. This file is only used within the fitting code.

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
        	if len(idx) == 1:  # make sure that there is only one match

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

                    print current_id, current_field, zgrism, zspec, zphot, zspec_source, zspec_qual, "    TKRS stats:", tkrs_z, tkrs_zqual, alt_z
                    # Comment out the "TKRS stats" part if you want to generate teh large_diff_specz_short file
                    tkrs_compare_count += 1

                    # PLot TKRS spectrum
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    # labels
                    ax.set_xlabel(r'$\mathrm{Wavelength\ [\AA]}$')
                    ax.set_ylabel(r'$\mathrm{Counts/sec}$')

                    # smooth and plot
                    gauss = Gaussian1DKernel(stddev=8)
                    spec = convolve(spec, gauss)

                    # put in rest frame using tkrs_z
                    if np.isfinite(tkrs_z):
                        redshift = tkrs_z
                    else:
                        redshift = zspec
                        
                    lam_rest = lam / (1 + redshift)    
                    spec_rest = spec * (1 + redshift)

                    ax.plot(lam, spec, '-', color='k')

                    # plot emission line wavelengths
                    ybottom, ytop = ax.get_ylim()

                    em_lines_to_display = np.array([mgii, ni_1, oii, h_delta, h_gamma, h_beta, oiii_1, oiii_2])
                    em_lines_to_display = em_lines_to_display * (1+redshift)
                    em_line_labels = [r'$\mathrm{MgII]}$', r'$\mathrm{[NI]}$', \
                    r'$\mathrm{[OII]}$', r'$\mathrm{H\delta}$', r'$\mathrm{H\gamma}$', r'$\mathrm{H\beta}$', \
                    r'$\mathrm{[OIII]}$', r'$\mathrm{[OIII]}$']

                    abs_lines_to_display = np.array([ca_h, ca_k, gband, mgb])
                    abs_lines_to_display = abs_lines_to_display * (1+redshift)
                    abs_line_labels = [r'$\mathrm{H}$', r'$\mathrm{K}$', r'$\mathrm{Gband}$', r'$\mathrm{MgB}$']

                    # EMISSION LINES
                    for j in range(len(em_lines_to_display)):

                        # only plot lines possible with redshift
                        if em_lines_to_display[j] > np.max(lam):
                            continue

                        if em_lines_to_display[j] < np.min(lam):
                            continue

                        # plot line location
                        ax.axvline(x=em_lines_to_display[j], ymin=0.55, ymax=0.85, lw='1.5', ls='--', color='r')

                        # Emission line labels
                        if em_lines_to_display[j] == h_beta*(1+redshift):
                            ax.text(x=em_lines_to_display[j]+7, y=ytop*0.85, s=em_line_labels[j], fontsize=10)
                        elif em_lines_to_display[j] == oiii_2*(1+redshift):
                            ax.text(x=em_lines_to_display[j]+7, y=ytop*0.75, s=em_line_labels[j], fontsize=10)
                        else:
                            ax.text(x=em_lines_to_display[j]+7, y=ytop*0.8, s=em_line_labels[j], fontsize=10)

                    # ABSORPTION LINES
                    for k in range(len(abs_lines_to_display)):

                        # only plot lines possible with redshift
                        if abs_lines_to_display[k] > np.max(lam):
                            continue

                        if abs_lines_to_display[k] < np.min(lam):
                            continue

                        # plot line location
                        ax.axvline(x=abs_lines_to_display[k], ymin=0.1, ymax=0.25, lw='1.5', ls='--', color='r')
                        # line label
                        if abs_lines_to_display[k] == ca_k*(1+redshift):
                            ax.text(x=abs_lines_to_display[k]-100, y=ytop*-0.04, s=abs_line_labels[k], fontsize=10)
                        elif abs_lines_to_display[k] == ca_h*(1+redshift):
                            ax.text(x=abs_lines_to_display[k]+20, y=ytop*-0.04, s=abs_line_labels[k], fontsize=10)
                        else:
                            ax.text(x=abs_lines_to_display[k], y=ytop*-0.04, s=abs_line_labels[k], fontsize=10)

                    # other plot commands
                    ax.minorticks_on()
                    ax.text(0.05, 0.97, 'TKRS zspec quality flag: ' + str(tkrs_zqual), verticalalignment='top', horizontalalignment='left', 
                        transform=ax.transAxes, color='k', size=12)

                    # reset lims
                    ax.set_ylim(-50, ytop)

                    # Save fig
                    fig.savefig(massive_figures_dir + 'large_diff_specz_sample/' + \
                        current_field + '_' + str(current_id) + '_TKRS.png', dpi=300, bbox_inches='tight')
                    #plt.show()

                    plt.clf()
                    plt.cla()
                    plt.close()

                    spec_hdu.close()

            else:  # i.e. if there is more than 1 match found
                print "Exiting. More than 1 match found for PEARS ID:", current_id, idx
                print "RA, DEC:", ra, dec
                sys.exit(0)

    print "Total to compare:", tkrs_compare_count

    sys.exit(0)