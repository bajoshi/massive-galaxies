from __future__ import division

from astropy.convolution import convolve, convolve_fft
import numpy as np
import numpy.ma as ma
from astropy.io import fits
from scipy.constants import c

import os
import sys
import time

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/fits_comp_spectra/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
new_codes_dir = home + "/Desktop/FIGS/new_codes/"
lsf_dir = new_codes_dir

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
import grid_coadd as gd
import fast_chi2_jackknife as fcj
from fast_chi2_jackknife_massive_galaxies import create_bc03_lib_main
import cosmology_calculator as cc

if __name__ == '__main__':

    # starting time
    start = time.time()

    # Read the matched galaxies catalog between 3DHST and PEARS
    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt', dtype=None, names=True, skip_header=2)
    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    # Find the indices for massive galaxies
    massive_galaxies_indices = np.where(stellarmass >= 11)[0]

    # Only use the galaxy ids that are unique
    pears_unique_ids, pears_unique_ids_indices = np.unique(pears_id[massive_galaxies_indices], return_index=True)
 
    # Pick a random galaxy
    current_pears_index = pears_unique_ids[4]
    count = pears_unique_ids_indices[4]
    
    # Find the redshift and other rest frame quants for the chosen galaxy
    redshift = photz[massive_galaxies_indices][count]
    lam_em, flam_em, ferr, specname = gd.fileprep(current_pears_index, redshift)
    
    ####### ------------------- with lsf ------------------- #######
    resampling_lam_grid = lam_em
    if not os.path.isfile(savefits_dir + 'all_comp_spectra_bc03_solar_withlsf_' + str(current_pears_index) + '.fits'):
        # create model library that is adapted to the specific galaxy
        create_bc03_lib_main(resampling_lam_grid, current_pears_index, redshift)

    # read in the model libraries
    bc03_spec = fits.open(savefits_dir + 'all_comp_spectra_bc03_solar_withlsf_' + str(current_pears_index) + '.fits')
    bc03_extens = fcj.get_total_extensions(bc03_spec)

    # Create and initialize numpy arrays using the adapted model library for faster chi2 computation
    comp_spec_bc03 = np.zeros([bc03_extens, len(resampling_lam_grid)], dtype=np.float64)
    for i in range(bc03_extens):
        comp_spec_bc03[i] = bc03_spec[i+1].data

    # Get random samples by jackknifing
    num_samp_to_draw = int(1e3)
    print "Running over", num_samp_to_draw, "random jackknifed samples."
    resampled_spec = ma.empty((len(flam_em), num_samp_to_draw))
    for i in range(len(flam_em)):
        if flam_em[i] is not ma.masked:
            resampled_spec[i] = np.random.normal(flam_em[i], ferr[i], num_samp_to_draw)
        else:
            resampled_spec[i] = ma.masked
    resampled_spec = resampled_spec.T

    # Do the actual chi2 fitting
    ages_bc03, metals_bc03, tau_bc03, tauv_bc03, exten_bc03, chi2_bc03, alpha_bc03 = fcj.fit_chi2(flam_em, ferr, comp_spec_bc03, bc03_extens, resampled_spec, num_samp_to_draw, 'bc03', bc03_spec)

    print "best age", np.median(ages_bc03)
    print "best tau", np.median(tau_bc03)

    # find the extension that corresponds to the best fit and read in the data from the corresponding model
    best_exten = int(np.median(exten_bc03))
    currentspec = bc03_spec[best_exten].data

    # find vertical scaling factor that minimizes chi2
    alpha = np.sum(flam_em * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)
    best_alpha = np.median(alpha_bc03)
    print best_alpha, alpha, stellarmass[massive_galaxies_indices][count]

    mpc, H_0, omega_m0, omega_r0, omega_lam0, year = cc.get_cosmology_params()
    dp = 1e-3*c*cc.proper_distance(H_0, omega_m0, omega_r0, omega_lam0, 1/(1+redshift))[0]  # in Mpc
    dp = dp * 3.09e24  # in cm
    dl = (1 + redshift)*dp
    print alpha * (4*np.pi*dl**2) / 3.846e33
    print 10**stellarmass[massive_galaxies_indices][count] / alpha

    # make the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lam_em, flam_em, color='k')
    ax.fill_between(lam_em, flam_em + ferr, flam_em - ferr, color='lightgray')
    ax.plot(lam_em, alpha * currentspec, color='b')

    ####### ------------------- without lsf ------------------- #######
    # Read in the model spectra
    bc03_spec = fits.open(home + '/Desktop/FIGS/new_codes/fits_comp_spectra/all_comp_spectra_bc03_solar_' + str(current_pears_index) + '.fits')
    bc03_extens = fcj.get_total_extensions(bc03_spec)

    # Find the extension for the best fit
    exten_bc03 = np.loadtxt(savefits_dir + 'jackknife' + str(current_pears_index) + '_exten_bc03.txt', usecols=range(int(1e3)))
    best_exten = int(np.median(exten_bc03))

    # find vertical scaling factor that minimizes chi2
    currentspec = bc03_spec[best_exten].data
    alpha = np.sum(flam_em * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)

    # put it on the the plot
    ax.plot(lam_em, alpha * currentspec, color='r')

    # total time taken
    print "total time taken", time.time() - start, "seconds."

    fig.savefig(home + '/Desktop/lsf_difference_pearsid_s65620.eps', dpi=300)

    sys.exit(0)
