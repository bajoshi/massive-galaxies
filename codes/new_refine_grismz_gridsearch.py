from __future__ import division

import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, convolve, Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
import grid_coadd as gd
import fast_chi2_jackknife_massive_galaxies as fcjm
import new_refine_grismz_iter as ni
import refine_redshifts_dn4000 as old_ref

def get_chi2(flam, ferr, object_lam_grid, model_comp_spec, model_resampling_lam_grid):

    # chop the model to be consistent with the objects lam grid
    model_lam_grid_indx_low = np.argmin(abs(model_resampling_lam_grid - object_lam_grid[0]))
    model_lam_grid_indx_high = np.argmin(abs(model_resampling_lam_grid - object_lam_grid[-1]))
    model_spec_in_objlamgrid = model_comp_spec[:, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # make sure that the arrays are the same length
    if int(model_spec_in_objlamgrid.shape[1]) != len(object_lam_grid):
        print "Arrays of unequal length. Must be fixed before moving forward. Exiting..."
        sys.exit(0)

    alpha_ = np.sum(flam * model_spec_in_objlamgrid / (ferr**2), axis=1) / np.sum(model_spec_in_objlamgrid**2 / ferr**2, axis=1)
    chi2_ = np.sum(((flam - (alpha_ * model_spec_in_objlamgrid.T).T) / ferr)**2, axis=1)

    return chi2_, alpha_

def do_fitting(flam_obs, ferr_obs, lam_obs, lsf, starting_z, resampling_lam_grid, \
        model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start_time):
    """
    All models are redshifted to each of the redshifts in the list defined below,
    z_arr_to_check. Then the model modifications are done at that redshift.

    For each iteration through the redshift list it computes a chi2 for each model.
    So there are 
    """

    # Set up redshift grid to check
    z_arr_to_check = np.asarray([0.77]) #np.linspace(starting_z - 0.2, starting_z + 0.2, 41)
    print "Will check the following redshifts:", z_arr_to_check

    # Loop over all redshifts to check
    # set up chi2 array
    chi2 = np.empty((len(z_arr_to_check), total_models))
    alpha = np.empty((len(z_arr_to_check), total_models))

    # looping
    count = 0
    for z in z_arr_to_check:

        print "\n", "Currently at redshift:", z

        # first modify the models at the current redshift to be able to compare with data
        model_comp_spec_modified = \
        ni.do_model_modifications(lam_obs, model_lam_grid, model_comp_spec, resampling_lam_grid, total_models, lsf, z)
        print "Model mods done at current z:", z, "\n", "Total time taken up to now --", time.time() - start_time, "seconds."

        # Now do the chi2 computation
        chi2[count], alpha[count] = get_chi2(flam_obs, ferr_obs, lam_obs, model_comp_spec, resampling_lam_grid)

        count += 1

    # Find the minimum chi2
    min_idx = np.argmin(chi2)
    min_idx_2d = np.unravel_index(min_idx, chi2.shape)

    print "Minimum chi2:", "{:.2}".format(chi2[min_idx_2d])
    print "New redshift:", z_arr_to_check[min_idx_2d[0]]

    # Get the best fit model parameters
    model_idx = int(min_idx_2d[1])

    age = bc03_all_spec_hdulist[model_idx + 1].header['LOG_AGE']
    # now check if the best fit model is an ssp or csp 
    # only the csp models have tau and tauV parameters
    # so if you try to get these keywords for the ssp fits files
    # it will fail with a KeyError
    if 'TAU_GYR' in list(bc03_all_spec_hdulist[model_idx + 1].header.keys()):
        tau = float(bc03_all_spec_hdulist[model_idx + 1].header['TAU_GYR'])
        tauv = float(bc03_all_spec_hdulist[model_idx + 1].header['TAUV'])
    else:
        # if the best fit model is an SSP then assign -99.0 to tau and tauV
        tau = -99.0
        tauv = -99.0

    print "Current best fit log(age [yr]):", "{:.2}".format(age)
    print "Current best fit Tau [Gyr]:", "{:.2}".format(tau)
    print "Current best fit Tau_V:", tauv

    # get things needed to plot and plot
    bestalpha = alpha[min_idx_2d]
    # chop model again to get the part within objects lam obs grid
    model_lam_grid_indx_low = np.argmin(abs(resampling_lam_grid - lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(abs(resampling_lam_grid - lam_obs[-1]))
    best_fit_model_in_objlamgrid = model_comp_spec[model_idx, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # again make sure that the arrays are the same length
    if int(best_fit_model_in_objlamgrid.shape[0]) != len(lam_obs):
        print "Arrays of unequal length. Must be fixed before moving forward. Exiting..."
        sys.exit(0)
    # plot
    ni.plot_fit_and_residual(lam_obs, flam_obs, ferr_obs, best_fit_model_in_objlamgrid, bestalpha)

    # This chi2 map can also be visualized as an image. 
    # Run imshow() and check what it looks like.
    plt.imshow(chi2, cmap='viridis')
    plt.show()

    return None

if __name__ == '__main__':
    """
    The __main__ part of this code is almost exactly similar to that in new_refine_grismz_iter.py

    """
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # --------------------------------------------- GET OBS DATA ------------------------------------------- #
    current_id = 69419
    current_field = 'GOODS-S'
    redshift = 0.9  # photo-z estimate
    # for 61447 GOODS-S
    # 0.84 Ferreras+2009  # candels 0.976 # 3dhst 0.9198
    # for 65620 GOODS-S
    # 0.97 Ferreras+2009  # candels 0.972 # 3dhst 0.9673 # spec-z 0.97
    lam_em, flam_em, ferr_em, specname, pa_chosen, netsig_chosen = gd.fileprep(current_id, redshift, current_field)

    # now make sure that the quantities are all in observer frame
    # the original function (fileprep in grid_coadd) will give quantities in rest frame
    # but I need these to be in the observed frame so I will redshift them again.
    # It gives them in the rest frame because getting the d4000 and the average 
    # lambda separation needs to be figured out in the rest frame.
    flam_obs = flam_em / (1 + redshift)
    ferr_obs = ferr_em / (1 + redshift)
    lam_obs = lam_em * (1 + redshift)

    # plot to check # Line useful for debugging. Do not remove. Just uncomment.
    #ni.plotspectrum(lam_obs, flam_obs, ferr_obs)
    # --------------------------------------------- Quality checks ------------------------------------------- #

    # ---------------------------------------------- MODELS ----------------------------------------------- #
    # put all models into one single fits file
    #ni.get_model_set()

    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_comp_spec = np.zeros([total_models, len(model_lam_grid)], dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec_hdulist[j+1].data

    # total run time up to now
    print "All models put in numpy array. Total time taken up to now --", time.time() - start, "seconds."

    # ---------------------------------------------- FITTING ----------------------------------------------- #
    # Read in LSF
    if current_field == 'GOODS-N':
        lsf_filename = lsfdir + "north_lsfs/" + "n" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"
    elif current_field == 'GOODS-S':
        lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"

    # read in LSF file
    lsf = np.loadtxt(lsf_filename)

    # extend lam_grid to be able to move the lam_grid later 
    avg_dlam = old_ref.get_avg_dlam(lam_obs)

    lam_low_to_insert = np.arange(5000, lam_obs[0], avg_dlam)
    lam_high_to_append = np.arange(lam_obs[-1] + avg_dlam, 10500, avg_dlam)

    resampling_lam_grid = np.insert(lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    # call actual fitting function
    do_fitting(flam_obs, ferr_obs, lam_obs, lsf, redshift, resampling_lam_grid, \
        model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start)

    sys.exit(0)
