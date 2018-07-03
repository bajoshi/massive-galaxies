from __future__ import division

import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
from joblib import Parallel, delayed

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import model_mods_cython_copytoedit as model_mods_cython
import refine_redshifts_dn4000 as old_ref
import dn4000_catalog as dc
import new_refine_grismz_gridsearch_parallel as ngp
import mag_hist as mh

def get_rand_err_arr():

    # GOODS-N
    id_list_gn = np.load(massive_figures_dir + '/full_run/id_list_gn.npy')
    field_list_gn = np.load(massive_figures_dir + '/full_run/field_list_gn.npy')

    # GOODS-S
    id_list_gs = np.load(massive_figures_dir + '/full_run/id_list_gs.npy')
    field_list_gs = np.load(massive_figures_dir + '/full_run/field_list_gs.npy')

    # Concatenate ID and field arrays
    id_arr = np.concatenate((id_list_gn, id_list_gs))
    field_arr = np.concatenate((field_list_gn, field_list_gs))

    # Empty list for storing average errors
    err_list = []

    # Now get data and check the error
    for i in range(len(id_arr)):

        # Get data
        lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(id_arr[i], field_arr[i])

        # Get current err and append
        current_err = np.nanmean(ferr_obs/flam_obs)
        err_list.append(current_err)

    err_arr = np.asarray(err_list)

    # Now generate a random array based on this error array
    new_rand_err_arr = np.random.choice(err_arr, size=1e4)

    # Now plot histograms for the two to compare them
    """
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.hist(err_arr, 50, range=(0, 1), color='k', histtype='step')
    ax2.hist(new_rand_err_arr, 50, range=(0, 1), color='r', histtype='step')

    plt.show()
    """

    return err_arr, new_rand_err_arr

def get_mock_spectrum(model_lam_grid, model_spectrum, test_redshift, random_err_chosen):

    # Do modifications 
    # redshift model
    lam_obs = model_lam_grid * (1 + test_redshift)
    flam_obs = model_spectrum / (1 + test_redshift)

    # Now restrict the spectrum to be within 6000 to 9500
    valid_lam_idx = np.where((lam_obs >= 6000) & (lam_obs <= 9500))[0]
    lam_obs = lam_obs[valid_lam_idx]
    flam_obs = flam_obs[valid_lam_idx]

    # Now use a fake LSF and also resample the spectrum to the grism resolution
    mock_lsf = Gaussian1DKernel(10.0)
    flam_obs = convolve(flam_obs, mock_lsf, boundary='extend')

    # resample 
    mock_resample_lam_grid = np.linspace(6000, 9500, 88)
    resampled_flam = np.zeros((len(mock_resample_lam_grid)))
    for k in range(len(mock_resample_lam_grid)):

        if k == 0:
            lam_step_high = mock_resample_lam_grid[k+1] - mock_resample_lam_grid[k]
            lam_step_low = lam_step_high
        elif k == len(mock_resample_lam_grid) - 1:
            lam_step_low = mock_resample_lam_grid[k] - mock_resample_lam_grid[k-1]
            lam_step_high = lam_step_low
        else:
            lam_step_high = mock_resample_lam_grid[k+1] - mock_resample_lam_grid[k]
            lam_step_low = mock_resample_lam_grid[k] - mock_resample_lam_grid[k-1]

        new_ind = np.where((lam_obs >= mock_resample_lam_grid[k] - lam_step_low) & \
            (lam_obs < mock_resample_lam_grid[k] + lam_step_high))[0]
        resampled_flam[k] = np.mean(flam_obs[new_ind])

    flam_obs = resampled_flam

    # multiply flam by a constant to get to some realistic flux levels
    #flam_obs *= 1e-14

    # assign a constant 5% error bar
    ferr_obs = np.ones(len(flam_obs))
    ferr_obs = random_err_chosen * flam_obs

    # put in random noise in the model
    for k in range(len(flam_obs)):
        scale = ferr_obs[k]  # i.e. std dev for normal dist in next line
        if scale < 0:
            scale = abs(scale)
        if scale == 0.0:
            scale = 1e-14  
            # I just randomly chose a small number because I was 
            # getting annoyed with the scale<=0 error causing my 
            # code to fail at unpredictable times.

        flam_obs[k] = np.random.normal(flam_obs[k], ferr_obs[k], 1)

    # plot to check it looks right
    # don't delete these lines for plotting
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Deredshift the wavelength array because the model by default is at zero redshift
    # You will also have to comment the line above that multiplies flam by a constant
    # for this plot to show up correctly 
    mock_resample_lam_grid /= (1 + test_redshift)
    ax.plot(model_lam_grid, model_spectrum)
    ax.plot(mock_resample_lam_grid, flam_obs)
    ax.fill_between(mock_resample_lam_grid, flam_obs + ferr_obs, flam_obs - ferr_obs, color='lightgray')
    ax.set_xlim(6000/(1 + test_redshift), 9500/(1 + test_redshift))
    plt.show()
    """

    return mock_resample_lam_grid, flam_obs, ferr_obs

def fit_model_and_plot(flam_obs, ferr_obs, lam_obs, lsf, starting_z, resampling_lam_grid, \
    model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start_time,\
    search_range, d4000_in, d4000_out, d4000_out_err, count):

    # Set up redshift grid to check
    z_arr_to_check = np.linspace(start=starting_z - search_range, stop=starting_z + search_range, num=81, dtype=np.float64)
    z_idx = np.where((z_arr_to_check >= 0.6) & (z_arr_to_check <= 1.235))
    z_arr_to_check = z_arr_to_check[z_idx]
    print "Will check the following redshifts:", z_arr_to_check

    ####### ------------------------------------ Main loop through redshfit array ------------------------------------ #######
    # Loop over all redshifts to check
    # set up chi2 and alpha arrays
    chi2 = np.empty((len(z_arr_to_check), total_models))
    alpha = np.empty((len(z_arr_to_check), total_models))

    # looping
    num_cores = 8
    chi2_alpha_list = Parallel(n_jobs=num_cores)(delayed(ngp.get_chi2_alpha_at_z)(z, \
    flam_obs, ferr_obs, lam_obs, model_lam_grid, model_comp_spec, resampling_lam_grid, total_models, lsf, start_time) \
    for z in z_arr_to_check)

    # the parallel code seems to like returning only a list
    # so I have to unpack the list
    for i in range(len(z_arr_to_check)):
        chi2[i], alpha[i] = chi2_alpha_list[i]

    ####### -------------------------------------- Min chi2 and best fit params -------------------------------------- #######
    # Sort through the chi2 and make sure that the age is physically meaningful
    sortargs = np.argsort(chi2, axis=None)  # i.e. it will use the flattened array to sort

    for k in range(len(chi2.ravel())):

        # Find the minimum chi2
        min_idx = sortargs[k]
        min_idx_2d = np.unravel_index(min_idx, chi2.shape)
        
        # Get the best fit model parameters
        # first get the index for the best fit
        model_idx = int(min_idx_2d[1])

        age = float(bc03_all_spec_hdulist[model_idx + 1].header['LOG_AGE'])
        current_z = z_arr_to_check[min_idx_2d[0]]
        age_at_z = cosmo.age(current_z).value * 1e9  # in yr

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

        # now check if the age is meaningful
        if (age < np.log10(age_at_z - 1e8)) and (age > 9 + np.log10(0.1)):
            # If the age is meaningful then you don't need to do anything
            # more. Just break out of the loop. the best fit parameters have
            # already been assigned to variables. This assignment is done before 
            # the if statement to make sure that there are best fit parameters 
            # even if the loop is broken out of in the first iteration.
            break

    print "Minimum chi2:", "{:.4}".format(chi2[min_idx_2d])
    z_grism = z_arr_to_check[min_idx_2d[0]]
    print "New redshift:", z_grism

    # Simply the minimum chi2 might not be right
    # Should check if the minimum is global or local
    ############# -------------------------- Errors on z and other derived params ----------------------------- #############
    min_chi2 = chi2[min_idx_2d]
    dof = len(lam_obs) - 1  # i.e. total data points minus the single fitting parameter
    chi2_red = chi2 / dof
    chi2_red_error = np.sqrt(2/dof)
    min_chi2_red = min_chi2 / dof
    print "Error in reduced chi-square:", chi2_red_error
    chi2_red_2didx = np.where((chi2_red >= min_chi2_red - chi2_red_error) & (chi2_red <= min_chi2_red + chi2_red_error))
    # use first dimension indices to get error on grism-z
    z_grism_range = z_arr_to_check[chi2_red_2didx[0]]
    low_z_lim = np.min(z_grism_range)
    upper_z_lim = np.max(z_grism_range)
    print "Min z_grism within 1-sigma error:", low_z_lim
    print "Max z_grism within 1-sigma error:", upper_z_lim

    ####### ------------------------------------------ Plotting ------------------------------------------ #######
    #### -------- Plot spectrum: Data, best fit model, and the residual --------- ####
    # get things needed to plot and plot
    bestalpha = alpha[min_idx_2d]
    print "Vertical scaling factor for best fit model:", bestalpha
    # chop model again to get the part within objects lam obs grid
    model_lam_grid_indx_low = np.argmin(abs(resampling_lam_grid - lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(abs(resampling_lam_grid - lam_obs[-1]))

    # make sure the types are correct before passing to cython code
    total_models = int(total_models)

    model_comp_spec_modified = \
    model_mods_cython.do_model_modifications(model_lam_grid, model_comp_spec, \
        resampling_lam_grid, total_models, lsf, z_grism)
    print "Model mods done (only for plotting purposes) at the new grism z:", z_grism
    print "Total time taken up to now --", time.time() - start_time, "seconds."

    best_fit_model_in_objlamgrid = model_comp_spec_modified[model_idx, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    plot_mock_fit(lam_obs, flam_obs, ferr_obs, best_fit_model_in_objlamgrid, bestalpha,\
    starting_z, z_grism, low_z_lim, upper_z_lim, min_chi2_red, d4000_in, d4000_out, d4000_out_err, count)

    return z_grism, low_z_lim, upper_z_lim

def plot_mock_fit(lam_obs, flam_obs, ferr_obs, best_fit_model_in_objlamgrid, bestalpha,\
    starting_z, grismz, low_z_lim, upper_z_lim, chi2, d4000_in, d4000_out, d4000_out_err, count):

    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[:8,:])
    ax2 = fig.add_subplot(gs[8:,:])

    ax1.set_ylabel(r'$\mathrm{f_\lambda\ [erg\,s^{-1}\,cm^{-2}\,\AA]}$')
    ax2.set_xlabel(r'$\mathrm{Wavelength\, [\AA]}$')
    ax2.set_ylabel(r'$\mathrm{\frac{f^{obs}_\lambda\ - f^{model}_\lambda}{f^{obs;error}_\lambda}}$')

    # define colors
    myblue = mh.rgb_to_hex(0, 100, 180)
    myred = mh.rgb_to_hex(214, 39, 40)  # tableau 20 red

    # plot
    ax1.plot(lam_obs, flam_obs, ls='-', color='k')
    ax1.plot(lam_obs, bestalpha*best_fit_model_in_objlamgrid, ls='-', color='r')
    ax1.fill_between(lam_obs, flam_obs + ferr_obs, flam_obs - ferr_obs, color='lightgray')

    resid_fit = (flam_obs - bestalpha*best_fit_model_in_objlamgrid) / ferr_obs
    ax2.plot(lam_obs, resid_fit, ls='-', color='k')

    # minor ticks
    ax1.minorticks_on()
    ax2.minorticks_on()

    # text for info
    ax1.text(0.45, 0.36, r'$\mathrm{D4000_{intrinsic}\, =\,}$' + "{:.4}".format(d4000_in), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.45, 0.31, r'$\mathrm{D4000_{mock}\, =\,}$' + "{:.4}".format(d4000_out) + r'$\pm$' + "{:.2}".format(d4000_out_err), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    low_zerr = grismz - low_z_lim
    high_zerr = upper_z_lim - grismz

    ax1.text(0.45, 0.26, \
    r'$\mathrm{Test\ redshift\, =\,}$' + "{:.4}".format(starting_z), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.45, 0.21, \
    r'$\mathrm{z_{grism}\, =\, }$' + \
    "{:.4}".format(grismz) + r'$\substack{+$' + "{:.3}".format(low_zerr) + r'$\\ -$' + "{:.3}".format(high_zerr) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.45, 0.12, r'$\mathrm{\chi^2_{red}\, =\, }$' + "{:.3}".format(chi2), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    # add horizontal line to residual plot
    ax2.axhline(y=0.0, ls='--', color=myblue)

    fig.savefig(massive_figures_dir + 'model_mockspectra_fits/model_mocksim_' + str(count) + '.png', \
        dpi=300, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    return None

def save_intermediate_results(d4000_in_list, d4000_out_list, d4000_out_err_list, mock_model_index_list, \
    test_redshift_list, mock_zgrism_list, mock_zgrism_lowerr_list, mock_zgrism_higherr_list, \
    model_age_list, model_metallicity_list, model_tau_list, model_av_list, count_list):

    # convert to numpy arrays and save
    d4000_in_list = np.asarray(d4000_in_list) 
    d4000_out_list = np.asarray(d4000_out_list)
    d4000_out_err_list = np.asarray(d4000_out_err_list) 
    mock_model_index_list = np.asarray(mock_model_index_list)
    test_redshift_list = np.asarray(test_redshift_list)
    mock_zgrism_list = np.asarray(mock_zgrism_list)
    mock_zgrism_lowerr_list = np.asarray(mock_zgrism_lowerr_list)
    mock_zgrism_higherr_list = np.asarray(mock_zgrism_higherr_list)
    model_age_list = np.asarray(model_age_list)
    model_metallicity_list = np.asarray(model_metallicity_list)
    model_tau_list = np.asarray(model_tau_list)
    model_av_list = np.asarray(model_av_list)
    count_list = np.asarray(count_list)

    # save
    d4000_range = '_1p2to1p4'
    model_mocksim_dir = massive_figures_dir + 'model_mockspectra_fits/'
    np.save(model_mocksim_dir + 'intermediate_d4000_in_list' + d4000_range + '.npy', d4000_in_list)
    np.save(model_mocksim_dir + 'intermediate_d4000_out_list' + d4000_range + '.npy', d4000_out_list)
    np.save(model_mocksim_dir + 'intermediate_d4000_out_err_list' + d4000_range + '.npy', d4000_out_err_list)
    np.save(model_mocksim_dir + 'intermediate_mock_model_index_list' + d4000_range + '.npy', mock_model_index_list)
    np.save(model_mocksim_dir + 'intermediate_test_redshift_list' + d4000_range + '.npy', test_redshift_list)
    np.save(model_mocksim_dir + 'intermediate_mock_zgrism_list' + d4000_range + '.npy', mock_zgrism_list)
    np.save(model_mocksim_dir + 'intermediate_mock_zgrism_lowerr_list' + d4000_range + '.npy', mock_zgrism_lowerr_list)
    np.save(model_mocksim_dir + 'intermediate_mock_zgrism_higherr_list' + d4000_range + '.npy', mock_zgrism_higherr_list)
    np.save(model_mocksim_dir + 'intermediate_model_age_list' + d4000_range + '.npy', model_age_list)
    np.save(model_mocksim_dir + 'intermediate_model_metallicity_list' + d4000_range + '.npy', model_metallicity_list)
    np.save(model_mocksim_dir + 'intermediate_model_tau_list' + d4000_range + '.npy', model_tau_list)
    np.save(model_mocksim_dir + 'intermediate_model_av_list' + d4000_range + '.npy', model_av_list)
    np.save(model_mocksim_dir + 'intermediate_count_list' + d4000_range + '.npy', count_list)

    return None

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # ---------------------------------------------- MODELS ----------------------------------------------- #
    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_lam_grid = model_lam_grid.astype(np.float64)
    model_comp_spec = np.zeros((total_models, len(model_lam_grid)), dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec_hdulist[j+1].data

    # total run time up to now
    print "All models put in numpy array. Total time taken up to now --", time.time() - start, "seconds."

    # ---------------------------------------------- Loop ----------------------------------------------- #
    # Get errror distribution of real galaxies
    err_arr, new_rand_err_arr = get_rand_err_arr()

    d4000_in_list = []  # D4000 measured on model before doing model modifications
    d4000_out_list = []  # D4000 measured on model after doing model modifications
    d4000_out_err_list = []

    # Create lists for saving later
    mock_model_index_list = []
    test_redshift_list = []
    mock_zgrism_list = []
    mock_zgrism_lowerr_list = []
    mock_zgrism_higherr_list = []
    model_age_list  = []
    model_metallicity_list = []
    model_tau_list = []
    model_av_list = []
    count_list = []

    galaxy_count = 0

    for i in range(15000):

        # Measure D4000 before doing modifications
        model_flam = model_comp_spec[i]
        model_ferr = np.zeros(len(model_flam))

        d4000_in, d4000_in_err = dc.get_d4000(model_lam_grid, model_flam, model_ferr, interpolate_flag=False)

        # Randomly pick a test redshift within 0.6 <= z <= 1.235
        # Make sure that the age of the model is not older than 
        # the age of the Universe at the chosen redshift
        model_age = 10**(float(bc03_all_spec_hdulist[i + 1].header['LOG_AGE'])) / 1e9  # in Gyr
        upper_z_lim_age = z_at_value(cosmo.age, model_age * u.Gyr)

        if upper_z_lim_age <= 0.6:
            print "Skipping this model because model age is older than the oldest age",
            print "we can probe with the ACS grism, i.e. the age corresponding",
            print "to the lowest redshift at which we can measure the 4000A break (z=0.6)."
            continue

        if upper_z_lim_age > 1.235:
            upper_z_lim_age = 1.235

        test_redshift = np.random.uniform(0.6, upper_z_lim_age, 1)
        if type(test_redshift) is np.ndarray:
            test_redshift = np.asscalar(test_redshift)

        # Modify model and create mock spectrum
        # Get random error that follows the error distribution of the real galaxies
        random_err_chosen = float(np.random.choice(err_arr))
        lam_obs, flam_obs, ferr_obs = get_mock_spectrum(model_lam_grid, model_comp_spec[i], test_redshift, random_err_chosen)

        # Now de-redshift and find D4000
        lam_em = lam_obs / (1 + test_redshift)
        flam_em = flam_obs * (1 + test_redshift)
        ferr_em = ferr_obs * (1 + test_redshift)

        d4000_out, d4000_out_err = dc.get_d4000(lam_em, flam_em, ferr_em)

        # Check D4000 value and only then proceed
        if d4000_out >= 1.2 and d4000_out < 1.4:
            d4000_in_list.append(d4000_in)
            d4000_out_list.append(d4000_out)
            d4000_out_err_list.append(d4000_out_err)
            print "Galaxies done so far:", galaxy_count, "Currently at model #", i+1, "with test redshift", test_redshift
            print "Model has intrinsic D4000:", d4000_in
            print "Simulated mock spectrum has D4000:", d4000_out

            # -------- Broaden the LSF ------- #
            broad_lsf = Gaussian1DKernel(10.0 * 1.118)

            # ------- Make new resampling grid ------- # 
            # extend lam_grid to be able to move the lam_grid later 
            avg_dlam = old_ref.get_avg_dlam(lam_obs)

            lam_low_to_insert = np.arange(5000, lam_obs[0], avg_dlam, dtype=np.float64)
            lam_high_to_append = np.arange(lam_obs[-1] + avg_dlam, 10500, avg_dlam, dtype=np.float64)

            resampling_lam_grid = np.insert(lam_obs, obj=0, values=lam_low_to_insert)
            resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

            # define other stuff for plot
            current_zspec = -99.0
            current_zphot = -99.0
            current_zgrism = test_redshift

            # Fit 
            mock_zgrism, mock_zgrism_lowlim, mock_zgrism_highlim = \
            fit_model_and_plot(flam_obs, ferr_obs, lam_obs, broad_lsf.array, test_redshift, resampling_lam_grid, \
            model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start,\
            0.2, d4000_in, d4000_out, d4000_out_err, i+1)

            # Convert limits to errors
            mock_zgrism_lowerr = mock_zgrism - mock_zgrism_lowlim
            mock_zgrism_higherr = mock_zgrism_highlim - mock_zgrism

            # Append to lists and save them as numpy arrays later
            mock_model_index_list.append(i+1)
            test_redshift_list.append(test_redshift)
            mock_zgrism_list.append(mock_zgrism)
            mock_zgrism_lowerr_list.append(mock_zgrism_lowerr)
            mock_zgrism_higherr_list.append(mock_zgrism_higherr)

            # Find model params and save them too
            model_age_list.append(model_age)
            model_met = float(bc03_all_spec_hdulist[i + 1].header['METAL'])

            # now check if the best fit model is an ssp or csp 
            # only the csp models have tau and tauV parameters
            # so if you try to get these keywords for the ssp fits files
            # it will fail with a KeyError
            if 'TAU_GYR' in list(bc03_all_spec_hdulist[i + 1].header.keys()):
                model_tau = float(bc03_all_spec_hdulist[i + 1].header['TAU_GYR'])
                model_tauv = float(bc03_all_spec_hdulist[i + 1].header['TAUV'])
            else:
                # if the best fit model is an SSP then assign -99.0 to tau and tauV
                model_tau = -99.0
                model_tauv = -99.0

            # Convert to Av
            model_av = (model_tauv/1.086)
            if model_av < 0:
                model_av = -99.0

            model_metallicity_list.append(model_met)
            model_tau_list.append(model_tau)
            model_av_list.append(model_av)
            count_list.append(i+1)

            galaxy_count += 1

            if galaxy_count == 100:
                save_intermediate_results(d4000_in_list, d4000_out_list, d4000_out_err_list, mock_model_index_list, \
                    test_redshift_list, mock_zgrism_list, mock_zgrism_lowerr_list, mock_zgrism_higherr_list, \
                    model_age_list, model_metallicity_list, model_tau_list, model_av_list, count_list)

            if galaxy_count > 500:
                break

    # convert to numpy arrays and save
    d4000_in_list = np.asarray(d4000_in_list) 
    d4000_out_list = np.asarray(d4000_out_list)
    d4000_out_err_list = np.asarray(d4000_out_err_list) 
    mock_model_index_list = np.asarray(mock_model_index_list)
    test_redshift_list = np.asarray(test_redshift_list)
    mock_zgrism_list = np.asarray(mock_zgrism_list)
    mock_zgrism_lowerr_list = np.asarray(mock_zgrism_lowerr_list)
    mock_zgrism_higherr_list = np.asarray(mock_zgrism_higherr_list)

    model_age_list = np.asarray(model_age_list)
    model_metallicity_list = np.asarray(model_metallicity_list)
    model_tau_list = np.asarray(model_tau_list)
    model_av_list = np.asarray(model_av_list)
    count_list = np.asarray(count_list)

    # save
    d4000_range = '_1p2to1p4'
    np.save(massive_figures_dir + 'model_mockspectra_fits/d4000_in_list' + d4000_range + '.npy', d4000_in_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/d4000_out_list' + d4000_range + '.npy', d4000_out_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/d4000_out_err_list' + d4000_range + '.npy', d4000_out_err_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/mock_model_index_list' + d4000_range + '.npy', mock_model_index_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/test_redshift_list' + d4000_range + '.npy', test_redshift_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/mock_zgrism_list' + d4000_range + '.npy', mock_zgrism_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/mock_zgrism_lowerr_list' + d4000_range + '.npy', mock_zgrism_lowerr_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/mock_zgrism_higherr_list' + d4000_range + '.npy', mock_zgrism_higherr_list)

    np.save(massive_figures_dir + 'model_mockspectra_fits/model_age_list' + d4000_range + '.npy', model_age_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/model_metallicity_list' + d4000_range + '.npy', model_metallicity_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/model_tau_list' + d4000_range + '.npy', model_tau_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/model_av_list' + d4000_range + '.npy', model_av_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/count_list' + d4000_range + '.npy', count_list)

    # Total time taken
    print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."
    sys.exit(0)