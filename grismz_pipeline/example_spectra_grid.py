from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(massive_galaxies_dir + 'grism_pipeline/')
sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import model_mods_cython_copytoedit as model_mods_cython
import new_refine_grismz_gridsearch_parallel as ngp
import dn4000_catalog as dc
import refine_redshifts_dn4000 as old_ref

def get_best_model_and_plot(flam_obs, ferr_obs, lam_obs, lsf, resampling_lam_grid, \
    model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start_time,\
    obj_id, obj_field, grismz, specz, photoz, d4000, d4000_err, low_zerr, high_zerr):

    chi2, alpha = ngp.get_chi2_alpha_at_z(grismz, flam_obs, ferr_obs, lam_obs, model_lam_grid, model_comp_spec, \
        resampling_lam_grid, total_models, lsf, start_time)

    ####### -------------------------------------- Min chi2 and best fit params -------------------------------------- #######
    # Sort through the chi2 and make sure that the age is physically meaningful
    sortargs = np.argsort(chi2, axis=None)  # i.e. it will use the flattened array to sort

    for k in range(len(chi2)):

        # Find the minimum chi2
        min_idx = sortargs[k]
        
        # Get the best fit model parameters
        # first get the index for the best fit
        model_idx = int(min_idx)

        age = float(bc03_all_spec_hdulist[model_idx + 1].header['LOG_AGE'])
        age_at_z = cosmo.age(grismz).value * 1e9  # in yr

        # now check if the age is meaningful
        if (age < np.log10(age_at_z - 1e8)) and (age > 9 + np.log10(0.1)):
            # If the age is meaningful then you don't need to do anything
            # more. Just break out of the loop. the best fit parameters have
            # already been assigned to variables. This assignment is done before 
            # the if statement to make sure that there are best fit parameters 
            # even if the loop is broken out of in the first iteration.
            break

    min_chi2 = chi2[min_idx]
    print "Minimum chi2:", "{:.4}".format(min_chi2)

    # Plotting stuff #
    bestalpha = alpha[min_idx]
    print "Vertical scaling factor for best fit model:", bestalpha
    # chop model again to get the part within objects lam obs grid
    model_lam_grid_indx_low = np.argmin(abs(resampling_lam_grid - lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(abs(resampling_lam_grid - lam_obs[-1]))

    model_comp_spec_modified = \
    model_mods_cython.do_model_modifications(model_lam_grid, model_comp_spec, \
        resampling_lam_grid, total_models, lsf, grismz)

    best_fit_model_in_objlamgrid = model_comp_spec_modified[model_idx, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    makeplot(lam_obs, flam_obs, ferr_obs, best_fit_model_in_objlamgrid, bestalpha,\
        obj_id, obj_field, specz, photoz, grismz, low_zerr, high_zerr, chi2, d4000, d4000_err)

    return None

def makeplot(lam_obs, flam_obs, ferr_obs, best_fit_model_in_objlamgrid, bestalpha,\
    obj_id, obj_field, specz, photoz, grismz, low_zerr, high_zerr, chi2, d4000, d4000_err):

    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[:8,:])
    ax2 = fig.add_subplot(gs[8:,:])

    ax1.set_ylabel(r'$\mathrm{f_\lambda\ [erg\,s^{-1}\,cm^{-2}\,\AA]}$')
    ax2.set_xlabel(r'$\mathrm{Wavelength\, [\AA]}$')
    ax2.set_ylabel(r'$\mathrm{\frac{f^{obs}_\lambda\ - f^{model}_\lambda}{f^{obs;error}_\lambda}}$')

    ax1.plot(lam_obs, flam_obs, ls='-', color='k')
    ax1.plot(lam_obs, bestalpha*best_fit_model_in_objlamgrid, ls='-', color='r')
    ax1.fill_between(lam_obs, flam_obs + ferr_obs, flam_obs - ferr_obs, color='lightgray')

    resid_fit = (flam_obs - bestalpha*best_fit_model_in_objlamgrid) / ferr_obs
    ax2.plot(lam_obs, resid_fit, ls='-', color='k')

    # minor ticks
    ax1.minorticks_on()
    ax2.minorticks_on()

    # text for info
    ax1.text(0.75, 0.4, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.75, 0.35, \
    r'$\mathrm{z_{grism}\, =\, }$' + "{:.4}".format(grismz) + r'$\substack{+$' + "{:.3}".format(low_zerr) + r'$\\ -$' + "{:.3}".format(high_zerr) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.27, r'$\mathrm{z_{spec}\, =\, }$' + "{:.4}".format(specz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.22, r'$\mathrm{z_{phot}\, =\, }$' + "{:.4}".format(photoz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.75, 0.17, r'$\mathrm{\chi^2\, =\, }$' + "{:.3}".format(chi2), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.75, 0.12, r'$\mathrm{D4000\, =\, }$' + "{:.2}".format(d4000) + r'$\pm$' + "{:.2}".format(d4000_err), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    fig.savefig(figs_dir + 'massive-galaxies-figures/' + obj_field + '_' + str(obj_id) + '.png', \
        dpi=300, bbox_inches='tight')

    return None

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # id list that I want. I chose these by eye.
    ids_to_plot = [36105, 40991, 46766, 53471, 57792, 58084, 61291, \
    76098, 88566, 88929, 119879, 124386, \
    74060, 88671, 103146, 104774, 113298, 127226]
    fields_to_plot = ['n', 'n', 'n', 'n', 'n', 'n', 'n', \
    'n', 'n', 'n', 'n', 'n', \
    's', 's', 's', 's', 's', 's']

    # read all arrays from full run
    id_n = np.load(massive_figures_dir + 'full_run/id_list_gn.npy')
    id_s = np.load(massive_figures_dir + 'full_run/id_list_gs.npy')

    zgrism_n = np.load(massive_figures_dir + 'full_run/zgrism_list_gn.npy')
    zgrism_s = np.load(massive_figures_dir + 'full_run/zgrism_list_gs.npy')

    zspec_n = np.load(massive_figures_dir + 'full_run/zspec_list_gn.npy')
    zspec_s = np.load(massive_figures_dir + 'full_run/zspec_list_gs.npy')

    zphot_n = np.load(massive_figures_dir + 'full_run/zphot_list_gn.npy')
    zphot_s = np.load(massive_figures_dir + 'full_run/zphot_list_gs.npy')

    zgrism_lowerr_n = np.load(massive_figures_dir + 'full_run/zgrism_lowerr_list_gn.npy')
    zgrism_lowerr_s = np.load(massive_figures_dir + 'full_run/zgrism_lowerr_list_gs.npy')

    zgrism_uperr_n = np.load(massive_figures_dir + 'full_run/zgrism_uperr_list_gn.npy')
    zgrism_uperr_s = np.load(massive_figures_dir + 'full_run/zgrism_uperr_list_gs.npy')

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

    # ---------------------------------------------- FITTING ----------------------------------------------- #
    # Loop over all the above ids and plot them on a grid
    for i in range(len(ids_to_plot)):

        current_id = ids_to_plot[i]
        current_field = fields_to_plot[i]

        if current_field == 'n':
            fieldname = 'GOODS-N'
            id_arr = id_n
            zgrism_arr = zgrism_n
            zgrism_lowerr_arr = zgrism_lowerr_n
            zgrism_uperr_arr = zgrism_uperr_n
            zphot_arr = zphot_n
            zspec_arr = zspec_n
        elif current_field == 's':
            fieldname = 'GOODS-S'
            id_arr = id_s
            zgrism_arr = zgrism_s
            zgrism_lowerr_arr = zgrism_lowerr_s
            zgrism_uperr_arr = zgrism_uperr_s
            zphot_arr = zphot_s
            zspec_arr = zspec_s

        id_idx = np.where(id_arr == current_id)[0]
        current_zgrism = zgrism_arr[id_idx]
        current_zgrism_lowerr = current_zgrism - zgrism_lowerr_arr[id_idx]
        current_zgrism_uperr = zgrism_uperr_arr[id_idx] - current_zgrism
        current_zspec = zspec_arr[id_idx]
        current_zphot = zphot_arr[id_idx]

        # get data and then d4000
        lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, fieldname)
        print current_id, fieldname, current_zgrism, current_zgrism_lowerr, current_zgrism_uperr, return_code

        lam_em = lam_obs / (1 + current_zgrism)
        flam_em = flam_obs * (1 + current_zgrism)
        ferr_em = ferr_obs * (1 + current_zgrism)

        d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

        # Read in LSF
        if fieldname == 'GOODS-N':
            lsf_filename = lsfdir + "north_lsfs/" + "n" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"
        elif fieldname == 'GOODS-S':
            lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"

        # read in LSF file
        lsf = np.genfromtxt(lsf_filename)
        lsf = lsf.astype(np.float64)  # Force dtype for cython code

        # -------- Broaden the LSF ------- #
        # SEE THE FILE -- /Users/baj/Desktop/test-codes/cython_test/cython_profiling/profile.py
        # FOR DETAILS ON BROADENING LSF METHOD USED BELOW.
        lsf_length = len(lsf)
        gauss_init = models.Gaussian1D(amplitude=np.max(lsf), mean=lsf_length/2, stddev=lsf_length/4)
        fit_gauss = fitting.LevMarLSQFitter()
        x_arr = np.arange(lsf_length)
        g = fit_gauss(gauss_init, x_arr, lsf)
        # get fit std.dev. and create a gaussian kernel with which to broaden
        kernel_std = 1.118 * g.parameters[2]
        broaden_kernel = Gaussian1DKernel(kernel_std)
        # broaden LSF
        broad_lsf = fftconvolve(lsf, broaden_kernel, mode='same')
        broad_lsf = broad_lsf.astype(np.float64)  # Force dtype for cython code

        # ------- Make new resampling grid ------- # 
        # extend lam_grid to be able to move the lam_grid later 
        avg_dlam = old_ref.get_avg_dlam(lam_obs)

        lam_low_to_insert = np.arange(5000, lam_obs[0], avg_dlam, dtype=np.float64)
        lam_high_to_append = np.arange(lam_obs[-1] + avg_dlam, 10500, avg_dlam, dtype=np.float64)

        resampling_lam_grid = np.insert(lam_obs, obj=0, values=lam_low_to_insert)
        resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

        get_best_model_and_plot(flam_obs, ferr_obs, lam_obs, broad_lsf, resampling_lam_grid, \
        model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start,\
        current_id, fieldname, current_zgrism, current_zspec, current_zphot, d4000, d4000_err, \
        current_zgrism_lowerr, current_zgrism_uperr)

    sys.exit(0)