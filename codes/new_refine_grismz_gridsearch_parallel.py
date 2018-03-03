from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.convolution import convolve_fft, convolve, Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo
from joblib import Parallel, delayed

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
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import grid_coadd as gd
import fast_chi2_jackknife_massive_galaxies as fcjm
import new_refine_grismz_iter as ni
import refine_redshifts_dn4000 as old_ref
import model_mods_cython_copytoedit as model_mods_cython
#import model_mods_cython

def get_line_mask(lam_grid, z):

    # ---------------- Mask potential emission lines ----------------- #
    # Will mask one point on each side of line center i.e. approx 80 A masked
    # These are all vacuum wavelengths
    oiii_5007 = 5008.24
    oiii_4959 = 4960.30
    hbeta = 4862.69
    hgamma = 4341.69
    oii_3727 = 3728.5
    # these two lines (3727 and 3729) are so close to each other 
    # that the line will always blend in grism spectra. 
    # avg wav of the two written here

    # Set up line mask
    line_mask = np.zeros(len(lam_grid), dtype=np.int)

    # Get redshifted wavelengths and mask
    oii_3727_idx = np.argmin(abs(lam_grid - oii_3727*(1 + z)))
    oiii_5007_idx = np.argmin(abs(lam_grid - oiii_5007*(1 + z)))
    oiii_4959_idx = np.argmin(abs(lam_grid - oiii_4959*(1 + z)))

    line_mask[oii_3727_idx-1 : oii_3727_idx+2] = 1
    line_mask[oiii_5007_idx-1 : oiii_5007_idx+2] = 1
    line_mask[oiii_4959_idx-1 : oiii_4959_idx+2] = 1

    return line_mask

def get_chi2(flam, ferr, object_lam_grid, model_comp_spec_mod, model_resampling_lam_grid):

    # chop the model to be consistent with the objects lam grid
    model_lam_grid_indx_low = np.argmin(abs(model_resampling_lam_grid - object_lam_grid[0]))
    model_lam_grid_indx_high = np.argmin(abs(model_resampling_lam_grid - object_lam_grid[-1]))
    model_spec_in_objlamgrid = model_comp_spec_mod[:, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # make sure that the arrays are the same length
    if int(model_spec_in_objlamgrid.shape[1]) != len(object_lam_grid):
        print "Arrays of unequal length. Must be fixed before moving forward. Exiting..."
        print "Model spectrum array shape:", model_spec_in_objlamgrid.shape
        print "Object spectrum length:", len(object_lam_grid)
        sys.exit(0)

    alpha_ = np.sum(flam * model_spec_in_objlamgrid / (ferr**2), axis=1) / np.sum(model_spec_in_objlamgrid**2 / ferr**2, axis=1)
    chi2_ = np.sum(((flam - (alpha_ * model_spec_in_objlamgrid.T).T) / ferr)**2, axis=1)

    return chi2_, alpha_

def get_chi2_alpha_at_z(z, flam_obs, ferr_obs, lam_obs, model_lam_grid, model_comp_spec, \
    resampling_lam_grid, total_models, lsf, start_time):

    print "\n", "Currently at redshift:", z

    # make sure the types are correct before passing to cython code
    #lam_obs = lam_obs.astype(np.float64)
    #model_lam_grid = model_lam_grid.astype(np.float64)
    #model_comp_spec = model_comp_spec.astype(np.float64)
    #resampling_lam_grid = resampling_lam_grid.astype(np.float64)
    total_models = int(total_models)
    #lsf = lsf.astype(np.float64)

    # first modify the models at the current redshift to be able to compare with data
    model_comp_spec_modified = \
    model_mods_cython.do_model_modifications(model_lam_grid, model_comp_spec, \
        resampling_lam_grid, total_models, lsf, z)
    print "Model mods done at current z:", z
    print "Total time taken up to now --", time.time() - start_time, "seconds."
    print model_comp_spec_modified.shape

    # Mask emission lines
    line_mask = get_line_mask(lam_obs, z)
    flam_obs = ma.array(flam_obs, mask=line_mask)
    ferr_obs = ma.array(ferr_obs, mask=line_mask)
    lam_obs = ma.array(lam_obs, mask=line_mask)

    # Now do the chi2 computation
    chi2_temp, alpha_temp = get_chi2(flam_obs, ferr_obs, lam_obs, model_comp_spec_modified, resampling_lam_grid)

    return chi2_temp, alpha_temp

def do_fitting(flam_obs, ferr_obs, lam_obs, lsf, starting_z, resampling_lam_grid, \
    model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start_time,\
    obj_id, obj_field, specz, photoz):

    """
    All models are redshifted to each of the redshifts in the list defined below,
    z_arr_to_check. Then the model modifications are done at that redshift.

    For each iteration through the redshift list it computes a chi2 for each model.
    So there are 
    """

    # Set up redshift grid to check
    z_arr_to_check = np.linspace(start=starting_z - 0.3, stop=starting_z + 0.3, num=301, dtype=np.float64)
    z_idx = np.where((z_arr_to_check >= 0.6) & (z_arr_to_check <= 1.235))
    z_arr_to_check = z_arr_to_check[z_idx]
    print "Will check the following redshifts:", z_arr_to_check
    if not z_arr_to_check.size:
        return -99.0

    ####### ------------------------------------ Main loop through redshfit array ------------------------------------ #######
    # Loop over all redshifts to check
    # set up chi2 and alpha arrays
    chi2 = np.empty((len(z_arr_to_check), total_models))
    alpha = np.empty((len(z_arr_to_check), total_models))

    # looping
    num_cores = 7
    chi2_alpha_list = Parallel(n_jobs=num_cores)(delayed(get_chi2_alpha_at_z)(z, \
    flam_obs, ferr_obs, lam_obs, model_lam_grid, model_comp_spec, resampling_lam_grid, total_models, lsf, start_time) \
    for z in z_arr_to_check)

    # the parallel code seems to like returning only a list
    # so I have to unpack the list
    for i in range(len(z_arr_to_check)):
        chi2[i], alpha[i] = chi2_alpha_list[i]

    # regular for loop 
    # use this if you dont want to use the parallel for loop above
    # comment it out if you don't 
    #count = 0
    #for z in z_arr_to_check:
    #    chi2[count], alpha[count] = get_chi2_alpha_at_z(z, flam_obs, ferr_obs, lam_obs, \
    #        model_lam_grid, model_comp_spec, resampling_lam_grid, total_models, lsf, start_time)
    #    count += 1

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

    print "Current best fit log(age [yr]):", "{:.4}".format(age)
    print "Current best fit Tau [Gyr]:", "{:.4}".format(tau)
    print "Current best fit Tau_V:", tauv

    # Simply the minimum chi2 might not be right
    # Should check if the minimum is global or local
    min_chi2 = chi2[min_idx_2d]
    low_chi2_idx = np.where(chi2 < min_chi2 + 0.1*min_chi2)
    print len(low_chi2_idx[0].ravel())
    #print low_chi2_idx

    ####### ------------------------------------------ Plotting ------------------------------------------ #######
    #### -------- Plot spectrum: Data, best fit model, and the residual --------- ####
    # get things needed to plot and plot
    bestalpha = alpha[min_idx_2d]
    print "Vertical scaling factor for best fit model:", bestalpha
    # chop model again to get the part within objects lam obs grid
    model_lam_grid_indx_low = np.argmin(abs(resampling_lam_grid - lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(abs(resampling_lam_grid - lam_obs[-1]))

    # make sure the types are correct before passing to cython code
    #lam_obs = lam_obs.astype(np.float64)
    #model_lam_grid = model_lam_grid.astype(np.float64)
    #model_comp_spec = model_comp_spec.astype(np.float64)
    #resampling_lam_grid = resampling_lam_grid.astype(np.float64)
    total_models = int(total_models)
    #lsf = lsf.astype(np.float64)

    # Will have to redo the model modifications at the new found z_grism
    # You have to do this to plot the correct best fit model with its 
    # modifications which was used for the fitting. 
    # Either it has to be done this way or you will have to keep the 
    # modified models in an array and then plot the best one here later.
    model_comp_spec_modified = \
    model_mods_cython.do_model_modifications(model_lam_grid, model_comp_spec, \
        resampling_lam_grid, total_models, lsf, z_grism)
    print "Model mods done (only for plotting purposes) at the new grism z:", z_grism
    print "Total time taken up to now --", time.time() - start_time, "seconds."

    best_fit_model_in_objlamgrid = model_comp_spec_modified[model_idx, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # again make sure that the arrays are the same length
    if int(best_fit_model_in_objlamgrid.shape[0]) != len(lam_obs):
        print "Arrays of unequal length. Must be fixed before moving forward. Exiting..."
        sys.exit(0)
    # plot
    plot_fit_and_residual_withinfo(lam_obs, flam_obs, ferr_obs, best_fit_model_in_objlamgrid, bestalpha,\
        obj_id, obj_field, specz, photoz, z_grism, (chi2[min_idx_2d]/len(lam_obs)), age, tau, (tauv/1.086))

    #### -------- Plot chi2 surface as 2D image --------- ####
    # This chi2 map can also be visualized as an image. 
    # Run imshow() and check what it looks like.
    #fig = plt.figure(figsize=(6,6))
    #ax = fig.add_subplot(111)

    #chi2[low_chi2_idx] = 0.0
    #ax.imshow(chi2)

    #ax.set_xscale('log')
    #ax.set_xlim(1,total_models)
    #plt.show()

    return z_grism

def plot_fit_and_residual_withinfo(lam_obs, flam_obs, ferr_obs, best_fit_model_in_objlamgrid, bestalpha,\
    obj_id, obj_field, specz, photoz, grismz, chi2, age, tau, av):

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
    ax1.tick_params('both', width=1, length=3, which='minor')
    ax1.tick_params('both', width=1, length=4.7, which='major')

    ax2.minorticks_on()
    ax2.tick_params('both', width=1, length=3, which='minor')
    ax2.tick_params('both', width=1, length=4.7, which='major')

    # text for info
    ax1.text(0.75, 0.4, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.35, r'$\mathrm{z_{grism}\, =\, }$' + "{:.4}".format(grismz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.3, r'$\mathrm{z_{spec}\, =\, }$' + "{:.4}".format(specz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.25, r'$\mathrm{z_{phot}\, =\, }$' + "{:.4}".format(photoz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.2, r'$\mathrm{\chi^2_{red}\, =\, }$' + "{:.3}".format(chi2), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.47, 0.3,'log(Age[yr]) = ' + "{:.4}".format(age), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.47, 0.25, r'$\tau$' + '[Gyr] = ' + "{:.3}".format(tau), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    if av < 0:
        av = -99.0

    ax1.text(0.47, 0.2, r'$\mathrm{A_V}$' + ' = ' + "{:.3}".format(av), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    fig.savefig(figs_dir + 'massive-galaxies-figures/new_specz_sample_fits/' + obj_field + '_' + str(obj_id) + '_fast1.png', \
        dpi=300, bbox_inches='tight')

    return None

if __name__ == '__main__':
    """
    The __main__ part of this code is similar to that in new_refine_grismz_iter.py
    """
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

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
    model_lam_grid = model_lam_grid.astype(np.float64)
    model_comp_spec = np.zeros((total_models, len(model_lam_grid)), dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec_hdulist[j+1].data

    # total run time up to now
    print "All models put in numpy array. Total time taken up to now --", time.time() - start, "seconds."

    # read in matched files to get photo-z
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    # Read in Specz comparison catalogs
    specz_goodsn = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-N.txt', dtype=None, names=True)
    specz_goodss = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-S.txt', dtype=None, names=True)

    all_speccats = [specz_goodsn, specz_goodss]

    # save lists for comparing after code is done
    id_list = []
    field_list = []
    zgrism_list = []
    zspec_list = []
    zphot_list = []

    # start looping
    for cat in all_speccats:

        for i in range(len(cat)):

            # --------------------------------------------- GET OBS DATA ------------------------------------------- #
            current_id = cat['pearsid'][i]
            current_field = cat['field'][i]

            if current_field == 'GOODS-N':
                match_cat = matched_cat_n
            elif current_field == 'GOODS-S':
                match_cat = matched_cat_s

            photo_z_idx = np.where(match_cat['pearsid'] == current_id)[0]
            if len(photo_z_idx) == 0:
                print "Skipping because no photo-z found for ID", current_id, "in", current_field
                continue
            redshift = float(match_cat['zphot'][photo_z_idx])
            current_specz = float(cat['specz'][i])

            # If you want to run it for a single galaxy then 
            # give the info here and put a sys.exit(0) after 
            # do_fitting()
            #current_id = 77715
            #current_field = 'GOODS-N'
            #redshift = 0.9805
            #current_specz = 0.851

            print "At ID", current_id, "in", current_field, "with specz and photo-z:", current_specz, redshift

            lam_em, flam_em, ferr_em, specname, pa_chosen, netsig_chosen = gd.fileprep(current_id, redshift, current_field)

            # now make sure that the quantities are all in observer frame
            # the original function (fileprep in grid_coadd) will give quantities in rest frame
            # but I need these to be in the observed frame so I will redshift them again.
            # It gives them in the rest frame because getting the d4000 and the average 
            # lambda separation needs to be figured out in the rest frame.
            flam_obs = flam_em / (1 + redshift)
            ferr_obs = ferr_em / (1 + redshift)
            lam_obs = lam_em * (1 + redshift)

            lam_obs = lam_obs.astype(np.float64)

            # plot to check # Line useful for debugging. Do not remove. Just uncomment.
            #ni.plotspectrum(lam_obs, flam_obs, ferr_obs)

            # --------------------------------------------- Quality checks ------------------------------------------- #
            # Netsig check
            if netsig_chosen < 100:
                print "Skipping", current_id, "in", current_field, "due to low NetSig:", netsig_chosen
                continue

            # Overall error check
            if np.sum(abs(ferr_obs)) > 0.2 * np.sum(abs(flam_obs)):
                print "Skipping", current_id, "in", current_field, "because of overall error."
                continue

            # ---------------------------------------------- FITTING ----------------------------------------------- #
            # Read in LSF
            if current_field == 'GOODS-N':
                lsf_filename = lsfdir + "north_lsfs/" + "n" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"
            elif current_field == 'GOODS-S':
                lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"

            # read in LSF file
            try:
                lsf = np.loadtxt(lsf_filename)
                lsf = lsf.astype(np.float64)
            except IOError:
                print "LSF not found. Moving to next galaxy."
                continue

            # extend lam_grid to be able to move the lam_grid later 
            avg_dlam = old_ref.get_avg_dlam(lam_obs)

            lam_low_to_insert = np.arange(5000, lam_obs[0], avg_dlam, dtype=np.float64)
            lam_high_to_append = np.arange(lam_obs[-1] + avg_dlam, 10500, avg_dlam, dtype=np.float64)

            resampling_lam_grid = np.insert(lam_obs, obj=0, values=lam_low_to_insert)
            resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

            # call actual fitting function
            zg = do_fitting(flam_obs, ferr_obs, lam_obs, lsf, redshift, resampling_lam_grid, \
                model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start,\
                current_id, current_field, current_specz, redshift)

            id_list.append(current_id)
            field_list.append(current_field)
            zgrism_list.append(zg)
            zspec_list.append(redshift)
            zphot_list.append(current_specz)

            #sys.exit(0)

    print id_list
    print field_list
    print zgrism_list
    print zspec_list
    print zphot_list

    zgrism_list = np.asarray(zgrism_list)
    zspec_list = np.asarray(zspec_list)
    zphot_list = np.asarray(zphot_list)

    print id_list
    print field_list
    print zgrism_list
    print zspec_list
    print zphot_list

    sys.exit(0)
