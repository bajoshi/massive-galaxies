from __future__ import division

import numpy as np
import numpy.ma as ma
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo
from joblib import Parallel, delayed

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

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
#import fast_chi2_jackknife_massive_galaxies as fcjm
#import new_refine_grismz_iter as ni
import refine_redshifts_dn4000 as old_ref
import model_mods_cython_copytoedit as model_mods_cython
import dn4000_catalog as dc

def get_line_mask(lam_grid, z):

    # ---------------- Mask potential emission lines ----------------- #
    # Will mask 3 points on each side of line center i.e. approx 240 A masked
    # In case of [OII]3727 maksing 5 points on blue side and 3 points on red side i.e. 320 A
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

    line_mask[oii_3727_idx-6 : oii_3727_idx+5] = 1
    line_mask[oiii_5007_idx-3 : oiii_5007_idx+4] = 1
    line_mask[oiii_4959_idx-3 : oiii_4959_idx+4] = 1

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

    # Mask emission lines
    line_mask = get_line_mask(lam_obs, z)
    flam_obs = ma.array(flam_obs, mask=line_mask)
    ferr_obs = ma.array(ferr_obs, mask=line_mask)

    # Now do the chi2 computation
    chi2_temp, alpha_temp = get_chi2(flam_obs, ferr_obs, lam_obs, model_comp_spec_modified, resampling_lam_grid)

    return chi2_temp, alpha_temp

def do_fitting(flam_obs, ferr_obs, lam_obs, lsf, starting_z, resampling_lam_grid, \
    model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start_time,\
    obj_id, obj_field, specz, photoz, netsig, d4000, search_range):

    """
    All models are redshifted to each of the redshifts in the list defined below,
    z_arr_to_check. Then the model modifications are done at that redshift.

    For each iteration through the redshift list it computes a chi2 for each model.
    """

    # Set up redshift grid to check
    z_arr_to_check = np.linspace(start=starting_z - search_range, stop=starting_z + search_range, num=81, dtype=np.float64)
    z_idx = np.where((z_arr_to_check >= 0.6) & (z_arr_to_check <= 1.235))
    z_arr_to_check = z_arr_to_check[z_idx]
    print "Will check the following redshifts:", z_arr_to_check
    if not z_arr_to_check.size:
        return -99.0, -99.0, -99.0, -99.0, -99.0

    ####### ------------------------------------ Main loop through redshfit array ------------------------------------ #######
    # Loop over all redshifts to check
    # set up chi2 and alpha arrays
    chi2 = np.empty((len(z_arr_to_check), total_models))
    alpha = np.empty((len(z_arr_to_check), total_models))

    # looping
    num_cores = 10
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
    ############# -------------------------- Errors on z and other derived params ----------------------------- #############
    min_chi2 = chi2[min_idx_2d]
    # See Andrae+ 2010;arXiv:1012.3754. The number of d.o.f. for non-linear models 
    # is not well defined and reduced chi2 should really not be used.
    # Seth's comment: My model is actually linear. Its just a factor 
    # times a set of fixed points. And this is linear, because each
    # model is simply a function of lambda, which is fixed for a given 
    # model. So every model only has one single free parameter which is
    # alpha i.e. the vertical scaling factor; that's true since alpha is 
    # the only one I'm actually solving for to get a min chi2. I'm not 
    # varying the other parameters - age, tau, av, metallicity, or 
    # z_grism - within a given model. Therefore, I can safely use the 
    # methods described in Andrae+ 2010 for linear models.
    dof = len(lam_obs) - 1  # i.e. total data points minus the single fitting parameter
    chi2_red = chi2 / dof
    chi2_red_error = np.sqrt(2/dof)
    min_chi2_red = min_chi2 / dof
    print "Error in reduced chi-square:", chi2_red_error
    chi2_red_2didx = np.where((chi2_red >= min_chi2_red - chi2_red_error) & (chi2_red <= min_chi2_red + chi2_red_error))
    print "Indices within 1-sigma of reduced chi-square:", chi2_red_2didx
    # use first dimension indices to get error on grism-z
    z_grism_range = z_arr_to_check[chi2_red_2didx[0]]
    print "z_grism range", z_grism_range
    low_z_lim = np.min(z_grism_range)
    upper_z_lim = np.max(z_grism_range)
    print "Min z_grism within 1-sigma error:", low_z_lim
    print "Max z_grism within 1-sigma error:", upper_z_lim

    # These low chi2 indices are useful as a first attempt to figure
    # out the spread in chi2 but otherwise not too enlightening.
    # I'm keeping these lines in here for now.
    #low_chi2_idx = np.where((chi2 < min_chi2 + 0.5*min_chi2) & (chi2 > min_chi2 - 0.5*min_chi2))[0]
    #print len(low_chi2_idx.ravel())
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
        obj_id, obj_field, specz, photoz, z_grism, low_z_lim, upper_z_lim, min_chi2_red, age, tau, (tauv/1.086), netsig, d4000)

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

    return z_grism, low_z_lim, upper_z_lim, min_chi2_red, age, tau, (tauv/1.086)

def plot_fit_and_residual_withinfo(lam_obs, flam_obs, ferr_obs, best_fit_model_in_objlamgrid, bestalpha,\
    obj_id, obj_field, specz, photoz, grismz, low_z_lim, upper_z_lim, chi2, age, tau, av, netsig, d4000):

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

    low_zerr = grismz - low_z_lim
    high_zerr = upper_z_lim - grismz

    ax1.text(0.75, 0.35, \
    r'$\mathrm{z_{grism}\, =\, }$' + "{:.4}".format(grismz) + \
    r'$\substack{+$' + "{:.3}".format(low_zerr) + r'$\\ -$' + "{:.3}".format(high_zerr) + r'$}$', \
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

    ax1.text(0.75, 0.12, r'$\mathrm{\NetSig\, =\, }$' + "{:.3}".format(netsig), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.07, r'$\mathrm{\D4000(from\ z_{phot})\, =\, }$' + "{:.3}".format(d4000), \
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

    fig.savefig(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/' + obj_field + '_' + str(obj_id) + '_broadlsf_linemask.png', \
        dpi=300, bbox_inches='tight')

    return None

def get_data(pears_index, field, check_contam=True):
    """
    Using code from fileprep in this function; not including 
    everything because it does a few things that I don't 
    actually need; also there is no contamination checking
    """

    # read in spectrum file
    data_path = home + "/Documents/PEARS/data_spectra_only/"
    # Get the correct filename and the number of extensions
    if field == 'GOODS-N':
        filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    elif field == 'GOODS-S':
        filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

    fitsfile = fits.open(filename)
    n_ext = fitsfile[0].header['NEXTEND']

    # Loop over all extensions and get the best PA
    # Get highest netsig to find the spectrum to be added
    if n_ext > 1:
        netsiglist = []
        palist = []
        for count in range(n_ext):
            #print "At PA", fitsfile[count+1].header['POSANG']  # Line useful for debugging. Do not remove. Just uncomment.
            fitsdata = fitsfile[count+1].data
            netsig = gd.get_net_sig(fitsdata)
            netsiglist.append(netsig)
            palist.append(fitsfile[count+1].header['POSANG'])
            #print "At PA", fitsfile[count+1].header['POSANG'], "with NetSig", netsig  
            # Above line also useful for debugging. Do not remove. Just uncomment.
        netsiglist = np.array(netsiglist)
        maxnetsigarg = np.argmax(netsiglist)
        netsig_chosen = np.max(netsiglist)
        spec_toadd = fitsfile[maxnetsigarg+1].data
        pa_chosen = fitsfile[maxnetsigarg+1].header['POSANG']
    elif n_ext == 1:
        spec_toadd = fitsfile[1].data
        pa_chosen = fitsfile[1].header['POSANG']
        netsig_chosen = gd.get_net_sig(fitsfile[1].data)
        
    # Now get the spectrum to be added
    lam_obs = spec_toadd['LAMBDA']
    flam_obs = spec_toadd['FLUX']
    ferr_obs = spec_toadd['FERROR']
    contam = spec_toadd['CONTAM']

    """
    In the next few lines within this function, I'm using a flag called
    return_code. This flag is used to tell the next part of the code,
    which is using the output from this function, if this function thinks 
    it returned anything useful. 1 = Useful. 0 = Not useful.
    """
 
    # Check that contamination level is not too high
    if check_contam:
        if np.nansum(contam) > 0.33 * np.nansum(flam_obs):
            print pears_index, " in ", field, " has an too high a level of contamination.",
            print "Contam =", np.nansum(contam) / np.nansum(flam_obs), " * F_lam. This galaxy will be skipped."
            return_code = 0
            fitsfile.close()
            return lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code

    # Check that input wavelength array is not empty
    if not lam_obs.size:
        print pears_index, " in ", field, " has an empty wav array. Returning empty array..."
        return_code = 0
        fitsfile.close()
        return lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code

    # Now chop off the ends and only look at the observed spectrum from 6000A to 9500A
    arg6000 = np.argmin(abs(lam_obs - 6000))
    arg9500 = np.argmin(abs(lam_obs - 9500))
        
    lam_obs = lam_obs[arg6000:arg9500]
    flam_obs = flam_obs[arg6000:arg9500]
    ferr_obs = ferr_obs[arg6000:arg9500]
    contam = contam[arg6000:arg9500]

    # subtract contamination if all okay
    flam_obs -= contam
    
    return_code = 1
    fitsfile.close()
    return lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code

if __name__ == '__main__':

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

    # ----------------------------------------- READ IN CATALOGS ----------------------------------------- #
    # read in matched files to get photo-z
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    # Read in Specz comparison catalogs
    specz_goodsn = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-N.txt', dtype=None, names=True)
    specz_goodss = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-S.txt', dtype=None, names=True)

    # large differences between specz and grismz
    large_diff_cat = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/large_diff_specz_short.txt', dtype=None, names=True)

    all_speccats = [specz_goodsn]  #[specz_goodsn, specz_goodss]
    all_match_cats = [large_diff_cat]  #[matched_cat_n, matched_cat_s]

    # save lists for comparing after code is done
    id_list = []
    field_list = []
    zgrism_list = []
    zgrism_lowerr_list = []
    zgrism_uperr_list = []
    zspec_list = []
    zphot_list = []
    chi2_list = []
    netsig_list = []
    age_list = []
    tau_list = []
    av_list = []
    d4000_list = []
    d4000_err_list = []

    # start looping
    catcount = 0
    for cat in all_match_cats:

        for i in range(len(cat)):

            # --------------------------------------------- GET OBS DATA ------------------------------------------- #
            current_id = cat['pearsid'][i]

            if catcount == 0:
                current_field = 'GOODS-N'
                spec_cat = specz_goodsn
            elif catcount == 1: 
                current_field = 'GOODS-S'
                spec_cat = specz_goodss

            # Get specz if it exists as initial guess, otherwise get photoz
            specz_idx = np.where(spec_cat['pearsid'] == current_id)[0]

            if len(specz_idx) == 1:
                current_specz = float(spec_cat['specz'][specz_idx])
                redshift = float(cat['zphot'][i])
                starting_z = current_specz
            elif len(specz_idx) == 0:
                current_specz = -99.0
                redshift = float(cat['zphot'][i])
                starting_z = redshift
            else:
                print "Got other than 1 or 0 matches for the specz for ID", current_id, "in", current_field
                print "This much be fixed. Check why it happens. Exiting now."
                sys.exit(0)

            # Check that the starting redshfit is within the required range
            if (starting_z < 0.6) or (starting_z > 1.235):
                print "Current galaxy", current_id, current_field, "at starting_z", starting_z, "not within redshift range.",
                print "Moving to the next galaxy."
                continue

            # If you want to run it for a single galaxy then 
            # give the info here and put a sys.exit(0) after 
            # do_fitting()
            #current_id = 36105
            #current_field = 'GOODS-N'
            #redshift = 0.9072
            #current_specz = 0.931
            #starting_z = current_specz

            print "At ID", current_id, "in", current_field, "with specz and photo-z:", current_specz, redshift

            lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = get_data(current_id, current_field)

            if return_code == 0:
                print "Skipping due to an error with the obs data. See the error message just above this one.",
                print "Moving to the next galaxy."
                continue

            # Force dtype for cython code
            # Apparently this (i.e. for flam_obs and ferr_obs) has  
            # to be done to avoid an obscure error from parallel in joblib --
            # AttributeError: 'numpy.ndarray' object has no attribute 'offset'
            lam_obs = lam_obs.astype(np.float64)
            flam_obs = flam_obs.astype(np.float64)
            ferr_obs = ferr_obs.astype(np.float64)

            # plot to check # Line useful for debugging. Do not remove. Just uncomment.
            #ni.plotspectrum(lam_obs, flam_obs, ferr_obs)

            # --------------------------------------------- Quality checks ------------------------------------------- #
            # Netsig check
            if netsig_chosen < 10:
                print "Skipping", current_id, "in", current_field, "due to low NetSig:", netsig_chosen
                continue

            # D4000 check # accept only if D4000 greater than 1.2
            # get d4000
            # You have to de-redshift it to get D4000. So if the original z is off then the D4000 will also be off.
            # This is way I'm letting some lower D4000 values into my sample. Just so I don't miss too many galaxies.
            # A few of the galaxies with really wrong starting_z will of course be missed.
            lam_em = lam_obs / (1 + starting_z)
            flam_em = flam_obs * (1 + starting_z)
            ferr_em = ferr_obs * (1 + starting_z)

            # Check that hte lambda array is not too incomplete 
            # I don't want the D4000 code extrapolating too much.
            # I'm choosing this limit to be 50A
            if np.max(lam_em) < 4200:
                print "Skipping because lambda array is incomplete by too much."
                print "i.e. the max val in rest-frame lambda is less than 4200A."
                continue

            d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)
            if d4000 < 1.1:
                print "Skipping", current_id, "in", current_field, "due to low D4000:", d4000
                continue

            # Overall error check. Suppressed for now.
            """
            if np.sum(abs(ferr_obs)) > 0.2 * np.sum(abs(flam_obs)):
                print "Skipping", current_id, "in", current_field, "because of overall error."
                continue
            """

            # ---------------------------------------------- FITTING ----------------------------------------------- #
            # Read in LSF
            if current_field == 'GOODS-N':
                lsf_filename = lsfdir + "north_lsfs/" + "n" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"
            elif current_field == 'GOODS-S':
                lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"

            # read in LSF file
            try:
                lsf = np.genfromtxt(lsf_filename)
                lsf = lsf.astype(np.float64)  # Force dtype for cython code
            except IOError:
                print "LSF not found. Moving to next galaxy."
                continue

            # -------- Broaden the LSF ------- #
            # SEE THE FILE -- /Users/baj/Desktop/test-codes/cython_test/cython_profiling/profile.py
            # FOR DETAILS ON BROADENING LSF METHOD USED BELOW.
            # fit
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

            # ------------- Call actual fitting function ------------- #
            zg, zerr_low, zerr_up, min_chi2, age, tau, av = \
            do_fitting(flam_obs, ferr_obs, lam_obs, broad_lsf, starting_z, resampling_lam_grid, \
                model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start,\
                current_id, current_field, current_specz, redshift, netsig_chosen, d4000, 0.2)

            # Get d4000 at new zgrism
            lam_em = lam_obs / (1 + zg)
            flam_em = flam_obs * (1 + zg)
            ferr_em = ferr_obs * (1 + zg)

            d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

            # ---------------------------------------------- SAVE PARAMETERS ----------------------------------------------- #
            id_list.append(current_id)
            field_list.append(current_field)
            zgrism_list.append(zg)
            zgrism_lowerr_list.append(zerr_low)
            zgrism_uperr_list.append(zerr_up)
            zspec_list.append(current_specz)
            zphot_list.append(redshift)
            chi2_list.append(min_chi2)
            netsig_list.append(netsig_chosen)
            age_list.append(age)
            tau_list.append(tau)
            av_list.append(av)
            d4000_list.append(d4000)
            d4000_err_list.append(d4000_err)

        catcount += 1

    id_list = np.asarray(id_list)
    field_list = np.asarray(field_list)
    zgrism_list = np.asarray(zgrism_list)
    zgrism_lowerr_list = np.asarray(zgrism_lowerr_list)
    zgrism_uperr_list = np.asarray(zgrism_uperr_list)
    zspec_list = np.asarray(zspec_list)
    zphot_list = np.asarray(zphot_list)
    chi2_list = np.asarray(chi2_list)
    netsig_list = np.asarray(netsig_list)
    age_list = np.asarray(age_list)
    tau_list = np.asarray(tau_list)
    av_list = np.asarray(av_list)
    d4000_list = np.asarray(d4000_list)
    d4000_err_list = np.asarray(d4000_err_list)

    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/id_list.npy', id_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/field_list.npy', field_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/zgrism_list.npy', zgrism_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/zgrism_lowerr_list.npy', zgrism_lowerr_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/zgrism_uperr_list.npy', zgrism_uperr_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/zspec_list.npy', zspec_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/zphot_list.npy', zphot_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/chi2_list.npy', chi2_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/netsig_list.npy', netsig_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/age_list.npy', age_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/tau_list.npy', tau_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/av_list.npy', av_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/d4000_list.npy', d4000_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/d4000_err_list.npy', d4000_err_list)

    # Total time taken
    print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."
    sys.exit(0)
