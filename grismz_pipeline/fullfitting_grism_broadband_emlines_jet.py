from __future__ import division

import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo
from joblib import Parallel, delayed
from scipy.interpolate import griddata, interp1d
from scipy.signal import fftconvolve
import pysynphot
from scipy.integrate import simps

import os
import sys
import glob
import time
import datetime

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"
threedhst_datadir = home + "/Desktop/3dhst_data/"
massive_figures_dir = figs_dir + 'massive-galaxies-figures/'

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import refine_redshifts_dn4000 as old_ref
import model_mods as mm
import dn4000_catalog as dc
import new_refine_grismz_gridsearch_parallel as ngp
import mocksim_results as mr

def check_spec_plot(grism_lam_obs, grism_flam_obs, grism_ferr_obs, phot_lam, phot_fluxes_arr, phot_errors_arr):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(grism_lam_obs, grism_flam_obs, 'o-', color='k', markersize=2)
    ax.fill_between(grism_lam_obs, grism_flam_obs + grism_ferr_obs, grism_flam_obs - grism_ferr_obs, color='lightgray')

    ax.errorbar(phot_lam, phot_fluxes_arr, yerr=phot_errors_arr, \
        fmt='.', color='firebrick', markeredgecolor='firebrick', \
        capsize=2, markersize=10.0, elinewidth=2.0)

    plt.show()

    return None

def get_filt_zp(filtname):

    filtname_arr = np.array(['F435W', 'F606W', 'F775W', 'F850LP', 'F125W', 'F140W', 'F160W'])

    # Corresponding lists containing AB and ST zeropoints
    # The first 4 filters i.e. 'F435W', 'F606W', 'F775W', 'F850LP'
    # are ACS/WFC filters and the correct ZP calculation is yet to be done for these.
    # ACS zeropoints are time-dependent and therefore the zeropoint calculator has to be used.
    # Check: http://www.stsci.edu/hst/acs/analysis/zeropoints
    # For WFC3/IR: http://www.stsci.edu/hst/wfc3/analysis/ir_phot_zpt
    # This page: http://www.stsci.edu/hst/acs/analysis/zeropoints/old_page/localZeropoints
    # gives the old ACS zeropoints.
    # I'm going to use the old zeropoints for now.
    # While these are outdated, I have no way of knowing the dates of every single GOODS observation 
    # (actually I could figure out the dates from the archive) and then using the zeropoint associated 
    # with that date and somehow combining all the photometric data to get some average zeropoint?

    # ------ After talking to Seth about this: Since we only really care about the difference
    # between the STmag and ABmag zeropoints it won't matter very much that we are using the 
    # older zeropoints. The STmag and the ABmag zeropoints should chagne by the same amount. 
    # This should therefore be okay.
    filt_zp_st_list = [25.16823, 26.67444, 26.41699, 25.95456, 28.0203, 28.479, 28.1875]
    filt_zp_ab_list = [25.68392, 26.50512, 25.67849, 24.86663, 26.2303, 26.4524, 25.9463]

    filt_idx = np.where(filtname_arr == filtname)[0][0]

    filt_zp_st = filt_zp_st_list[filt_idx]
    filt_zp_ab = filt_zp_ab_list[filt_idx]

    return filt_zp_st, filt_zp_ab

def get_flam(filtname, cat_flux):
    """ Convert everything to f_lambda units """

    filt_zp_st, filt_zp_ab = get_filt_zp(filtname)

    cat_flux = float(cat_flux)  # because it should be a single float
    abmag = 25.0 - 2.5*np.log10(cat_flux)  # 3DHST fluxes are normalized to an ABmag ZP of 25.0
    stmag = filt_zp_st - filt_zp_ab + abmag

    flam = 10**(-1 * (stmag + 21.1) / 2.5)

    return flam

def show_example_for_adding_emission_lines():
    """
    Call this function from within __main__ if you want to show
    the final result of adding emission lines to the models. 
    """

    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_lam_grid = model_lam_grid.astype(np.float64)

    # Get the number of Lyman continuum photons being produced
    model_idx = 1650
    """
    # Models 1600 to 1650ish are older than ~1Gyr # Models 1500 to 1600 are younger.
    # for e.g. 
    try 1600: age 143 Myr  # really strong lines
    1615: age 806 Myr # relatively weaker lines
    1630: age 2.4 Gyr # relatively stronger lines
    # You can use these indices to show how the lines change RELATIVE to the continuum.
    # i.e. The lines are quite strong relative to the continuum at young ages as expected.
    # What I hadn't expected was that the lines aren't too strong relative to the continuum
    # for Balmer break spectra. However, the lines again start to become stronger relative 
    # to the continuum as the Balmer break turns into a 4000A break.

    # Try making a plot of [OII]3727 EW vs Model Age to illustrate this point.
    # EW [A] = Total_line_flux / Continuum_flux_density_around_line
    # Identify which ages show a Balmer break and which ages show a 4000A break
    # in this plot.
    """

    nlyc = float(bc03_all_spec_hdulist[model_idx].header['NLyc'])

    # print model info
    model_logage = float(bc03_all_spec_hdulist[model_idx].header['LOG_AGE'])
    print "Model age[yr]:", "{:.3}".format(10**model_logage)
    print "Model SFH constant (tau [Gyr]):", float(bc03_all_spec_hdulist[model_idx].header['TAU_GYR'])
    print "Model A_V:", "{:.3}".format(float(bc03_all_spec_hdulist[model_idx].header['TAUV']) / 1.086)

    bc03_spec_lam_withlines, bc03_spec_withlines  = \
    emission_lines(0.02, model_lam_grid, bc03_all_spec_hdulist[model_idx].data, nlyc, silent=False)

    sys.exit(0)

    return None

def make_oii_ew_vs_age_plot():

    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_lam_grid = model_lam_grid.astype(np.float64)

    total_emission_lines_to_add = 12  # Make sure that this changes if you decide to add more lines to the models
    model_comp_spec_withlines = np.zeros((total_models, len(model_lam_grid) + total_emission_lines_to_add), dtype=np.float64)

    oii_ew_list = []
    model_logage_list = []

    for j in range(total_models):
        nlyc = float(bc03_all_spec_hdulist[j+1].header['NLYC'])
        metallicity = float(bc03_all_spec_hdulist[j+1].header['METAL'])
        model_logage_list.append(float(bc03_all_spec_hdulist[j+1].header['LOG_AGE']))

        model_lam_grid_withlines, model_comp_spec_withlines[j], oii_ew = \
        emission_lines(metallicity, model_lam_grid, bc03_all_spec_hdulist[j+1].data, nlyc)

        oii_ew_list.append(oii_ew)

    oii_ew_arr = np.asarray(oii_ew_list)
    model_logage_arr = np.asarray(model_logage_list)
    model_age_arr = 10**model_logage_arr

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(model_age_arr, oii_ew_arr, 'o', markersize=2)

    ax.set_xlim(100e6, 6e9)
    ax.set_xscale('log')

    plt.show()

    sys.exit(0)

    return None

def emission_lines(metallicity, bc03_spec_lam, bc03_spec, nlyc, silent=True):

    # Metallicity dependent line ratios relative to H-beta flux
    # Now use the relations specified in Anders & Alvensleben 2003 A&A
    hbeta_flux = 4.757e-13 * nlyc  # equation on second page of the paper

    if not silent:
        print "H-beta flux:", hbeta_flux, "erg s^-1"

    # Make sure the units of hte H-beta flux and the BC03 spectrum are the same 
    # BC03 spectra are in units of L_sol A^-1 i.e. 3.826e33 erg s^-1 A^-1
    # The H-beta flux as written above is in erg s^-1
    # 
    hbeta_flux /= 3.826e33

    # ------------------ Metal lines ------------------ #
    # Read in line list for non-Hydrogen metal emission lines
    non_H_linelist = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/linelist_anders_2003.txt', dtype=None, names=True)

    # ------------------ Hydrogen Lines ------------------ #
    # Line intensity relative to H-beta as calculated by Hummer & Storey 1987 MNRAS
    # Case B recombination assumed along with Te = 1e4 K and Ne = 1e2 cm-3
    # Check page 59 of the pdf copy of hte paper
    halpha_ratio = 2.86
    hgamma_ratio = 4.68e-1
    hdelta_ratio = 2.59e-1

    # ------------------ Line Wavelengths in Vacuum ------------------ #
    # Hydrogen
    h_alpha = 6564.61
    h_beta = 4862.68
    h_gamma = 4341.68
    h_delta = 4102.89

    # Metals
    # I'm defining only the important ones that I want to include here
    # Not using the wavelengths given in the Anders & Alvensleben 2003 A&A paper
    # since those are air wavelengths.
    MgII = 2799.117
    OII_1 = 3727.092
    OIII_1 = 4960.295
    OIII_2 = 5008.240
    NII_1 = 6549.86
    NII_2 = 6585.27
    SII_1 = 6718.29
    SII_2 = 6732.67

    # ------------------ Put lines in ------------------ #
    # The Anders paper only has 5 metallicity values but the BC03 models have 6.
    # I"m simply using the same line ratios from the Anders paper for the lowest 
    # two metallicities. I'm not sure how to do it differently for now.

    # Set the metallicity to the column name in the linelist file
    if (metallicity == 0.008) or (metallicity == 0.02) or (metallicity == 0.05):
        metallicity = 'z3_z5'
    elif (metallicity == 0.004):
        metallicity = 'z2'
    elif (metallicity == 0.0004) or (metallicity == 0.0001):
        metallicity = 'z1'

    # I'm going to put the liens in at the exact wavelengths.
    # So, I'll interpolate the model flam array to get a measuremnet
    # at the exact line wavelength and then just add the line flux to hte continuum.

    # I know that I haven't set up these arrays to find line ratios efficiently
    # Will improve later.
    all_line_wav = np.array([MgII, OII_1, OIII_1, OIII_2, NII_1, NII_2, SII_1, SII_2, h_alpha, h_beta, h_gamma, h_delta])
    all_line_names = np.array(['MgII', 'OII_1', 'OIII_1', 'OIII_2', 'NII_1', 'NII_2', 'SII_1', 'SII_2', 'h_alpha', 'h_beta', 'h_gamma', 'h_delta'])
    all_hlines = np.array(['h_alpha', 'h_gamma', 'h_delta'])
    all_hline_ratios = np.array([halpha_ratio, hgamma_ratio, hdelta_ratio])

    bc03_spec_lam_withlines = bc03_spec_lam
    bc03_spec_withlines = bc03_spec

    for i in range(len(all_line_wav)):

        line_wav = all_line_wav[i]

        line_idx_exact = np.where(bc03_spec_lam_withlines == line_wav)[0]

        if not line_idx_exact.size:
            # This if check is for redundancy
            # i.e. if there isn't an exact measurement at the line wavelength which is almost certainly the case

            insert_idx = np.where(bc03_spec_lam_withlines > line_wav)[0][0]
            # The additional [0] makes sure that only the first wavelength greater than line_wav gets chosen

            # Now interpolate the flux and insert both continuum flux and wavelength
            f = interp1d(bc03_spec_lam_withlines, bc03_spec_withlines)  # This gives the interpolating function
            flam_insert_val = f(line_wav)  # Evaluate the interpolating function at the exact wavelength needed
            bc03_spec_withlines = np.insert(bc03_spec_withlines, insert_idx, flam_insert_val)
            bc03_spec_lam_withlines = np.insert(bc03_spec_lam_withlines, insert_idx, line_wav)
            # This inserts the line wavelength into the wavelength array

            # Now add the line
            # Get line flux first
            line_name = all_line_names[i]
            if 'h' in line_name:
                if line_name == 'h_beta':
                    line_ratio = 1.0
                    line_flux = hbeta_flux
                else:
                    hline_idx = np.where(all_hlines == line_name)[0]
                    line_ratio = float(all_hline_ratios[hline_idx])
                    # This forced type conversion here makes sure that I only have a single idx from the np.where search
                    line_flux = hbeta_flux * line_ratio

            else:
                line_name = '[' + line_name + ']'
                metal_line_idx = np.where(non_H_linelist['line'] == line_name)[0]
                line_ratio = float(non_H_linelist[metallicity][metal_line_idx])
                line_flux = hbeta_flux * line_ratio
                """
                if line_name == '[OII_1]':
                    oii_flux = line_flux

                    # find pseudo continuum level
                    pseudo_cont_arr_left = bc03_spec[insert_idx-20:insert_idx-10]
                    pseudo_cont_arr_right = bc03_spec[insert_idx+10:insert_idx+20]
                    cont_level = np.nanmean(np.concatenate((pseudo_cont_arr_left, pseudo_cont_arr_right)))

                    oii_ew = oii_flux / cont_level
                """

            if not silent:
                print "\n", "Adding line", line_name, "at wavelength", line_wav
                print "This line has a intensity relative to H-beta:", line_ratio

            # Add line to continuum
            idx = np.where(bc03_spec_lam_withlines == line_wav)[0]
            bc03_spec_withlines[idx] += line_flux

    # Plot to check
    if not silent:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(bc03_spec_lam_withlines, bc03_spec_withlines)
        ax.plot(bc03_spec_lam, bc03_spec)

        ax.set_xlim(1e3, 1e4)

        plt.show()

    return bc03_spec_lam_withlines, bc03_spec_withlines

def check_modified_lsf(lsf, modified_lsf):

    print "Original LSF:", lsf
    print "Modified LSF:", modified_lsf

    print "Original LSF length:", len(lsf)
    print "Modified LSF length:", len(modified_lsf)

    print "Area under original LSF:", simps(lsf)
    print "Area under modified LSF:", simps(modified_lsf)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(np.arange(len(lsf)), lsf, color='midnightblue')
    ax.plot(np.arange(len(modified_lsf)), modified_lsf, color='indianred')

    plt.show()

    return None

def lsf_convolve(model_comp_spec, lsf, total_models):

    model_comp_spec_lsfconv = np.zeros(model_comp_spec.shape, dtype=np.float64)

    for i in range(total_models):
        model_comp_spec_lsfconv[i] = fftconvolve(model_comp_spec[i], lsf, mode = 'same')

    return model_comp_spec_lsfconv

def redshift_and_resample(model_comp_spec_lsfconv, z, total_models, model_lam_grid, resampling_lam_grid, resampling_lam_grid_length):

    # --------------- Redshift model --------------- #
    redshift_factor = 1.0 + z
    model_lam_grid_z = model_lam_grid * redshift_factor
    model_comp_spec_redshifted = model_comp_spec_lsfconv / redshift_factor

    # --------------- Do resampling --------------- #
    # Define array to save modified models
    model_comp_spec_modified = np.zeros((total_models, resampling_lam_grid_length), dtype=np.float64)

    # --------------- Get indices for resampling --------------- #
    # These indices are going to be different each time depending on the redshfit.
    # i.e. Since it uses the redshifted model_lam_grid_z to get indices.
    indices = []
    ### Zeroth element
    lam_step = resampling_lam_grid[1] - resampling_lam_grid[0]
    indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[0] - lam_step) & (model_lam_grid_z < resampling_lam_grid[0] + lam_step))[0])

    ### all elements in between
    for i in range(1, resampling_lam_grid_length - 1):
        indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[i-1]) & (model_lam_grid_z < resampling_lam_grid[i+1]))[0])

    ### Last element
    lam_step = resampling_lam_grid[-1] - resampling_lam_grid[-2]
    indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[-1] - lam_step) & (model_lam_grid_z < resampling_lam_grid[-1] + lam_step))[0])

    # ---------- Run for loop to resample ---------- #
    for k in range(total_models):
        for q in range(resampling_lam_grid_length):
            model_comp_spec_modified[k, q] = np.mean(model_comp_spec_redshifted[k][indices[q]])

    return model_comp_spec_modified

def get_chi2(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs,\
        all_filt_flam_model, model_comp_spec_mod, model_resampling_lam_grid, total_models):

    # chop the model to be consistent with the objects lam grid
    model_lam_grid_indx_low = np.argmin(abs(model_resampling_lam_grid - grism_lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(abs(model_resampling_lam_grid - grism_lam_obs[-1]))
    model_spec_in_objlamgrid = model_comp_spec_mod[:, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # make sure that the arrays are the same length
    if int(model_spec_in_objlamgrid.shape[1]) != len(grism_lam_obs):
        print "Arrays of unequal length. Must be fixed before moving forward. Exiting..."
        print "Model spectrum array shape:", model_spec_in_objlamgrid.shape
        print "Object spectrum length:", len(grism_lam_obs)
        sys.exit(0)

    if use_broadband:
        # For both data and model, combine grism+photometry into one spectrum.
        # The chopping above has to be done before combining the grism+photometry
        # because todo the insertion correctly the model and grism wavelength
        # grids have to match.

        # Convert the model array to a python list of lists
        # This has to be done because np.insert() returns a new changed array
        # with the new value inserted but I cannot assign it back to the old
        # array because that changes the shape. This works for the grism arrays
        # since I'm simply using variable names to point to them but since the
        # model array is 2D I'm using indexing and that causes the np.insert()
        # statement to throw an error.
        model_spec_in_objlamgrid_list = []
        for j in range(total_models):
            model_spec_in_objlamgrid_list.append(model_spec_in_objlamgrid[j].tolist())

        count = 0
        combined_lam_obs = grism_lam_obs
        combined_flam_obs = grism_flam_obs
        combined_ferr_obs = grism_ferr_obs
        for phot_wav in phot_lam_obs:

            if phot_wav < combined_lam_obs[0]:
                lam_obs_idx_to_insert = 0

            elif phot_wav > combined_lam_obs[-1]:
                lam_obs_idx_to_insert = len(combined_lam_obs)

            else:
                lam_obs_idx_to_insert = np.where(combined_lam_obs > phot_wav)[0][0]

            # For grism
            combined_lam_obs = np.insert(combined_lam_obs, lam_obs_idx_to_insert, phot_wav)
            combined_flam_obs = np.insert(combined_flam_obs, lam_obs_idx_to_insert, phot_flam_obs[count])
            combined_ferr_obs = np.insert(combined_ferr_obs, lam_obs_idx_to_insert, phot_ferr_obs[count])

            # For model
            for i in range(total_models):
                model_spec_in_objlamgrid_list[i] = np.insert(model_spec_in_objlamgrid_list[i], lam_obs_idx_to_insert, all_filt_flam_model[i, count])

            count += 1

        # Convert back to numpy array
        del model_spec_in_objlamgrid  # Trying to free up the memory allocated to the object pointed by the older model_spec_in_objlamgrid
        # Not sure if the del works because I'm using the same name again. Also just not sure of how del exactly works.
        model_spec_in_objlamgrid = np.asarray(model_spec_in_objlamgrid_list)

        # compute alpha and chi2
        alpha_ = np.sum(combined_flam_obs * model_spec_in_objlamgrid / (combined_ferr_obs**2), axis=1) / np.sum(model_spec_in_objlamgrid**2 / combined_ferr_obs**2, axis=1)
        chi2_ = np.sum(((combined_flam_obs - (alpha_ * model_spec_in_objlamgrid.T).T) / combined_ferr_obs)**2, axis=1)

        print "Min chi2 for redshift:", min(chi2_)

    else:
        # Check and make sure that all the photometry arrays are zeros
        assert not np.any(phot_flam_obs), "Not using broadband data in fit but photometry flux array contains non-zero elements."
        assert not np.any(phot_ferr_obs), "Not using broadband data in fit but photometry flux error array contains non-zero elements."
        assert not np.any(phot_lam_obs), "Not using broadband data in fit but photometry wavelength array contains non-zero elements."
        assert not np.any(all_filt_flam_model), "Not using broadband data in fit but model photometry flux array contains non-zero elements."

        # compute alpha and chi2
        alpha_ = np.sum(grism_flam_obs * model_spec_in_objlamgrid / (grism_ferr_obs**2), axis=1) / np.sum(model_spec_in_objlamgrid**2 / grism_ferr_obs**2, axis=1)
        chi2_ = np.sum(((grism_flam_obs - (alpha_ * model_spec_in_objlamgrid.T).T) / grism_ferr_obs)**2, axis=1)

        print "Min chi2 for redshift:", min(chi2_)

    # This following block is useful for debugging.
    # Do not delete. Simply uncomment it if you don't need it.
    """
    # Check arrays and shapes
    print alpha_
    print chi2_
    print alpha_.shape
    print chi2_.shape

    print combined_lam_obs.shape
    print combined_flam_obs.shape
    print combined_ferr_obs.shape

    print model_spec_in_objlamgrid.shape

    # Do the computation explicitly with nested for loops and check that the arrays are the same
    alpha_explicit = np.zeros((total_models), dtype=np.float64)
    chi2_explicit = np.zeros((total_models), dtype=np.float64)
    len_data = len(combined_lam_obs)
    print "Length of combined obs data:", len_data
    print "Length of each model with its combined data (should be same as above for obs data):", model_spec_in_objlamgrid.shape[1]
    for i in range(total_models):

        alpha_num = 0
        alpha_den = 0
            
        chi2_num = 0
        chi2_den = 0

        for j in range(len_data):
            alpha_num += combined_flam_obs[j] * model_spec_in_objlamgrid[i, j] / combined_ferr_obs[j]**2
            alpha_den += model_spec_in_objlamgrid[i, j]**2 / combined_ferr_obs[j]**2
            
        alpha_explicit[i] = alpha_num / alpha_den

        chi2_terms_sum = 0
        for j in range(len_data):

            chi2_num = (combined_flam_obs[j] - alpha_explicit[i] * model_spec_in_objlamgrid[i, j])**2
            chi2_den = combined_ferr_obs[j]**2

            chi2_terms_sum += chi2_num / chi2_den

        chi2_explicit[i] = chi2_terms_sum

    print alpha_explicit
    print chi2_explicit
    print alpha_explicit.shape
    print chi2_explicit.shape

    print "Test for equality for alpha computation done implicitly (vectorized) and explicitly (for loops):", np.array_equal(alpha_, alpha_explicit)
    print "Test for closeness for alpha computation done implicitly (vectorized) and explicitly (for loops):", np.allclose(alpha_, alpha_explicit)

    print "Test for equality for chi2 computation done implicitly (vectorized) and explicitly (for loops):", np.array_equal(chi2_, chi2_explicit)
    print "Test for closeness for chi2 computation done implicitly (vectorized) and explicitly (for loops):", np.allclose(chi2_, chi2_explicit)

    # plot to check
    for i in range(5005, 5015):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(combined_lam_obs, combined_flam_obs, 'o-', color='k', markersize=2)
        ax.fill_between(combined_lam_obs, combined_flam_obs + combined_ferr_obs, combined_flam_obs - combined_ferr_obs, color='lightgray')
        ax.plot(combined_lam_obs, alpha_[i]*model_spec_in_objlamgrid[i], ls='-', color='r')

        ax.set_xlim(1e3, 2e4)

        plt.show()
    """

    return chi2_, alpha_

def get_chi2_alpha_at_z(z, grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs, \
    model_lam_grid, model_comp_spec, model_comp_spec_lsfconv, \
    resampling_lam_grid, resampling_lam_grid_length, total_models, start_time):

    print "\n", "Currently at redshift:", z

    # make sure the types are correct before passing to cython code
    #lam_obs = lam_obs.astype(np.float64)
    #model_lam_grid = model_lam_grid.astype(np.float64)
    #model_comp_spec = model_comp_spec.astype(np.float64)
    #resampling_lam_grid = resampling_lam_grid.astype(np.float64)
    #total_models = int(total_models)
    #lsf = lsf.astype(np.float64)

    # ------------ Get photomtery for model by convolving with filters ------------- #
    """
    The photometry is computed using the higest resolution model spectrum 
    available.... since this is how mother nature kinda does it anyway.
    The only effect you need to take into account is redshift.
    """
    if use_broadband:

        all_filt_flam_model = np.zeros((num_filters, total_models), dtype=np.float64)

        # Redshift the base models and also the lsf convolved model flux
        model_comp_spec_z = model_comp_spec / (1+z)
        model_lam_grid_z = model_lam_grid * (1+z)

        model_comp_spec_lsfconv_z = model_comp_spec_lsfconv / (1+z)

        filt_count = 0
        for filt in all_filters:

            # first interpolate the grism transmission curve to the model lam grid
            filt_interp = griddata(points=filt.binset, values=filt(filt.binset), xi=model_lam_grid_z, method='linear')

            # multiply model spectrum to filter curve
            for i in range(total_models):

                num = np.nansum(model_comp_spec_z[i] * filt_interp)
                den = np.nansum(filt_interp)

                filt_flam_model = num / den
                all_filt_flam_model[filt_count,i] = filt_flam_model

            filt_count += 1

        # transverse array to make shape consistent with others
        # I did it this way so that in the above for loop each filter is looped over only once
        # i.e. minimizing the number of times each filter is gridded on to the model grid
        all_filt_flam_model = all_filt_flam_model.T

        print "Filter f_lam for models computed."
        print "Total time taken up to now --", time.time() - start_time, "seconds."

    else:
        all_filt_flam_model = np.zeros((num_filters, total_models), dtype=np.float64)

    # ------------- Now do the modifications for the grism data and get a chi2 using both grism and photometry ------------- #
    # first modify the models at the current redshift to be able to compare with data
    model_comp_spec_modified = mm.redshift_and_resample(model_comp_spec_lsfconv, z, total_models, model_lam_grid, resampling_lam_grid, resampling_lam_grid_length)
    print "Model mods done at current z:", z
    print "Total time taken up to now --", time.time() - start_time, "seconds."

    # Check all model modifications and that the model photometry line up
    # Do not delete. Useful for debugging.
    """
    for i in range(10):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(model_lam_grid_z, model_comp_spec_z[i], color='k')
        ax.plot(model_lam_grid_z, model_comp_spec_lsfconv_z[i], color='lawngreen', zorder=3)
        ax.plot(resampling_lam_grid, model_comp_spec_modified[i], color='fuchsia', zorder=5)
        ax.scatter(phot_lam_obs, all_filt_flam_model[i], s=40, color='red', zorder=10)

        axt = ax.twinx()
        for filt in all_filters:
            axt.plot(filt.binset, filt(filt.binset))
            axt.fill_between(filt.binset, filt(filt.binset), np.zeros(len(filt.binset)), alpha=0.25)

        ax.set_xlim(1e3, 2e4)

        plt.show()
        plt.clf()
        plt.cla()
        plt.close()
    """

    # ------------- Now do the chi2 computation ------------- #
    chi2_temp, alpha_temp = get_chi2(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs,\
        all_filt_flam_model, model_comp_spec_modified, resampling_lam_grid, total_models)

    return chi2_temp, alpha_temp

def do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs, \
    lsf, starting_z, resampling_lam_grid, resampling_lam_grid_length, \
    model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start_time,\
    obj_id, obj_field, specz, photoz, netsig, d4000, search_range):
    """
    All models are redshifted to each of the redshifts in the list defined below,
    z_arr_to_check. Then the model modifications are done at that redshift.

    For each iteration through the redshift list it computes a chi2 for each model.
    """

    # Set up redshift grid to check
    z_arr_to_check = np.linspace(start=starting_z - search_range, stop=starting_z + search_range, num=41, dtype=np.float64)
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

    # First do the convolution with the LSF
    model_comp_spec_lsfconv = lsf_convolve(model_comp_spec, lsf, total_models)
    print "Convolution done.",
    print "Total time taken up to now --", time.time() - start_time, "seconds."

    # looping
    num_cores = 4
    chi2_alpha_list = Parallel(n_jobs=num_cores)(delayed(get_chi2_alpha_at_z)(z, \
    grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs, \
    model_lam_grid, model_comp_spec, model_comp_spec_lsfconv, \
    resampling_lam_grid, resampling_lam_grid_length, total_models, start_time) \
    for z in z_arr_to_check)

    # the parallel code seems to like returning only a list
    # so I have to unpack the list
    for i in range(len(z_arr_to_check)):
        chi2[i], alpha[i] = chi2_alpha_list[i]

    # regular for loop 
    # use this if you dont want to use the parallel for loop above
    # comment it out if you don't need it
    """
    count = 0
    for z in z_arr_to_check:
        chi2[count], alpha[count] = get_chi2_alpha_at_z(z, \
    grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs, \
    model_lam_grid, model_comp_spec, model_comp_spec_lsfconv, \
    resampling_lam_grid, resampling_lam_grid_length, total_models, start_time)
        count += 1
    """

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

    print "Current best fit log(age [yr]):", "{:.4}".format(age)
    print "Current best fit Tau [Gyr]:", "{:.4}".format(tau)
    print "Current best fit Tau_V:", tauv

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
    if use_broadband:
        dof = len(grism_lam_obs) + len(phot_lam_obs) - 1  # i.e. total data points minus the single fitting parameter
    else:
        dof = len(grism_lam_obs) - 1  # i.e. total data points minus the single fitting parameter

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

    # Simply the minimum chi2 might not be right
    # Should check if the minimum is global or local
    #ngp.plot_chi2(chi2, dof, z_arr_to_check, z_grism, specz, obj_id, obj_field, total_models)
    # Save chi2 map
    np.save(massive_figures_dir + 'large_diff_specz_sample/' + obj_field + '_' + str(obj_id) + '_chi2_map.npy', chi2/dof)
    np.save(massive_figures_dir + 'large_diff_specz_sample/' + obj_field + '_' + str(obj_id) + '_z_arr.npy', z_arr_to_check)

    pz = get_pz_and_plot(chi2/dof, z_arr_to_check, specz, photoz, z_grism, low_z_lim, upper_z_lim, obj_id, obj_field)

    # Save p(z)
    np.save(massive_figures_dir + 'large_diff_specz_sample/' + obj_field + '_' + str(obj_id) + '_pz.npy', pz)
    z_wt = np.sum(z_arr_to_check * pz)
    print "Weighted z:", "{:.3}".format(z_wt)
    print "Grism redshift:", z_grism
    print "Ground-based spectroscopic redshift [-99.0 if it does not exist]:", specz
    print "Photometric redshift:", photoz

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
    """
    # chop model again to get the part within objects lam obs grid
    model_lam_grid_indx_low = np.argmin(abs(resampling_lam_grid - grism_lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(abs(resampling_lam_grid - grism_lam_obs[-1]))

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
    model_comp_spec_modified = mm.redshift_and_resample(model_comp_spec_lsfconv, z_grism, total_models, model_lam_grid, resampling_lam_grid, resampling_lam_grid_length)
    print "Model mods done (only for plotting purposes) at the new grism z:", z_grism
    print "Total time taken up to now --", time.time() - start_time, "seconds."

    best_fit_model_in_objlamgrid = model_comp_spec_modified[model_idx, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    if use_broadband:

        # ------------ Get photomtery for model by convolving with filters ------------- #
        # This has to be done again at the correct z_grism
        all_filt_flam_model = np.zeros((num_filters, total_models), dtype=np.float64)

        # Redshift the base models
        model_comp_spec_z = model_comp_spec / (1+z_grism)
        model_lam_grid_z = model_lam_grid * (1+z_grism)
        filt_count = 0
        for filt in all_filters:

            # first interpolate the grism transmission curve to the model lam grid
            filt_interp = griddata(points=filt.binset, values=filt(filt.binset), xi=model_lam_grid_z, method='linear')

            # multiply model spectrum to filter curve
            for i in range(total_models):

                num = np.nansum(model_comp_spec_z[i] * filt_interp)
                den = np.nansum(filt_interp)

                filt_flam_model = num / den
                all_filt_flam_model[filt_count,i] = filt_flam_model

            filt_count += 1

        # transverse array to make shape consistent with others
        # I did it this way so that in the above for loop each filter is looped over only once
        # i.e. minimizing the number of times each filter is gridded on to the model grid
        all_filt_flam_model = all_filt_flam_model.T

        # Get the flam for the best model
        all_filt_flam_bestmodel = all_filt_flam_model[model_idx]

    else:
        all_filt_flam_bestmodel = np.zeros(num_filters)

    # Get best fit model at full resolution
    best_fit_model_fullres = model_comp_spec[model_idx]
    """

    # ---------------------------------------------------------
    # again make sure that the arrays are the same length
    #if int(best_fit_model_in_objlamgrid.shape[0]) != len(lam_obs):
    #    print "Arrays of unequal length. Must be fixed before moving forward. Exiting..."
    #    sys.exit(0)
    # plot
    #plot_fit(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs,
    #    all_filt_flam_bestmodel, best_fit_model_in_objlamgrid, bestalpha, model_lam_grid, best_fit_model_fullres,
    #    obj_id, obj_field, specz, photoz, z_grism, low_z_lim, upper_z_lim, min_chi2_red, age, tau, (tauv/1.086), netsig, d4000, z_wt)

    return z_grism, low_z_lim, upper_z_lim, min_chi2_red, age, tau, (tauv/1.086)

def plot_fit(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs,
    all_filt_flam_bestmodel, best_fit_model_in_objlamgrid, bestalpha, model_lam_grid, best_fit_model_fullres,
    obj_id, obj_field, specz, photoz, grismz, low_z_lim, upper_z_lim, chi2, age, tau, av, netsig, d4000, weightedz):

    # ---------- Create figure ---------- #
    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[:8,:])
    ax2 = fig.add_subplot(gs[8:,:])

    # ---------- labels ---------- #
    ax1.set_ylabel(r'$\mathrm{f_\lambda\ [erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]}$')
    ax2.set_xlabel(r'$\mathrm{Wavelength\, [\AA]}$')
    ax2.set_ylabel(r'$\mathrm{\frac{f^{obs}_\lambda\ - f^{model}_\lambda}{f^{obs;error}_\lambda}}$')

    # ---------- plot data, model, and residual ---------- #
    # plot full res model but you'll have to redshift it
    ax1.plot(model_lam_grid * (1+grismz), bestalpha*best_fit_model_fullres / (1+grismz), color='mediumblue', alpha=0.3)

    # plot data
    ax1.plot(grism_lam_obs, grism_flam_obs, 'o-', color='k', markersize=2, zorder=10)
    ax1.fill_between(grism_lam_obs, grism_flam_obs + grism_ferr_obs, grism_flam_obs - grism_ferr_obs, color='lightgray', zorder=10)

    if use_broadband:
        ax1.errorbar(phot_lam_obs, phot_flam_obs, yerr=phot_ferr_obs, \
            fmt='.', color='midnightblue', markeredgecolor='midnightblue', \
            capsize=2, markersize=10.0, elinewidth=2.0)

    # plot best fit model
    ax1.plot(grism_lam_obs, bestalpha*best_fit_model_in_objlamgrid, ls='-', color='indianred', zorder=20)

    if use_broadband:
        ax1.scatter(phot_lam_obs, bestalpha*all_filt_flam_bestmodel, s=20, color='indianred', zorder=20)

    # Residuals
    # For the grism points
    resid_fit_grism = (grism_flam_obs - bestalpha*best_fit_model_in_objlamgrid) / grism_ferr_obs

    # Now plot
    ax2.scatter(grism_lam_obs, resid_fit_grism, s=4, color='k')
    ax2.axhline(y=0, ls='--', color='k')

    if use_broadband:
        # For the photometry
        resid_fit_phot = (phot_flam_obs - bestalpha*all_filt_flam_bestmodel) / phot_ferr_obs
        ax2.scatter(phot_lam_obs, resid_fit_phot, s=4, color='k')

    # ---------- limits ---------- #
    max_y_obs = np.max(np.concatenate((grism_flam_obs, phot_flam_obs)))
    min_y_obs = np.min(np.concatenate((grism_flam_obs, phot_flam_obs)))

    max_ylim = 1.25 * max_y_obs
    min_ylim = 0.75 * min_y_obs

    if use_broadband:
        max_ylim = 1.25 * max_y_obs
        min_ylim = 0.75 * min_y_obs

        ax1.set_ylim(min_ylim, max_ylim)

        ax1.set_xlim(3000, 17000)
        ax2.set_xlim(3000, 17000)

    else:
        max_ylim = 1.1 * max_y_obs
        min_ylim = 0.9 * min_y_obs

        ax1.set_ylim(min_ylim, max_ylim)

        ax1.set_xlim(6000, 9500)
        ax2.set_xlim(6000, 9500)

    # ---------- minor ticks ---------- #
    ax1.minorticks_on()
    ax2.minorticks_on()

    # ---------- text for info ---------- #
    ax1.text(0.75, 0.45, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.75, 0.4, r'$\mathrm{z_{weighted}\, =\, }$' + "{:.4}".format(weightedz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    low_zerr = grismz - low_z_lim
    high_zerr = upper_z_lim - grismz

    ax1.text(0.75, 0.35, \
    r'$\mathrm{z_{min\,\chi^2}\, =\, }$' + "{:.4}".format(grismz) + \
    r'$\substack{+$' + "{:.3}".format(high_zerr) + r'$\\ -$' + "{:.3}".format(low_zerr) + r'$}$', \
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

    ax1.text(0.75, 0.12, r'$\mathrm{NetSig\, =\, }$' + mr.convert_to_sci_not(netsig), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=8)
    ax1.text(0.75, 0.07, r'$\mathrm{D4000(from\ z_{phot})\, =\, }$' + "{:.3}".format(d4000), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=8)


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

    # ---------- Save figure ---------- #
    if use_broadband:
        fig.savefig(massive_figures_dir + 'large_diff_specz_sample/' + obj_field + '_' + str(obj_id) + '.png', \
            dpi=300, bbox_inches='tight')
    else:
        fig.savefig(massive_figures_dir + 'large_diff_specz_sample/' + obj_field + '_' + str(obj_id) + '_NoPhotometry.png', \
            dpi=300, bbox_inches='tight')
    
    plt.clf()
    plt.cla()
    plt.close()

    return None

def get_pz_and_plot(chi2_map, z_arr_to_check, specz, photoz, grismz, low_z_lim, upper_z_lim, obj_id, obj_field):

    # Convert chi2 to likelihood
    likelihood = np.exp(-1 * chi2_map / 2)

    # Normalize likelihood function
    norm_likelihood = likelihood / np.sum(likelihood)

    # Get p(z)
    pz = np.zeros(len(z_arr_to_check))

    for i in range(len(z_arr_to_check)):
        pz[i] = np.sum(norm_likelihood[i])

    # Find peak in p(z) and a measure of uncertainty in grism_z

    # PLot and save plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(z_arr_to_check, pz)

    ax.minorticks_on()

    # ---------- text for info ---------- #
    ax.text(0.65, 0.4, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=10)

    low_zerr = grismz - low_z_lim
    high_zerr = upper_z_lim - grismz

    ax.text(0.65, 0.35, \
    r'$\mathrm{z_{grism}\, [from\ min\ \chi^2]\, =\, }$' + "{:.4}".format(grismz) + \
    r'$\substack{+$' + "{:.3}".format(high_zerr) + r'$\\ -$' + "{:.3}".format(low_zerr) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=10)
    ax.text(0.65, 0.27, r'$\mathrm{z_{spec}\, =\, }$' + "{:.4}".format(specz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=10)
    ax.text(0.65, 0.22, r'$\mathrm{z_{phot}\, =\, }$' + "{:.4}".format(photoz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=10)

    fig.savefig(massive_figures_dir + 'large_diff_specz_sample/' +  obj_field + '_' + str(obj_id) + '_pz.png', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()
    """

    return pz

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # make_oii_ew_vs_age_plot()
    # show_example_for_adding_emission_lines()
    # sys.exit(0)

    # Flags to turn on-off broadband and emission lines in the fit
    use_broadband = True
    use_emlines = True
    modify_lsf = True
    num_filters = 7

    # ------------------------------ Add emission lines to models ------------------------------ #
    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_lam_grid = model_lam_grid.astype(np.float64)

    total_emission_lines_to_add = 12  # Make sure that this changes if you decide to add more lines to the models
    model_comp_spec_withlines = np.zeros((total_models, len(model_lam_grid) + total_emission_lines_to_add), dtype=np.float64)
    for j in range(total_models):
        nlyc = float(bc03_all_spec_hdulist[j+1].header['NLYC'])
        metallicity = float(bc03_all_spec_hdulist[j+1].header['METAL'])
        model_lam_grid_withlines, model_comp_spec_withlines[j] = \
        emission_lines(metallicity, model_lam_grid, bc03_all_spec_hdulist[j+1].data, nlyc)
        # Also checked that in every case model_lam_grid_withlines is the exact same
        # SO i'm simply using hte output from the last model.

        # To include the models without the lines as well you will have to make sure that 
        # the two types of models (ie. with and without lines) are on the same lambda grid.
        # I guess you could simply interpolate the model without lines on to the grid of
        # the models wiht lines.

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

    # ----------------------------------------- READ IN CATALOGS ----------------------------------------- #
    # read in matched files to get photo-z
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    # Read in Specz comparison catalogs
    specz_goodsn = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-N.txt', dtype=None, names=True)
    specz_goodss = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-S.txt', dtype=None, names=True)

    # ------------------------------- Read in photometry and grism+photometry catalogs ------------------------------- #
    # GOODS-N from 3DHST
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55), skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55), skip_header=3)

    # large differences between specz and grismz
    #large_diff_cat = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/large_diff_specz_short.txt', dtype=None, names=True)

    all_speccats = [specz_goodss]
    all_match_cats = [matched_cat_s]

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
    catcount = 1
    galaxy_count = 0
    for cat in all_match_cats:

        for i in range(len(cat)):

            # --------------------------------------------- GET OBS DATA ------------------------------------------- #
            current_id = cat['pearsid'][i]

            if catcount == 0:
                current_field = 'GOODS-N'
                spec_cat = specz_goodsn
                phot_cat_3dhst = goodsn_phot_cat_3dhst
            elif catcount == 1: 
                current_field = 'GOODS-S'
                spec_cat = specz_goodss
                phot_cat_3dhst = goodss_phot_cat_3dhst
    
            threed_ra = phot_cat_3dhst['ra']
            threed_dec = phot_cat_3dhst['dec']

            # Get specz if it exists as initial guess, otherwise get photoz
            specz_idx = np.where(spec_cat['pearsid'] == current_id)[0]

            # For now running this code only on the specz sample
            if len(specz_idx) != 1:
                print "Match not found in specz catalog for ID", current_id, "in", current_field, "... Skipping."
                continue

            if len(specz_idx) == 1:
                current_specz = float(spec_cat['specz'][specz_idx])
                current_photz = float(cat['zphot'][i])
                starting_z = current_specz
            elif len(specz_idx) == 0:
                current_specz = -99.0
                current_photz = float(cat['zphot'][i])
                starting_z = current_photz
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
            #current_id = 118100
            #current_field = 'GOODS-S'
            #current_specz = 0.644
            #current_photz = 0.627
            #starting_z = current_specz

            print "At ID", current_id, "in", current_field, "with specz and photo-z:", current_specz, current_photz

            grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)

            if return_code == 0:
                print "Skipping due to an error with the obs data. See the error message just above this one.",
                print "Moving to the next galaxy."
                continue

            if use_broadband:
                # ------------------------------- Match and get photometry data ------------------------------- #
                # find grism obj ra,dec
                cat_idx = np.where(cat['pearsid'] == current_id)[0]
                if cat_idx.size:
                    current_ra = float(cat['pearsra'][cat_idx])
                    current_dec = float(cat['pearsdec'][cat_idx])

                # Now match
                ra_lim = 0.5/3600  # arcseconds in degrees
                dec_lim = 0.5/3600
                threed_phot_idx = np.where((threed_ra >= current_ra - ra_lim) & (threed_ra <= current_ra + ra_lim) & \
                    (threed_dec >= current_dec - dec_lim) & (threed_dec <= current_dec + dec_lim))[0]

                """
                If there are multiple matches with the photometry catalog 
                within 0.5 arseconds then choose the closest one.
                """
                if len(threed_phot_idx) > 1:

                    ra_two = current_ra
                    dec_two = current_dec

                    dist_list = []
                    for v in range(len(threed_phot_idx)):

                        ra_one = threed_ra[threed_phot_idx][v]
                        dec_one = threed_dec[threed_phot_idx][v]

                        dist = np.arccos(np.cos(dec_one*np.pi/180) * np.cos(dec_two*np.pi/180) * np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + 
                            np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
                        dist_list.append(dist)

                    dist_list = np.asarray(dist_list)
                    dist_idx = np.argmin(dist_list)
                    threed_phot_idx = threed_phot_idx[dist_idx]

                # ------------------------------- Get photometric fluxes and their errors ------------------------------- #
                flam_f435w = get_flam('F435W', phot_cat_3dhst['f_F435W'][threed_phot_idx])
                flam_f606w = get_flam('F606W', phot_cat_3dhst['f_F606W'][threed_phot_idx])
                flam_f775w = get_flam('F775W', phot_cat_3dhst['f_F775W'][threed_phot_idx])
                flam_f850lp = get_flam('F850LP', phot_cat_3dhst['f_F850LP'][threed_phot_idx])
                flam_f125w = get_flam('F125W', phot_cat_3dhst['f_F125W'][threed_phot_idx])
                flam_f140w = get_flam('F140W', phot_cat_3dhst['f_F140W'][threed_phot_idx])
                flam_f160w = get_flam('F160W', phot_cat_3dhst['f_F160W'][threed_phot_idx])

                ferr_f435w = get_flam('F435W', phot_cat_3dhst['e_F435W'][threed_phot_idx])
                ferr_f606w = get_flam('F606W', phot_cat_3dhst['e_F606W'][threed_phot_idx])
                ferr_f775w = get_flam('F775W', phot_cat_3dhst['e_F775W'][threed_phot_idx])
                ferr_f850lp = get_flam('F850LP', phot_cat_3dhst['e_F850LP'][threed_phot_idx])
                ferr_f125w = get_flam('F125W', phot_cat_3dhst['e_F125W'][threed_phot_idx])
                ferr_f140w = get_flam('F140W', phot_cat_3dhst['e_F140W'][threed_phot_idx])
                ferr_f160w = get_flam('F160W', phot_cat_3dhst['e_F160W'][threed_phot_idx])

                # ------------------------------- Apply aperture correction ------------------------------- #
                # We need to do this because the grism spectrum and the broadband photometry don't line up properly.
                # This is due to different apertures being used when extrating the grism spectrum and when measuring
                # the broadband flux in any given filter.
                # This will multiply hte grism spectrum by a multiplicative factor that scales it to the measured 
                # i-band (F775W) flux. Basically, the grism spectrum is "convolved" with the F775W filter curve,
                # since this is the filter whose coverage completely overlaps that of the grism, to get an i-band
                # magnitude (or actually f_lambda) using the grism data. The broadband i-band mag is then divided 
                # by this grism i-band mag to get the factor that multiplies the grism spectrum.
                # Filter curves from:
                # ACS: http://www.stsci.edu/hst/acs/analysis/throughputs
                # WFC3: Has to be done through PySynphot. See: https://pysynphot.readthedocs.io/en/latest/index.html
                # Also take a look at this page: http://www.stsci.edu/hst/observatory/crds/throughput.html
                # Pysynphot has been set up correctly. Its pretty useful for many other things too. See Pysynphot docs.

                # read in filter curves
                f435w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f435w')
                f606w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f606w')
                f775w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f775w')
                f850lp_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f850lp')

                f125w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f125w')
                f140w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f140w')
                f160w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f160w')

                all_filters = [f435w_filt_curve, f606w_filt_curve, f775w_filt_curve, f850lp_filt_curve, f125w_filt_curve, f140w_filt_curve, f160w_filt_curve]

                """
                Example to plot filter curves for ACS:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(f435w_filt_curve.binset, f435w_filt_curve(f435w_filt_curve.binset), color='midnightblue')
                ax.plot(f850lp_filt_curve.binset, f850lp_filt_curve(f850lp_filt_curve.binset), color='seagreen')
                ax.plot(f606w_filt_curve.binset, f606w_filt_curve(f606w_filt_curve.binset), color='orange')
                ax.plot(f775w_filt_curve.binset, f775w_filt_curve(f775w_filt_curve.binset), color='rebeccapurple')
                plt.show()
                """

                # First interpolate the given filter curve on to the wavelength frid of the grism data
                # You only need the F775W filter here since you're only using this filter to get the 
                # aperture correction factor.
                f775w_trans_interp = griddata(points=f775w_filt_curve.binset, values=f775w_filt_curve(f775w_filt_curve.binset), xi=grism_lam_obs, method='linear')

                # check that the interpolated curve looks like hte original one
                """
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(f775w_filt_curve['wav'], f775w_filt_curve['trans'])
                ax.plot(lam_obs, f775w_trans_interp)
                plt.show()
                """

                # multiply grism spectrum to filter curve
                num = 0
                den = 0
                for w in range(len(grism_flam_obs)):
                    num += grism_flam_obs[w] * f775w_trans_interp[w]
                    den += f775w_trans_interp[w]

                avg_f775w_flam_grism = num / den
                aper_corr_factor = flam_f775w / avg_f775w_flam_grism
                print "Aperture correction factor:", "{:.3}".format(aper_corr_factor)

                grism_flam_obs *= aper_corr_factor  # applying factor

                # ------------------------------- Make unified photometry arrays ------------------------------- #
                phot_fluxes_arr = np.array([flam_f435w, flam_f606w, flam_f775w, flam_f850lp, flam_f125w, flam_f140w, flam_f160w])
                phot_errors_arr = np.array([ferr_f435w, ferr_f606w, ferr_f775w, ferr_f850lp, ferr_f125w, ferr_f140w, ferr_f160w])

                #phot_errors_arr *= 2.0
                grism_ferr_obs *= 2.0

                # Pivot wavelengths
                # From here --
                # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/#keywords
                # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
                phot_lam = np.array([4328.2, 5921.1, 7692.4, 9033.1, 12486, 13923, 15369])  # angstroms

                # ------------------------------- Plot to check ------------------------------- #
                """
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(grism_lam_obs, grism_flam_obs, 'o-', color='k', markersize=2)
                ax.fill_between(grism_lam_obs, grism_flam_obs + grism_ferr_obs, grism_flam_obs - grism_ferr_obs, color='lightgray')
                plt.show(block=False)

                check_spec_plot(grism_lam_obs, grism_flam_obs, grism_ferr_obs, phot_lam, phot_fluxes_arr, phot_errors_arr)
                sys.exit(0)
                """
            else:
                print "Not using broadband data in fit. Setting photometry related arrays to zero arrays."
                phot_fluxes_arr = np.zeros(num_filters)
                phot_errors_arr = np.zeros(num_filters)
                phot_lam = np.zeros(num_filters)

            # ------------------------------ Now start fitting ------------------------------ #
            # --------- Force dtype for cython code --------- #
            # Apparently this (i.e. for flam_obs and ferr_obs) has  
            # to be done to avoid an obscure error from parallel in joblib --
            # AttributeError: 'numpy.ndarray' object has no attribute 'offset'
            grism_lam_obs = grism_lam_obs.astype(np.float64)
            grism_flam_obs = grism_flam_obs.astype(np.float64)
            grism_ferr_obs = grism_ferr_obs.astype(np.float64)

            phot_lam = phot_lam.astype(np.float64)
            phot_fluxes_arr = phot_fluxes_arr.astype(np.float64)
            phot_errors_arr = phot_errors_arr.astype(np.float64)

            # --------------------------------------------- Quality checks ------------------------------------------- #
            # Netsig check
            if netsig_chosen < 10:
                print "Skipping", current_id, "in", current_field, "due to low NetSig:", netsig_chosen
                continue

            # D4000 check 
            # You have to de-redshift it to get D4000. So if the original z is off then the D4000 will also be off.
            # This is way I'm letting some lower D4000 values into my sample. Just so I don't miss too many galaxies.
            # A few of the galaxies with really wrong starting_z will of course be missed.
            lam_em = grism_lam_obs / (1 + starting_z)
            flam_em = grism_flam_obs * (1 + starting_z)
            ferr_em = grism_ferr_obs * (1 + starting_z)

            # Check that hte lambda array is not too incomplete 
            # I don't want the D4000 code extrapolating too much.
            # I'm choosing this limit to be 50A
            if np.max(lam_em) < 4200:
                print "Skipping because lambda array is incomplete by too much."
                print "i.e. the max val in rest-frame lambda is less than 4200A."
                continue

            d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)
            if (d4000 < 1.1) or (d4000 > 1.6):
                print "Skipping", current_id, "in", current_field, "due to D4000:", d4000
                continue

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
            # FOR DETAILS ON BROADENING LSF METHOD.
            # In here I'm stretching the LSF instead of broadening it.
            if modify_lsf:
                """
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

                #check_modified_lsf(lsf, broad_lsf)  # Comment out if you dont want to check the LSF modification result

                lsf_to_use = broad_lsf
                """

                # Stretch LSF instead of broadening
                lsf_length = len(lsf)
                x_arr = np.arange(lsf_length)
                num_interppoints = int(1.118 * lsf_length)
                stretched_lsf_arr = np.linspace(0, lsf_length, num_interppoints, endpoint=False)
                stretched_lsf = griddata(points=x_arr, values=lsf, xi=stretched_lsf_arr, method='linear')

                # Make sure that the new LSF does not have NaN values in ti
                stretched_lsf = stretched_lsf[np.isfinite(stretched_lsf)]

                # Area under stretched LSF should be 1.0
                current_area = simps(stretched_lsf)
                stretched_lsf *= (1/current_area)

                #check_modified_lsf(lsf, stretched_lsf)  # Comment out if you dont want to check the LSF modification result

                lsf_to_use = stretched_lsf

            else:
                lsf_to_use = lsf

            # ------- Make new resampling grid ------- # 
            # extend lam_grid to be able to move the lam_grid later 
            avg_dlam = old_ref.get_avg_dlam(grism_lam_obs)

            lam_low_to_insert = np.arange(6000, grism_lam_obs[0], avg_dlam, dtype=np.float64)
            lam_high_to_append = np.arange(grism_lam_obs[-1] + avg_dlam, 10000, avg_dlam, dtype=np.float64)

            resampling_lam_grid = np.insert(grism_lam_obs, obj=0, values=lam_low_to_insert)
            resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

            # Make sure that the photometry arrays all have finite values
            # If any vlues are NaN then throw them out
            phot_fluxes_finite_idx = np.where(np.isfinite(phot_fluxes_arr))[0]
            phot_errors_finite_idx = np.where(np.isfinite(phot_errors_arr))[0]

            phot_fin_idx = reduce(np.intersect1d, (phot_fluxes_finite_idx, phot_errors_finite_idx))

            phot_fluxes_arr = phot_fluxes_arr[phot_fin_idx]
            phot_errors_arr = phot_errors_arr[phot_fin_idx]
            phot_lam = phot_lam[phot_fin_idx]

            # ------------- Call actual fitting function ------------- #
            zg, zerr_low, zerr_up, min_chi2, age, tau, av = \
            do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_fluxes_arr, phot_errors_arr, phot_lam, \
                lsf_to_use, starting_z, resampling_lam_grid, len(resampling_lam_grid), \
                model_lam_grid_withlines, total_models, model_comp_spec_withlines, bc03_all_spec_hdulist, start,\
                current_id, current_field, current_specz, current_photz, netsig_chosen, d4000, 0.2)# all_filters)

            # Get d4000 at new zgrism
            lam_em = grism_lam_obs / (1 + zg)
            flam_em = grism_flam_obs * (1 + zg)
            ferr_em = grism_ferr_obs * (1 + zg)

            d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

            # ---------------------------------------------- SAVE PARAMETERS ----------------------------------------------- #
            id_list.append(current_id)
            field_list.append(current_field)
            zgrism_list.append(zg)
            zgrism_lowerr_list.append(zerr_low)
            zgrism_uperr_list.append(zerr_up)
            zspec_list.append(current_specz)
            zphot_list.append(current_photz)
            chi2_list.append(min_chi2)
            netsig_list.append(netsig_chosen)
            age_list.append(age)
            tau_list.append(tau)
            av_list.append(av)
            d4000_list.append(d4000)
            d4000_err_list.append(d4000_err)

            galaxy_count += 1

        catcount += 1

    print "Total galaxies considered:", galaxy_count

    # Convert to numpy arrays
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

    # Save files
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_id_list.npy', id_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_field_list.npy', field_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_zgrism_list.npy', zgrism_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_zgrism_lowerr_list.npy', zgrism_lowerr_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_zgrism_uperr_list.npy', zgrism_uperr_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_zspec_list.npy', zspec_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_zphot_list.npy', zphot_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_chi2_list.npy', chi2_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_netsig_list.npy', netsig_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_age_list.npy', age_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_tau_list.npy', tau_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_av_list.npy', av_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_d4000_list.npy', d4000_list)
    np.save(figs_dir + 'massive-galaxies-figures/large_diff_specz_sample/withemlines_d4000_err_list.npy', d4000_err_list)

    # Close HDUs
    bc03_all_spec_hdulist.close()

    # Total time taken
    print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."
    sys.exit(0)

