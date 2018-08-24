from __future__ import division

import numpy as np
import numpy.ma as ma
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo
from joblib import Parallel, delayed
from scipy.interpolate import griddata, interp1d

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"
threedhst_datadir = home + "/Desktop/3dhst_data/"

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import grid_coadd as gd
import refine_redshifts_dn4000 as old_ref
import model_mods_cython_copytoedit as model_mods_cython
import dn4000_catalog as dc
import new_refine_grismz_gridsearch_parallel as ngp

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

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    #make_oii_ew_vs_age_plot()
    #show_example_for_adding_emission_lines()
    #sys.exit(0)

    # ------------------------------- Read in photometry and grism+photometry catalogs ------------------------------- #
    # GOODS-N from 3DHST
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshfits, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55), skip_header=3)

    threed_ra = goodsn_phot_cat_3dhst['ra']
    threed_dec = goodsn_phot_cat_3dhst['dec']

    # Read in grism data
    current_id = 121302
    current_field = 'GOODS-N'
    lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)
    lam_obs_grism = lam_obs   # Need this later to get avg_dlam

    # ------------------------------- Match and get photometry data ------------------------------- #
    # read in matched files for grism spectra info
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    # find grism obj ra,dec
    cat_idx = np.where(matched_cat_n['pearsid'] == current_id)[0]
    if cat_idx.size:
        zphot = float(matched_cat_n['zphot'][cat_idx])
        current_ra = float(matched_cat_n['pearsra'][cat_idx])
        current_dec = float(matched_cat_n['pearsdec'][cat_idx])

    # Now match
    ra_lim = 0.5/3600  # arcseconds in degrees
    dec_lim = 0.5/3600
    threed_phot_idx = np.where((threed_ra >= current_ra - ra_lim) & (threed_ra <= current_ra + ra_lim) & \
        (threed_dec >= current_dec - dec_lim) & (threed_dec <= current_dec + dec_lim))[0]

    # ------------------------------- Get photometric fluxes and their errors ------------------------------- #
    flam_f435w = get_flam('F435W', goodsn_phot_cat_3dhst['f_F435W'][threed_phot_idx])
    flam_f606w = get_flam('F606W', goodsn_phot_cat_3dhst['f_F606W'][threed_phot_idx])
    flam_f775w = get_flam('F775W', goodsn_phot_cat_3dhst['f_F775W'][threed_phot_idx])
    flam_f850lp = get_flam('F850LP', goodsn_phot_cat_3dhst['f_F850LP'][threed_phot_idx])
    flam_f125w = get_flam('F125W', goodsn_phot_cat_3dhst['f_F125W'][threed_phot_idx])
    flam_f140w = get_flam('F140W', goodsn_phot_cat_3dhst['f_F140W'][threed_phot_idx])
    flam_f160w = get_flam('F160W', goodsn_phot_cat_3dhst['f_F160W'][threed_phot_idx])

    ferr_f435w = get_flam('F435W', goodsn_phot_cat_3dhst['e_F435W'][threed_phot_idx])
    ferr_f606w = get_flam('F606W', goodsn_phot_cat_3dhst['e_F606W'][threed_phot_idx])
    ferr_f775w = get_flam('F775W', goodsn_phot_cat_3dhst['e_F775W'][threed_phot_idx])
    ferr_f850lp = get_flam('F850LP', goodsn_phot_cat_3dhst['e_F850LP'][threed_phot_idx])
    ferr_f125w = get_flam('F125W', goodsn_phot_cat_3dhst['e_F125W'][threed_phot_idx])
    ferr_f140w = get_flam('F140W', goodsn_phot_cat_3dhst['e_F140W'][threed_phot_idx])
    ferr_f160w = get_flam('F160W', goodsn_phot_cat_3dhst['e_F160W'][threed_phot_idx])

    # ------------------------------- Apply aperture correction ------------------------------- #
    # We need to do this because the grism spectrum and the broadband photometry don't line up properly.
    # This is due to different apertures being used when extrating the grism spectrum and when measuring
    # the broadband flux in any given filter.
    # This will multiply hte grism spectrum by a multiplicative factor that scales it to the measured 
    # i-band (F775W) flux. Basically, the grism spectrum is "convolved" with the F775W filter curve,
    # since this is the filter whose coverage completely overlaps that of the grism, to get an i-band
    # magnitude (or actually f_lambda) using the grism data. The broadband i-band mag is then divided 
    # by this grism i-band mag to get the factor that multiplies the grism spectrum.

    # read in filter curve
    f775w_filt_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/wfc_F775W.dat', dtype=None, names=['wav', 'trans'])

    # First interpolate the given filter curve on to the wavelength frid of the grism data
    f775w_trans_interp = griddata(points=f775w_filt_curve['wav'], values=f775w_filt_curve['trans'], xi=lam_obs, method='linear')

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
    for i in range(len(flam_obs)):
        num += flam_obs[i] * f775w_trans_interp[i]
        den += f775w_trans_interp[i]

    avg_f775w_flam_grism = num / den
    aper_corr_factor = flam_f775w / avg_f775w_flam_grism
    print "Aperture correction factor:", "{:.3}".format(aper_corr_factor)

    flam_obs *= aper_corr_factor  # applying factor

    # ------------------------------- Make a unified grism+photometry spectrum array ------------------------------- #
    phot_fluxes_arr = np.array([flam_f435w, flam_f606w, flam_f775w, flam_f850lp, flam_f125w, flam_f140w, flam_f160w])
    phot_errors_arr = np.array([ferr_f435w, ferr_f606w, ferr_f775w, ferr_f850lp, ferr_f125w, ferr_f140w, ferr_f160w])

    # Pivot wavelengths
    # From here --
    # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/#keywords
    # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
    phot_lam = np.array([4328.2, 5921.1, 7692.4, 9033.1, 12486, 13923, 15369])  # angstroms

    # Combine grism+photometry into one spectrum
    count = 0
    for phot_wav in phot_lam:
        
        if phot_wav < lam_obs[0]:
            lam_obs_idx_to_insert = 0

        elif phot_wav > lam_obs[-1]:
            lam_obs_idx_to_insert = len(lam_obs)

        else: 
            lam_obs_idx_to_insert = np.where(lam_obs > phot_wav)[0][0] 

        lam_obs = np.insert(lam_obs, lam_obs_idx_to_insert, phot_wav)
        flam_obs = np.insert(flam_obs, lam_obs_idx_to_insert, phot_fluxes_arr[count])
        ferr_obs = np.insert(ferr_obs, lam_obs_idx_to_insert, phot_errors_arr[count])

        count += 1

    # ------------------------------- Plot to check ------------------------------- #
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lam_obs, flam_obs, 'o-', color='k', markersize=2)
    ax.fill_between(lam_obs, flam_obs + ferr_obs, flam_obs - ferr_obs, color='lightgray')
    plt.show(block=False)

    check_spec_plot(lam_obs, flam_obs, ferr_obs, phot_lam, phot_fluxes_arr, phot_errors_arr)
    sys.exit(0)
    """

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

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

    # ------------------------------ Now start fitting ------------------------------ #
    # --------- Get starting redshift
    # Get specz if it exists as initial guess, otherwise get photoz
    current_photz = 0.7781
    current_specz = 0.778
    starting_z = current_specz # spec-z

    # Force dtype for cython code
    # Apparently this (i.e. for flam_obs and ferr_obs) has  
    # to be done to avoid an obscure error from parallel in joblib --
    # AttributeError: 'numpy.ndarray' object has no attribute 'offset'
    lam_obs = lam_obs.astype(np.float64)
    flam_obs = flam_obs.astype(np.float64)
    ferr_obs = ferr_obs.astype(np.float64)

    # --------------------------------------------- Quality checks ------------------------------------------- #
    # Netsig check
    if netsig_chosen < 10:
        print "Skipping", current_id, "in", current_field, "due to low NetSig:", netsig_chosen
        #continue

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
        #continue

    d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)
    if d4000 < 1.1:
        print "Skipping", current_id, "in", current_field, "due to low D4000:", d4000
        #continue

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
        #continue

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
    avg_dlam = old_ref.get_avg_dlam(lam_obs_grism)

    lam_low_to_insert = np.arange(4000, lam_obs[0], avg_dlam, dtype=np.float64)
    lam_high_to_append = np.arange(lam_obs[-1] + avg_dlam, 16000, avg_dlam, dtype=np.float64)

    resampling_lam_grid = np.insert(lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    print "Resampling wavelength grid:", resampling_lam_grid

    # ------------- Call actual fitting function ------------- #
    zg, zerr_low, zerr_up, min_chi2, age, tau, av = \
    ngp.do_fitting(flam_obs, ferr_obs, lam_obs, broad_lsf, starting_z, resampling_lam_grid, \
        model_lam_grid, total_models, model_comp_spec_withlines, bc03_all_spec_hdulist, start,\
        current_id, current_field, current_specz, current_photz, netsig_chosen, d4000, 0.2)

    # Close HDUs
    bc03_all_spec_hdulist.close()

    sys.exit(0)