from __future__ import division, print_function

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.integrate import simps
from scipy.signal import fftconvolve

import sys
import os
import time
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

figs_data_dir = "/home/bajoshi/models_and_photometry/"
cluster_spz_scripts = "/home/bajoshi/spz_scripts/"
lsfdir = "/home/bajoshi/pears_lsfs/"

# Only for testing with firstlight
# Comment this out before copying code to Agave
# Uncomment above directory paths which are correct for Agave
#home = os.getenv('HOME')
#figs_data_dir = home + '/Desktop/FIGS/'
#cluster_spz_scripts = home + '/Desktop/FIGS/massive-galaxies/cluster_codes/'
#lsfdir = home + '/Desktop/FIGS/new_codes/pears_lsfs/'

sys.path.append(cluster_spz_scripts)
import cluster_do_fitting as cf

matplotlib.rc('text', usetex = True)
matplotlib.rc('font', **{'family' : "sans-serif"})
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
matplotlib.rcParams.update(params)

def get_all_redshifts_v2(current_id, current_field, current_ra, current_dec, current_specz,
    goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam, 
    model_lam_grid_withlines, model_comp_spec_withlines, all_model_flam, total_models, start, 
    log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, 
    ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, sfr_arr,
    get_spz, get_grismz, run_for_full_pears, ignore_irac, ignore_irac_ch3_ch4, chosen_imf):

    print("\n", "Working on:", current_field, current_id, "at", current_specz)

    # Get the correct directory to save results in
    if run_for_full_pears and ('firstlight' in os.uname()[1]):
        spz_outdir = '/Users/baj/Desktop/FIGS/massive-galaxies-figures/full_pears_results/'
    elif (not run_for_full_pears) and ('firstlight' in os.uname()[1]):
        spz_outdir = '/Users/baj/Desktop/FIGS/massive-galaxies/cluster_results/'
    elif run_for_full_pears and ('agave' in os.uname()[1]):
        spz_outdir = '/home/bajoshi/full_pears_results/'
    elif (not run_for_full_pears) and ('agave' in os.uname()[1]):
        spz_outdir = "/home/bajoshi/spz_out/"

    # Check that analysis has not already been done.
    # Move to next galaxy if the fitting result file already exists.
    do_precheck = False
    if do_precheck:
        results_filename = spz_outdir + 'redshift_fitting_results_' + current_field + '_' + str(current_id) + '.txt'

        #t = os.path.getmtime(results_filename)
        #ts = str(dt.datetime.fromtimestamp(t))
        #if ("2019-06-12" in ts) or ("2019-06-13" in ts) or ("2019-06-14" in ts) or ("2019-06-15" in ts) or ("2019-06-16" in ts):
        #    print current_field, current_id, "already done. Moving to next galaxy."
        #    return None

        if os.path.isfile(results_filename):
            print(current_field, current_id, "already done. Moving to next galaxy.")
            return None

    modify_lsf = True

    # ------------------------------- Set field ------------------------------- #
    # Assign catalogs 
    if current_field == 'GOODS-N':
        phot_cat_3dhst = goodsn_phot_cat_3dhst
    elif current_field == 'GOODS-S':
        phot_cat_3dhst = goodss_phot_cat_3dhst

    # ---------------------- Get grism data and covariance matrix and then match with photometry ---------------------- #
    grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = \
    cf.get_data(current_id, current_field)

    if return_code == 0:
        print(current_id, current_field)
        print("Return code should not have been 0. Exiting.")
        sys.exit(0)

    threed_ra = phot_cat_3dhst['ra']
    threed_dec = phot_cat_3dhst['dec']

    # Now match
    ra_lim = 0.3/3600  # arcseconds in degrees
    dec_lim = 0.3/3600
    threed_phot_idx = np.where((threed_ra >= current_ra - ra_lim) & (threed_ra <= current_ra + ra_lim) & \
        (threed_dec >= current_dec - dec_lim) & (threed_dec <= current_dec + dec_lim))[0]

    """
    If there are multiple matches with the photometry catalog 
    within 0.3 arseconds then choose the closest one.
    """
    if len(threed_phot_idx) > 1:
        print("Multiple matches found in photmetry catalog. Choosing the closest one.")

        ra_two = current_ra
        dec_two = current_dec

        dist_list = []
        for v in range(len(threed_phot_idx)):

            ra_one = threed_ra[threed_phot_idx][v]
            dec_one = threed_dec[threed_phot_idx][v]

            dist = np.arccos(np.cos(dec_one*np.pi/180) * \
                np.cos(dec_two*np.pi/180) * np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + \
                np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
            dist_list.append(dist)

        dist_list = np.asarray(dist_list)
        dist_idx = np.argmin(dist_list)
        threed_phot_idx = threed_phot_idx[dist_idx]

    elif len(threed_phot_idx) == 0:
        print("Match not found in Photmetry catalog. Exiting.")
        sys.exit(0)

    # ------------------------------- Get photometric fluxes and their errors ------------------------------- #
    flam_f435w = cf.get_flam('F435W', phot_cat_3dhst['f_F435W'][threed_phot_idx])
    flam_f606w = cf.get_flam('F606W', phot_cat_3dhst['f_F606W'][threed_phot_idx])
    flam_f775w = cf.get_flam('F775W', phot_cat_3dhst['f_F775W'][threed_phot_idx])
    flam_f850lp = cf.get_flam('F850LP', phot_cat_3dhst['f_F850LP'][threed_phot_idx])
    flam_f125w = cf.get_flam('F125W', phot_cat_3dhst['f_F125W'][threed_phot_idx])
    flam_f140w = cf.get_flam('F140W', phot_cat_3dhst['f_F140W'][threed_phot_idx])
    flam_f160w = cf.get_flam('F160W', phot_cat_3dhst['f_F160W'][threed_phot_idx])

    flam_U = cf.get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['f_U'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac1 = cf.get_flam_nonhst('irac1', phot_cat_3dhst['f_IRAC1'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac2 = cf.get_flam_nonhst('irac2', phot_cat_3dhst['f_IRAC2'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac3 = cf.get_flam_nonhst('irac3', phot_cat_3dhst['f_IRAC3'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac4 = cf.get_flam_nonhst('irac4', phot_cat_3dhst['f_IRAC4'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

    ferr_f435w = cf.get_flam('F435W', phot_cat_3dhst['e_F435W'][threed_phot_idx])
    ferr_f606w = cf.get_flam('F606W', phot_cat_3dhst['e_F606W'][threed_phot_idx])
    ferr_f775w = cf.get_flam('F775W', phot_cat_3dhst['e_F775W'][threed_phot_idx])
    ferr_f850lp = cf.get_flam('F850LP', phot_cat_3dhst['e_F850LP'][threed_phot_idx])
    ferr_f125w = cf.get_flam('F125W', phot_cat_3dhst['e_F125W'][threed_phot_idx])
    ferr_f140w = cf.get_flam('F140W', phot_cat_3dhst['e_F140W'][threed_phot_idx])
    ferr_f160w = cf.get_flam('F160W', phot_cat_3dhst['e_F160W'][threed_phot_idx])

    ferr_U = cf.get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['e_U'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac1 = cf.get_flam_nonhst('irac1', phot_cat_3dhst['e_IRAC1'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac2 = cf.get_flam_nonhst('irac2', phot_cat_3dhst['e_IRAC2'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac3 = cf.get_flam_nonhst('irac3', phot_cat_3dhst['e_IRAC3'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac4 = cf.get_flam_nonhst('irac4', phot_cat_3dhst['e_IRAC4'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

    # ------------------------------- Apply aperture correction ------------------------------- #
    # First interpolate the given filter curve on to the wavelength frid of the grism data
    # You only need the F775W filter here since you're only using this filter to get the 
    # aperture correction factor.
    f775w_filt_curve = np.genfromtxt(figs_data_dir + 'filter_curves/f775w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    f775w_trans_interp = griddata(points=f775w_filt_curve['wav'], values=f775w_filt_curve['trans'], \
        xi=grism_lam_obs, method='linear')

    # multiply grism spectrum to filter curve
    num = 0
    den = 0
    for w in range(len(grism_flam_obs)):
        num += grism_flam_obs[w] * f775w_trans_interp[w]
        den += f775w_trans_interp[w]

    avg_f775w_flam_grism = num / den
    aper_corr_factor = flam_f775w / avg_f775w_flam_grism
    print("Aperture correction factor:", "{:.3}".format(aper_corr_factor))

    grism_flam_obs *= aper_corr_factor  # applying factor

    # ------------------------------- Make unified photometry arrays ------------------------------- #
    phot_fluxes_arr = np.array([flam_U, flam_f435w, flam_f606w, flam_f775w, flam_f850lp, flam_f125w, flam_f140w, flam_f160w,
        flam_irac1, flam_irac2, flam_irac3, flam_irac4])
    phot_errors_arr = np.array([ferr_U, ferr_f435w, ferr_f606w, ferr_f775w, ferr_f850lp, ferr_f125w, ferr_f140w, ferr_f160w,
        ferr_irac1, ferr_irac2, ferr_irac3, ferr_irac4])

    # Pivot wavelengths
    # From here --
    # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/
    # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
    # KPNO/MOSAIC U-band: https://www.noao.edu/kpno/mosaic/filters/k1001.html
    # Spitzer IRAC channels: http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/6/#_Toc410728283
    phot_lam = np.array([3582.0, 4328.2, 5921.1, 7692.4, 9033.1, 12486.0, 13923.0, 15369.0, 
    35500.0, 44930.0, 57310.0, 78720.0])  # angstroms

    """
    Testing block: do not delete

    print phot_lam
    print phot_fluxes_arr
    print phot_errors_arr

    print grism_lam_obs
    print grism_flam_obs
    print grism_ferr_obs

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(grism_lam_obs, grism_flam_obs, 'o-', color='k', markersize=2)
    ax.fill_between(grism_lam_obs, grism_flam_obs + grism_ferr_obs, grism_flam_obs - grism_ferr_obs, color='lightgray')
    ax.errorbar(phot_lam, phot_fluxes_arr, yerr=phot_errors_arr, \
        fmt='.', color='firebrick', markeredgecolor='firebrick', \
        capsize=2, markersize=10.0, elinewidth=2.0)

    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()
    return None
    """

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
        print("LSF not found. Moving to next galaxy.")
        return None

    # -------- Stetch the LSF ------- #
    if modify_lsf:
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

        lsf_to_use = stretched_lsf

    # ------- Make new resampling grid ------- # 
    # extend lam_grid to be able to move the lam_grid later 
    avg_dlam = cf.get_avg_dlam(grism_lam_obs)

    lam_low_to_insert = np.arange(6000, grism_lam_obs[0], avg_dlam, dtype=np.float64)
    lam_high_to_append = np.arange(grism_lam_obs[-1] + avg_dlam, 10000, avg_dlam, dtype=np.float64)

    resampling_lam_grid = np.insert(grism_lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    # ------- Finite photometry values ------- # 
    # Make sure that the photometry arrays all have finite values
    # If any vlues are NaN then throw them out
    phot_fluxes_finite_idx = np.where(np.isfinite(phot_fluxes_arr))[0]
    phot_errors_finite_idx = np.where(np.isfinite(phot_errors_arr))[0]

    phot_fin_idx = reduce(np.intersect1d, (phot_fluxes_finite_idx, phot_errors_finite_idx))
    # np.intersect1d() is correct. 
    # This is because you are selecting only the finite values that are in BOTH arrays.

    if run_for_full_pears:  # i.e., only mess with selection of IRAC photometry and IMF when running for full PEARS sample
        if ignore_irac_ch3_ch4 and ignore_irac:
            # i.e., last FOUR wavebands are to be ignored ALWAYS in this case!
            phot_fin_idx = np.setdiff1d(phot_fin_idx, np.array([8, 9, 10, 11]))

            # Now fix path
            spz_outdir = spz_outdir.replace('full_pears_results', 'full_pears_results_no_irac')
            if chosen_imf == 'Chabrier':
                spz_outdir = spz_outdir.replace('full_pears_results_no_irac', 'full_pears_results_chabrier_no_irac')

        elif ignore_irac_ch3_ch4 and (not ignore_irac):
            # i.e., last TWO wavebands are to be ignored ALWAYS in this case!
            phot_fin_idx = np.setdiff1d(phot_fin_idx, np.array([10, 11]))

            # Now fix path
            spz_outdir = spz_outdir.replace('full_pears_results', 'full_pears_results_no_irac_ch3_ch4')
            if chosen_imf == 'Chabrier':
                spz_outdir = spz_outdir.replace('full_pears_results_no_irac_ch3_ch4', 'full_pears_results_chabrier_no_irac_ch3_ch4')

        elif (not ignore_irac_ch3_ch4) and (not ignore_irac):
            # most common case where Spitzer wavebands are NOT ignored
            if chosen_imf == 'Chabrier':
                spz_outdir = spz_outdir.replace('full_pears_results', 'full_pears_results_chabrier')
        #else:
        #    print("Unrecognized option for ignoring IRAC photometry." )
        #    print("Check given IRAC photometry options. Exiting.")
        #    sys.exit(1)

    phot_fluxes_arr = phot_fluxes_arr[phot_fin_idx]
    phot_errors_arr = phot_errors_arr[phot_fin_idx]
    phot_lam = phot_lam[phot_fin_idx]

    # Get covariance matrix
    # Has to be ddone after tge step that could 
    # cut down the photometric data array.
    #covmat = cf.get_covmat(combined_lam_obs, combined_flam_obs, combined_ferr_obs, silent=False)
    # Dummy covaraiance matrix used for now since we haven't yet gotten to combined_* arrays.
    covmat = np.identity(len(grism_lam_obs) + len(phot_lam))

    # ------------- Call fitting function for photo-z ------------- #
    print("Computing photo-z now.")
    
    zp_minchi2, zp, zp_zerr_low, zp_zerr_up, zp_min_chi2, zp_bestalpha, \
    zp_template_ms, zp_ms, zp_sfr, zp_uv, zp_vj, zp_model_idx, zp_age, zp_tau, zp_av = \
    cf.do_photoz_fitting_lookup(phot_fluxes_arr, phot_errors_arr, phot_lam, \
        model_lam_grid_withlines, total_models, model_comp_spec_withlines, start,\
        current_id, current_field, all_model_flam, phot_fin_idx, current_specz, spz_outdir, \
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, \
        bv_col_arr, vj_col_arr, ms_arr, mgal_arr, sfr_arr, run_for_full_pears)

    # ------------- Call fitting function for SPZ ------------- #
    if get_spz:
        print("\n", "Photo-z done. Moving on to SPZ computation now.")
    
        zspz_minchi2, zspz, zspz_zerr_low, zspz_zerr_up, zspz_min_chi2, zspz_bestalpha, \
        zspz_template_ms, zspz_ms, zspz_sfr, zspz_uv, zspz_vj, zspz_model_idx, zspz_age, zspz_tau, zspz_av = \
        cf.do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_fluxes_arr, phot_errors_arr, phot_lam, covmat, \
            lsf_to_use, resampling_lam_grid, len(resampling_lam_grid), all_model_flam, phot_fin_idx, \
            model_lam_grid_withlines, total_models, model_comp_spec_withlines, start, current_id, current_field, current_specz, zp_minchi2, \
            log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, sfr_arr, \
            run_for_full_pears, spz_outdir, use_broadband=True, single_galaxy=False, for_loop_method='sequential')
    
    # ------------- Call fitting function for grism-z ------------- #
    # Essentially just calls the same function as above but switches off broadband for the fit
    if get_grismz:
        print("\n", "SPZ done. Moving on to grism-z computation now.")
            
        zg_minchi2, zg, zg_zerr_low, zg_zerr_up, zg_min_chi2, zg_bestalpha, \
        zg_template_ms, zg_ms, zg_sfr, zg_uv, zg_vj, zg_model_idx, zg_age, zg_tau, zg_av = \
        cf.do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_fluxes_arr, phot_errors_arr, phot_lam, covmat, \
            lsf_to_use, resampling_lam_grid, len(resampling_lam_grid), all_model_flam, phot_fin_idx, \
            model_lam_grid_withlines, total_models, model_comp_spec_withlines, start, current_id, current_field, current_specz, zp_minchi2, \
            log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, sfr_arr, \
            run_for_full_pears, spz_outdir, use_broadband=False, single_galaxy=False, for_loop_method='sequential')

    print("All redshifts computed for:", current_field, current_id, "    Will save results now.")

    # ------------------------------ Save all fitting results to text file ------------------------------ #
    with open(spz_outdir + 'redshift_fitting_results_' + current_field + '_' + str(current_id) + '.txt', 'w') as fh:

        # Write header first
        hdr_line1 = "# All redshift fitting results from pipeline for galaxy: " + current_field + " " + str(current_id)
        hdr_line2 = "#  PearsID  Field  RA  DEC  zspec  zp_minchi2  zspz_minchi2  zg_minchi2" + \
        "  zp  zspz  zg  zp_zerr_low  zp_zerr_up  zspz_zerr_low  zspz_zerr_up  zg_zerr_low  zg_zerr_up" + \
        "  zp_min_chi2  zspz_min_chi2  zg_min_chi2  zp_bestalpha  zspz_bestalpha  zg_bestalpha" + \
        "  zp_model_idx  zspz_model_idx  zg_model_idx  zp_age  zp_tau  zp_av" + \
        "  zspz_age  zspz_tau  zspz_av  zg_age  zg_tau  zg_av" + \
        "  zp_template_ms  zp_ms  zp_sfr  zp_uv  zp_vj" + \
        "  zspz_template_ms  zspz_ms  zspz_sfr  zspz_uv  zspz_vj" + \
        "  zg_template_ms  zg_ms  zg_sfr  zg_uv  zg_vj"

        fh.write(hdr_line1 + '\n')
        fh.write(hdr_line2 + '\n')

        # Now write actual fitting results
        ra_to_write = "{:.7f}".format(current_ra)
        dec_to_write = "{:.6f}".format(current_dec)
        zspec_to_write = "{:.3f}".format(current_specz)

        str_to_write1 = str(current_id) + "  " + current_field + "  " + \
        ra_to_write + "  " + dec_to_write + "  " + zspec_to_write + "  "
        str_to_write8 = "{:.2e}".format(zp_age) + "  " + "{:.2e}".format(zp_tau) + "  " + "{:.2f}".format(zp_av) + "  "
        str_to_write11 = "{:.5f}".format(zp_template_ms) + "  " + "{:.4e}".format(zp_ms) + "  " + \
        "{:.2e}".format(zp_sfr) + "  " + "{:.4f}".format(zp_uv) + "  " + "{:.4f}".format(zp_vj) + "  "
        
        # Now write the results depending on what redshifts were computed.
        # The photometric redshift is always computed, while there are
        # boolean flags for getting the SPZ and grism-z.
        # In principle, there should be the following 4 check cases:
        # if get_spz and get_grismz:
        # elif get_spz and (not get_grismz):
        # elif (not get_spz) and get_grismz:
        # elif (not get_spz) and (not get_grismz):

        # But because I never compute the grism-z without having the SPZ,
        # i.e., the case of elif (not get_spz) and get_grismz: will not happen,
        # I do not include it in hte following condition checks.

        if get_spz and get_grismz:  # All redshifts computed

            str_to_write2 = "{:.2f}".format(zp_minchi2) + "  " + "{:.2f}".format(zspz_minchi2) + "  " + "{:.2f}".format(zg_minchi2) + "  "
            str_to_write3 = "{:.2f}".format(zp) + "  " + "{:.2f}".format(zspz) + "  " + "{:.2f}".format(zg) + "  "
            str_to_write4 = "{:.2f}".format(zp_zerr_low) + "  " + "{:.2f}".format(zp_zerr_up) + "  " + \
            "{:.2f}".format(zspz_zerr_low) + "  " + "{:.2f}".format(zspz_zerr_up) + "  " + \
            "{:.2f}".format(zg_zerr_low) + "  " + "{:.2f}".format(zg_zerr_up) + "  "
            str_to_write5 = "{:.2f}".format(zp_min_chi2) + "  " + "{:.2f}".format(zspz_min_chi2) + "  " + "{:.2f}".format(zg_min_chi2) + "  "
            str_to_write6 = "{:.2e}".format(zp_bestalpha) + "  " + "{:.2e}".format(zspz_bestalpha) + "  " + "{:.2e}".format(zg_bestalpha) + "  "
            str_to_write7 = str(int(zp_model_idx)) + "  " + str(int(zspz_model_idx)) + "  " + str(int(zg_model_idx)) + "  "
            str_to_write9 = "{:.2e}".format(zspz_age) + "  " + "{:.2e}".format(zspz_tau) + "  " + "{:.2f}".format(zspz_av) + "  "
            str_to_write10 = "{:.2e}".format(zg_age) + "  " + "{:.2e}".format(zg_tau) + "  " + "{:.2f}".format(zg_av) + "  "
            str_to_write12 = "{:.5f}".format(zspz_template_ms) + "  " + "{:.4e}".format(zspz_ms) + "  " + \
            "{:.2f}".format(zspz_sfr) + "  " + "{:.4f}".format(zspz_uv) + "  " + "{:.4f}".format(zspz_vj) + "  "
            str_to_write13 = "{:.5f}".format(zg_template_ms) + "  " + "{:.4e}".format(zg_ms) + "  " + \
            "{:.2f}".format(zg_sfr) + "  " + "{:.4f}".format(zg_uv) + "  " + "{:.4f}".format(zg_vj) + "  "

        elif get_spz and (not get_grismz):  # Only photo-z and SPZ computed

            str_to_write2 = "{:.2f}".format(zp_minchi2) + "  " + "{:.2f}".format(zspz_minchi2) + "  " + "-99.0" + "  "
            str_to_write3 = "{:.2f}".format(zp) + "  " + "{:.2f}".format(zspz) + "  " + "-99.0" + "  "
            str_to_write4 = "{:.2f}".format(zp_zerr_low) + "  " + "{:.2f}".format(zp_zerr_up) + "  " + \
            "{:.2f}".format(zspz_zerr_low) + "  " + "{:.2f}".format(zspz_zerr_up) + "  " + \
            "-99.0" + "  " + "-99.0" + "  "
            str_to_write5 = "{:.2f}".format(zp_min_chi2) + "  " + "{:.2f}".format(zspz_min_chi2) + "  " + "-99.0" + "  "
            str_to_write6 = "{:.2e}".format(zp_bestalpha) + "  " + "{:.2e}".format(zspz_bestalpha) + "  " + "-99.0" + "  "
            str_to_write7 = str(int(zp_model_idx)) + "  " + str(int(zspz_model_idx)) + "  " + str(-99.0) + "  "
            str_to_write9 = "{:.2e}".format(zspz_age) + "  " + "{:.2e}".format(zspz_tau) + "  " + "{:.2f}".format(zspz_av) + "  "
            str_to_write10 = "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  "
            str_to_write12 = "{:.5f}".format(zspz_template_ms) + "  " + "{:.4e}".format(zspz_ms) + "  " + \
            "{:.2f}".format(zspz_sfr) + "  " + "{:.4f}".format(zspz_uv) + "  " + "{:.4f}".format(zspz_vj) + "  "
            str_to_write13 = "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  "

        elif (not get_spz) and (not get_grismz):  # Only photo-z computed

            str_to_write2 = "{:.2f}".format(zp_minchi2) + "  " + "-99.0" + "  " + "-99.0" + "  "
            str_to_write3 = "{:.2f}".format(zp) + "  " + "-99.0" + "  " + "-99.0" + "  "
            str_to_write4 = "{:.2f}".format(zp_zerr_low) + "  " + "{:.2f}".format(zp_zerr_up) + "  " + \
            "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  "
            str_to_write5 = "{:.2f}".format(zp_min_chi2) + "  " + "-99.0" + "  " + "-99.0" + "  "
            str_to_write6 = "{:.2e}".format(zp_bestalpha) + "  " + "-99.0" + "  " + "-99.0" + "  "
            str_to_write7 = str(int(zp_model_idx)) + "  " + str(-99.0) + "  " + str(-99.0) + "  "
            str_to_write9 = "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  "
            str_to_write10 = "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  "
            str_to_write12 = "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  "
            str_to_write13 = "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  " + "-99.0" + "  "

        # Combine hte above strings and write
        fh.write(str_to_write1 + str_to_write2 + str_to_write3 + str_to_write4 + str_to_write5 + \
            str_to_write6 + str_to_write7 + str_to_write8 + str_to_write9 + str_to_write10 + \
            str_to_write11 + str_to_write12 + str_to_write13)

    print("Results saved for:", current_field, current_id)

    # ------------------------------ Plots ------------------------------ #
    check_plot = False
    if check_plot:

        print("\n" + "Working on plotting the results." + "\n")

        # Read in the results 
        # This requires that the results be saved 
        results_filename = spz_outdir + 'redshift_fitting_results_' + current_field + '_' + str(current_id) + '.txt'

        # Read in the required stuff
        current_result = np.genfromtxt(results_filename, dtype=None, names=True, skip_header=1)
        # 
        zp_model_idx = int(current_result['zp_model_idx'])
        zp_minchi2 = current_result['zp_minchi2']
        zp_bestalpha = current_result['zp_bestalpha']
        zp_zerr_low = current_result['zp_zerr_low']
        zp_zerr_up = current_result['zp_zerr_up']
        zp_min_chi2 = current_result['zp_min_chi2']
        zp_age = current_result['zp_age']
        zp_tau = current_result['zp_tau']
        zp_av = current_result['zp_av']
        # 
        zspz_model_idx = int(current_result['zspz_model_idx'])
        zspz_minchi2 = current_result['zspz_minchi2']
        zspz_bestalpha = current_result['zspz_bestalpha']
        zspz_zerr_low = current_result['zspz_zerr_low']
        zspz_zerr_up = current_result['zspz_zerr_up']
        zspz_min_chi2 = current_result['zspz_min_chi2']
        zspz_age = current_result['zspz_age']
        zspz_tau = current_result['zspz_tau']
        zspz_av = current_result['zspz_av']

        # ------------------------------- Get best fit model for plotting ------------------------------- #
        # Will have to do this at the photo-z and SPZ separtely otherwise the plots will not look right
        # ------------ Get best fit model for photo-z ------------ #
        zp_best_fit_model_fullres = model_comp_spec_withlines[zp_model_idx]
        zp_all_filt_flam_bestmodel = get_photometry_best_fit_model(zp_minchi2, zp_model_idx, phot_fin_idx, all_model_flam, total_models)

        # ------------------------------- Plotting ------------------------------- #
        plot_photoz_fit(phot_lam, phot_fluxes_arr, phot_errors_arr, model_lam_grid_withlines, \
        zp_best_fit_model_fullres, zp_all_filt_flam_bestmodel, zp_bestalpha, \
        current_id, current_field, current_specz, zp_minchi2, zp_zerr_low, zp_zerr_up, zp_min_chi2, \
        zp_age, zp_tau, zp_av, netsig_chosen, spz_outdir)

        """
        # ------------ Get best fit model for SPZ ------------ #
        zspz_best_fit_model_in_objlamgrid, zspz_all_filt_flam_bestmodel, zspz_best_fit_model_fullres = \
        get_best_fit_model_spz(resampling_lam_grid, len(resampling_lam_grid), model_lam_grid_withlines, model_comp_spec_withlines, \
            grism_lam_obs, zspz_minchi2, zspz_model_idx, phot_fin_idx, all_model_flam, lsf_to_use, total_models)

        # ------------------------------- Plotting ------------------------------- #
        plot_spz_fit(grism_lam_obs, grism_flam_obs, grism_ferr_obs, phot_lam, phot_fluxes_arr, phot_errors_arr, \
        model_lam_grid_withlines, zspz_best_fit_model_fullres, zspz_best_fit_model_in_objlamgrid, zspz_all_filt_flam_bestmodel, zspz_bestalpha, \
        current_id, current_field, current_specz, zp_zerr_low, zp_zerr_up, zp_minchi2, zspz_zerr_low, zspz_zerr_up, zspz_minchi2, \
        zspz_min_chi2, zspz_age, zspz_tau, zspz_av, netsig_chosen, spz_outdir)
        """

    return None

def makefig():
    # ---------- Create figure ---------- #
    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[:8,:])
    ax2 = fig.add_subplot(gs[8:,:])

    # ---------- labels ---------- #
    ax1.set_ylabel(r'$\mathrm{f_\lambda\ [erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]}$', fontsize=14)
    ax2.set_xlabel(r'$\mathrm{Wavelength\, [\AA]}$', fontsize=15)
    ax2.set_ylabel(r'$\mathrm{\frac{f^{obs}_\lambda\ - \alpha f^{mod}_\lambda}{\sigma_{f_{\lambda}^{\rm obs}}}}$', fontsize=17)

    return fig, ax1, ax2

def make_pz_labels(ax):

    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('p(z)', fontsize=12)

    ax.xaxis.set_label_coords(0.8,-0.05)

    return ax

def get_photometry_best_fit_model(redshift, model_idx, phot_fin_idx, all_model_flam, total_models):
    # All you need from here is the photometry for the best fit model
    # ------------ Get photomtery for model ------------- #
    # The model mags were computed on a finer redshift grid
    # So make sure to get the z_idx correct
    z_model_arr = np.arange(0.005, 6.005, 0.005)

    z_idx = np.where(z_model_arr == redshift)[0]

    # and because for some reason it does not find matches 
    # in the model redshift array (sometimes), I need this check here.
    if not z_idx.size:
        z_idx = np.argmin(abs(z_model_arr - redshift))

    all_filt_flam_model = all_model_flam[:, z_idx, :]
    all_filt_flam_model = all_filt_flam_model[phot_fin_idx, :]
    all_filt_flam_model = all_filt_flam_model.reshape(len(phot_fin_idx), total_models)

    all_filt_flam_model_t = all_filt_flam_model.T
    all_filt_flam_bestmodel = all_filt_flam_model_t[model_idx]

    return all_filt_flam_bestmodel

def get_best_fit_model_spz(resampling_lam_grid, resampling_lam_grid_length, model_lam_grid, model_comp_spec, \
    grism_lam_obs, redshift, model_idx, phot_fin_idx, all_model_flam, lsf, total_models):

    # ------------ Get best fit model at grism resolution ------------ #
    # First do the convolution with the LSF
    model_comp_spec_lsfconv = np.zeros(model_comp_spec.shape)
    for i in range(total_models):
        model_comp_spec_lsfconv[i] = fftconvolve(model_comp_spec[i], lsf, mode = 'same')

    # chop model to get the part within objects lam obs grid
    model_lam_grid_indx_low = np.argmin(abs(resampling_lam_grid - grism_lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(abs(resampling_lam_grid - grism_lam_obs[-1]))

    # Will have to redo the model modifications at the new found redshift
    model_comp_spec_modified = \
    cf.redshift_and_resample(model_comp_spec_lsfconv, redshift, total_models, \
        model_lam_grid, resampling_lam_grid, resampling_lam_grid_length)
    print("Model mods done (only for plotting purposes) at the new SPZ:", redshift)

    best_fit_model_in_objlamgrid = model_comp_spec_modified[model_idx, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # ------------ Get photomtery for model ------------- #
    all_filt_flam_bestmodel = get_photometry_best_fit_model(redshift, model_idx, phot_fin_idx, all_model_flam, total_models)

    # ------------ Get best fit model at full resolution ------------ #
    best_fit_model_fullres = model_comp_spec[model_idx]

    return best_fit_model_in_objlamgrid, all_filt_flam_bestmodel, best_fit_model_fullres

def plot_photoz_fit(phot_lam_obs, phot_flam_obs, phot_ferr_obs, model_lam_grid, \
    best_fit_model_fullres, all_filt_flam_bestmodel, bestalpha, \
    obj_id, obj_field, specz, zp, low_z_lim, upper_z_lim, chi2, age, tau, av, netsig, savedir):

    # Make figure and place on grid
    fig, ax1, ax2 = makefig()

    # ---------- plot data, model, and residual ---------- #
    # Plot full res model but you'll have to redshift it
    dl = cf.get_lum_dist(zp)  # in Mpc
    dl = dl * 3.086e24  # convert Mpc to cm
    ax1.plot(model_lam_grid * (1+zp), bestalpha * best_fit_model_fullres / (4 * np.pi * dl * dl * (1+zp)), color='dimgrey', alpha=0.2)

    # Plot model photometry
    # The flux here does not need to be redshifted
    # The compute_flam code returns the filter fluxes in f_lambda units
    ax1.scatter(phot_lam_obs, bestalpha * all_filt_flam_bestmodel, s=20, color='lightseagreen', zorder=20)

    # ----- plot data
    ax1.errorbar(phot_lam_obs, phot_flam_obs, yerr=phot_ferr_obs, fmt='.', color='crimson', markeredgecolor='crimson', \
        capsize=2, markersize=10.0, elinewidth=2.0)

    # ----- Residuals
    # For the photometry
    resid_fit_phot = (phot_flam_obs - bestalpha * all_filt_flam_bestmodel) / phot_ferr_obs
    ax2.scatter(phot_lam_obs, resid_fit_phot, s=4, color='k')
    ax2.axhline(y=0, ls='--', color='k')

    # ---------- limits ---------- #
    max_y_obs = np.max(phot_flam_obs)
    min_y_obs = np.min(phot_flam_obs)

    max_ylim = 1.25 * max_y_obs
    min_ylim = 0.2 * min_y_obs

    ax1.set_ylim(min_ylim, max_ylim)

    ax1.set_xlim(3000, 85000)
    ax2.set_xlim(3000, 85000)

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # ---------- tick labels for the logarithmic axis ---------- #
    ax2.set_xticks([4000, 10000, 20000, 50000, 80000])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # ---------- minor ticks ---------- #
    ax1.minorticks_on()
    ax2.minorticks_on()

    # ---------- text for info ---------- #
    ax1.text(0.71, 0.61, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    low_zerr = zp - low_z_lim
    high_zerr = upper_z_lim - zp

    zp_string = r'$\mathrm{z_{p;best} = }$' + "{:.4}".format(zp) + \
    r'$\substack{+$' + "{:.3}".format(high_zerr) + r'$\\ -$' + "{:.3}".format(low_zerr) + r'$}$'
    ax1.text(0.71, 0.55, zp_string, \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.71, 0.48, r'$\mathrm{z_{spec} = }$' + "{:.4}".format(specz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.71, 0.43, r'$\mathrm{\chi^2 = }$' + "{:.3}".format(chi2), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.71, 0.36, r'$\mathrm{NetSig = }$' + cf.convert_to_sci_not(netsig), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=8)


    ax1.text(0.37, 0.18,'log(Age[yr]) = ' + "{:.4}".format(age), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.37, 0.12, r'$\tau$' + '[Gyr] = ' + "{:.3}".format(tau), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    if av < 0:
        av = -99.0

    ax1.text(0.37, 0.06, r'$\mathrm{A_V}$' + ' = ' + "{:.3}".format(av), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    # ---------- Plot p(z) curve in an inset figure ---------- #
    # Solution for inset came from SO:
    # https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.62, 0.72, 0.3, 0.2]
    ax3 = fig.add_axes([left, bottom, width, height])

    # Read in p(z) curve. It should be in the same folder where all these figures are being saved.
    pz = np.load(savedir + obj_field + '_' + str(obj_id) + '_photoz_pz.npy')
    zarr = np.load(savedir + obj_field + '_' + str(obj_id) + '_photoz_z_arr.npy')

    ax3 = make_pz_labels(ax3)

    ax3.plot(zarr, pz)
    if specz != -99.0:
        ax3.axvline(x=specz, ls='--', color='darkred')
    ax3.minorticks_on()

    # ---------- Save figure ---------- #
    fig.savefig(savedir + obj_field + '_' + str(obj_id) + '_photoz_fit.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

def plot_spz_fit(grism_lam_obs, grism_flam_obs, grism_ferr_obs, phot_lam_obs, phot_flam_obs, phot_ferr_obs, \
    model_lam_grid, best_fit_model_fullres, best_fit_model_in_objlamgrid, all_filt_flam_bestmodel, bestalpha, \
    obj_id, obj_field, specz, low_zp_lim, upper_zp_lim, zp, low_zspz_lim, upper_zspz_lim, zspz, \
    chi2, age, tau, av, netsig, savedir):

    # Make figure and place on grid
    fig, ax1, ax2 = makefig()

    # ---------- plot data, model, and residual ---------- #
    # plot full res model but you'll have to redshift it
    dl = cf.get_lum_dist(zspz)  # in Mpc
    dl = dl * 3.086e24  # convert Mpc to cm
    ax1.plot(model_lam_grid * (1+zspz), bestalpha * best_fit_model_fullres / (4 * np.pi * dl * dl * (1+zspz)), color='dimgrey', alpha=0.2)

    # ----- plot data
    ax1.plot(grism_lam_obs, grism_flam_obs, 'o-', color='k', markersize=2, lw=2, zorder=10)
    ax1.fill_between(grism_lam_obs, grism_flam_obs + grism_ferr_obs, grism_flam_obs - grism_ferr_obs, color='gray', zorder=10)

    ax1.errorbar(phot_lam_obs, phot_flam_obs, yerr=phot_ferr_obs, fmt='.', color='crimson', markeredgecolor='crimson', \
        capsize=2, markersize=10.0, elinewidth=2.0)

    # ----- plot best fit model
    ax1.plot(grism_lam_obs, bestalpha * best_fit_model_in_objlamgrid, ls='-', lw=1.2, color='lightseagreen', zorder=20)
    ax1.scatter(phot_lam_obs, bestalpha * all_filt_flam_bestmodel, s=20, color='lightseagreen', zorder=20)

    # ----- Residuals
    # For the grism points
    resid_fit_grism = (grism_flam_obs - bestalpha * best_fit_model_in_objlamgrid) / grism_ferr_obs

    # Now plot
    ax2.scatter(grism_lam_obs, resid_fit_grism, s=4, color='k')
    ax2.axhline(y=0, ls='--', color='k')

    # For the photometry
    resid_fit_phot = (phot_flam_obs - bestalpha * all_filt_flam_bestmodel) / phot_ferr_obs
    ax2.scatter(phot_lam_obs, resid_fit_phot, s=4, color='k')

    # ---------- limits ---------- #
    max_y_obs = np.max(np.concatenate((grism_flam_obs, phot_flam_obs)))
    min_y_obs = np.min(np.concatenate((grism_flam_obs, phot_flam_obs)))

    max_ylim = 1.25 * max_y_obs
    min_ylim = 0.2 * min_y_obs

    ax1.set_ylim(min_ylim, max_ylim)
    ax2.set_ylim(-14, 14)

    ax1.set_xlim(3000, 85000)
    ax2.set_xlim(3000, 85000)

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # ---------- tick labels for the logarithmic axis ---------- #
    ax2.set_xticks([4000, 10000, 20000, 50000, 80000])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # ---------- minor ticks ---------- #
    ax1.minorticks_on()
    ax2.minorticks_on()

    # ---------- text for info ---------- #
    ax1.text(0.71, 0.61, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    low_zspz_err = zspz - low_zspz_lim
    high_zspz_err = upper_zspz_lim - zspz

    low_zp_err = zp - low_zp_lim
    high_zp_err = upper_zp_lim - zp

    ax1.text(0.71, 0.55, \
    r'$\mathrm{z_{spz;best} = }$' + "{:.3}".format(zspz) + \
    r'$\substack{+$' + "{:.2f}".format(high_zspz_err) + r'$\\ -$' + "{:.2f}".format(low_zspz_err) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.71, 0.48, r'$\mathrm{z_{spec} = }$' + "{:.3f}".format(specz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.71, 0.42, \
    r'$\mathrm{z_{p;best} = }$' + "{:.3}".format(zp) + \
    r'$\substack{+$' + "{:.2f}".format(high_zp_err) + r'$\\ -$' + "{:.2f}".format(low_zp_err) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.71, 0.36, r'$\mathrm{\chi^2 = }$' + "{:.3}".format(chi2), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.71, 0.3, r'$\mathrm{NetSig = }$' + cf.convert_to_sci_not(netsig), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.17, 0.18,'log(Age[yr]) = ' + "{:.4}".format(age), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.17, 0.12, r'$\tau$' + '[Gyr] = ' + "{:.3}".format(tau), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    if av < 0:
        av = -99.0

    ax1.text(0.17, 0.06, r'$\mathrm{A_V}$' + ' = ' + "{:.3}".format(av), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    # ---------- Plot p(z) curve in an inset figure ---------- #
    # Solution for inset came from SO:
    # https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.62, 0.72, 0.3, 0.2]
    ax3 = fig.add_axes([left, bottom, width, height])

    # Read in p(z) curve. It should be in the same folder where all these figures are being saved.
    pz = np.load(savedir + obj_field + '_' + str(obj_id) + '_spz_pz.npy')
    zarr = np.load(savedir + obj_field + '_' + str(obj_id) + '_spz_z_arr.npy')

    ax3 = make_pz_labels(ax3)

    ax3.plot(zarr, pz)
    if specz != -99.0:
        ax3.axvline(x=specz, ls='--', color='darkred')
    ax3.minorticks_on()

    # ---------- Save figure ---------- #
    fig.savefig(savedir + obj_field + '_' + str(obj_id) + '_spz_fit.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None



