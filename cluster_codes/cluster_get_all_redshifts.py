from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.integrate import simps

import sys
import os
import time
import datetime as dt

figs_data_dir = "/home/bajoshi/models_and_photometry/"
cluster_spz_scripts = "/home/bajoshi/spz_scripts/"
spz_outdir = "/home/bajoshi/spz_out/"
lsfdir = "/home/bajoshi/pears_lsfs/"

# Only for testing with firstlight
# Comment this out before copying code to Agave
# Uncomment above directory paths which are correct for Agave
#figs_data_dir = '/Users/baj/Desktop/FIGS/'
#cluster_spz_scripts = '/Users/baj/Desktop/FIGS/massive-galaxies/cluster_codes/'
#spz_outdir = '/Users/baj/Desktop/FIGS/massive-galaxies/cluster_results/'
#lsfdir = '/Users/baj/Desktop/FIGS/new_codes/pears_lsfs/'

sys.path.append(cluster_spz_scripts)
import cluster_do_fitting as cf

def get_all_redshifts_v2(current_id, current_field, current_ra, current_dec, current_specz,\
    goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam, \
    model_lam_grid_withlines, model_comp_spec_withlines, all_model_flam, total_models, start, \
    log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, \
    get_spz, get_grismz, run_for_full_pears):

    print "\n", "Working on:", current_field, current_id, "at", current_specz

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
            print current_field, current_id, "already done. Moving to next galaxy."
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
        print current_id, current_field
        print "Return code should not have been 0. Exiting."
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
        print "Multiple matches found in photmetry catalog. Choosing the closest one."

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
        print "Match not found in Photmetry catalog. Exiting."
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
    print "Aperture correction factor:", "{:.3}".format(aper_corr_factor)

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
        print "LSF not found. Moving to next galaxy."
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
    print "Computing photo-z now."
    
    zp_minchi2, zp, zp_zerr_low, zp_zerr_up, zp_min_chi2, zp_bestalpha, zp_model_idx, zp_age, zp_tau, zp_av = \
    cf.do_photoz_fitting_lookup(phot_fluxes_arr, phot_errors_arr, phot_lam, \
        model_lam_grid_withlines, total_models, model_comp_spec_withlines, start,\
        current_id, current_field, all_model_flam, phot_fin_idx, current_specz, spz_outdir, \
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, \
        bv_col_arr, vj_col_arr, ms_arr, mgal_arr, run_for_full_pears)

    # ------------- Call fitting function for SPZ ------------- #
    if get_spz:
        print "\n", "Photo-z done. Moving on to SPZ computation now."
    
        zspz_minchi2, zspz, zspz_zerr_low, zspz_zerr_up, zspz_min_chi2, \
        zspz_bestalpha, zspz_model_idx, zspz_age, zspz_tau, zspz_av = \
        cf.do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_fluxes_arr, phot_errors_arr, phot_lam, covmat, \
            lsf_to_use, resampling_lam_grid, len(resampling_lam_grid), all_model_flam, phot_fin_idx, \
            model_lam_grid_withlines, total_models, model_comp_spec_withlines, start, current_id, current_field, current_specz, zp, \
            log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, \
            run_for_full_pears, use_broadband=True, single_galaxy=False, for_loop_method='sequential')
    
    # ------------- Call fitting function for grism-z ------------- #
    # Essentially just calls the same function as above but switches off broadband for the fit
    if get_grismz:
        print "\n", "SPZ done. Moving on to grism-z computation now."
            
        zg_minchi2, zg, zg_zerr_low, zg_zerr_up, zg_min_chi2, zg_bestalpha, zg_model_idx, zg_age, zg_tau, zg_av = \
        cf.do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_fluxes_arr, phot_errors_arr, phot_lam, covmat, \
            lsf_to_use, resampling_lam_grid, len(resampling_lam_grid), all_model_flam, phot_fin_idx, \
            model_lam_grid_withlines, total_models, model_comp_spec_withlines, start, current_id, current_field, current_specz, zp, \
            log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, \
            run_for_full_pears, use_broadband=False, single_galaxy=False, for_loop_method='sequential')

    print "All redshifts computed for:", current_field, current_id, "    Will save results now."

    # ------------------------------ Save all fitting results to text file ------------------------------ #
    with open(spz_outdir + 'redshift_fitting_results_' + current_field + '_' + str(current_id) + '.txt', 'w') as fh:

        # Write header first
        hdr_line1 = "# All redshift fitting results from pipeline for galaxy: " + current_field + " " + str(current_id)
        hdr_line2 = "#  PearsID  Field  RA  DEC  zspec  zp_minchi2  zspz_minchi2  zg_minchi2" + \
        "  zp  zspz  zg  zp_zerr_low  zp_zerr_up  zspz_zerr_low  zspz_zerr_up  zg_zerr_low  zg_zerr_up" + \
        "  zp_min_chi2  zspz_min_chi2  zg_min_chi2  zp_bestalpha  zspz_bestalpha  zg_bestalpha" + \
        "  zp_model_idx  zspz_model_idx  zg_model_idx  zp_age  zp_tau  zp_av" + \
        "  zspz_age  zspz_tau  zspz_av  zg_age  zg_tau  zg_av"

        fh.write(hdr_line1 + '\n')
        fh.write(hdr_line2 + '\n')

        # Now write actual fitting results
        ra_to_write = "{:.7f}".format(current_ra)
        dec_to_write = "{:.6f}".format(current_dec)
        zspec_to_write = "{:.3f}".format(current_specz)

        str_to_write1 = str(current_id) + "  " + current_field + "  " + ra_to_write + "  " + dec_to_write + "  " + zspec_to_write + "  "
        str_to_write8 = "{:.2e}".format(zp_age) + "  " + "{:.2e}".format(zp_tau) + "  " + "{:.2f}".format(zp_av) + "  "

        """
        Now write the results depending on what redshifts were computed.
        The photometric redshift is always computed, while there are
        boolean flags for getting the SPZ and grism-z.
        In principle, there should be the following 4 check cases:
        if get_spz and get_grismz:
        elif get_spz and (not get_grismz):
        elif (not get_spz) and get_grismz:
        elif (not get_spz) and (not get_grismz):

        But because I never compute the grism-z without having the SPZ,
        i.e., the case of elif (not get_spz) and get_grismz: will not happen,
        I do not include it in hte following condition checks.
        """

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

        # Combine hte above strings and write
        fh.write(str_to_write1 + str_to_write2 + str_to_write3 + str_to_write4 + str_to_write5 + \
            str_to_write6 + str_to_write7 + str_to_write8 + str_to_write9 + str_to_write10)

    print "Results saved for:", current_field, current_id

    return None