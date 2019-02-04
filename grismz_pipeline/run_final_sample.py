from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.integrate import simps
import multiprocessing as mp

import os
import sys
import time
import datetime

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"
threedhst_datadir = home + "/Desktop/3dhst_data/"
massive_figures_dir = figs_dir + 'massive-galaxies-figures/'
savedir_photoz = massive_figures_dir + 'photoz_run_jan2019/'  # Required to save p(z) curve and z_arr
savedir_spz = massive_figures_dir + 'spz_run_jan2019/'  # Required to save p(z) curve and z_arr
savedir_grismz = massive_figures_dir + 'grismz_run_jan2019/'  # Required to save p(z) curve and z_arr

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import refine_redshifts_dn4000 as old_ref
from fullfitting_grism_broadband_emlines import do_fitting, get_flam, get_flam_nonhst
from photoz import do_photoz_fitting_lookup
from new_refine_grismz_gridsearch_parallel import get_data
import model_mods as mm

speed_of_light = 299792458e10  # angstroms per second

def get_all_redshifts(current_id, current_field, current_ra, current_dec, current_specz,\
    goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam, \
    model_lam_grid_withlines, model_comp_spec_withlines, all_model_flam, total_models, start, \
    log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr):

    # Skip galaxies already done. Read in data for galaxies already done.
    # Yes this has to be read in every time this function is called.
    # This is because if this function does run successfully (as in the normal
    # case) then it will append to these arrays and change their contents.
    # So it is necessary for this function to know the latest contents
    # of these arrays.
    id_arr = np.load(savedir_photoz + 'id_arr.npy')
    field_arr = np.load(savedir_photoz + 'field_arr.npy')
    zs_arr = np.load(savedir_photoz + 'zs_arr.npy')

    zp_minchi2_arr = np.load(savedir_photoz + 'zp_minchi2_arr.npy')
    zp_arr = np.load(savedir_photoz + 'zp_arr.npy')
    zp_zerr_low_arr = np.load(savedir_photoz + 'zp_zerr_low_arr.npy')
    zp_zerr_up_arr = np.load(savedir_photoz + 'zp_zerr_up_arr.npy')
    zp_min_chi2_arr = np.load(savedir_photoz + 'zp_min_chi2_arr.npy')
    zp_bestalpha_arr = np.load(savedir_photoz + 'zp_bestalpha_arr.npy')
    zp_model_idx_arr = np.load(savedir_photoz + 'zp_model_idx_arr.npy')
    zp_age_arr = np.load(savedir_photoz + 'zp_age_arr.npy')
    zp_tau_arr = np.load(savedir_photoz + 'zp_tau_arr.npy')
    zp_av_arr = np.load(savedir_photoz + 'zp_av_arr.npy')

    zspz_minchi2_arr = np.load(savedir_spz + 'zspz_minchi2_arr.npy')
    zspz_arr = np.load(savedir_spz + 'zspz_arr.npy')
    zspz_zerr_low_arr = np.load(savedir_spz + 'zspz_zerr_low_arr.npy')
    zspz_zerr_up_arr = np.load(savedir_spz + 'zspz_zerr_up_arr.npy')
    zspz_min_chi2_arr = np.load(savedir_spz + 'zspz_min_chi2_arr.npy')
    zspz_bestalpha_arr = np.load(savedir_spz + 'zspz_bestalpha_arr.npy')
    zspz_model_idx_arr = np.load(savedir_spz + 'zspz_model_idx_arr.npy')
    zspz_age_arr = np.load(savedir_spz + 'zspz_age_arr.npy')
    zspz_tau_arr = np.load(savedir_spz + 'zspz_tau_arr.npy')
    zspz_av_arr = np.load(savedir_spz + 'zspz_av_arr.npy')

    zg_minchi2_arr = np.load(savedir_grismz + 'zg_minchi2_arr.npy')
    zg_arr = np.load(savedir_grismz + 'zg_arr.npy')
    zg_zerr_low_arr = np.load(savedir_grismz + 'zg_zerr_low_arr.npy')
    zg_zerr_up_arr = np.load(savedir_grismz + 'zg_zerr_up_arr.npy')
    zg_min_chi2_arr = np.load(savedir_grismz + 'zg_min_chi2_arr.npy')
    zg_bestalpha_arr = np.load(savedir_grismz + 'zg_bestalpha_arr.npy')
    zg_model_idx_arr = np.load(savedir_grismz + 'zg_model_idx_arr.npy')
    zg_age_arr = np.load(savedir_grismz + 'zg_age_arr.npy')
    zg_tau_arr = np.load(savedir_grismz + 'zg_tau_arr.npy')
    zg_av_arr = np.load(savedir_grismz + 'zg_av_arr.npy')

    # First check that the galaxy hasn't already been done
    if (current_id in id_arr) and (current_field in field_arr):
        print "At:", current_id, "in", current_field, "Skipping since this has been done."
        return None

    print "\n", "Working on:", current_field, current_id, "at", current_specz

    modify_lsf = True

    # Assign catalogs 
    if current_field == 'GOODS-N':
        phot_cat_3dhst = goodsn_phot_cat_3dhst
    elif current_field == 'GOODS-S':
        phot_cat_3dhst = goodss_phot_cat_3dhst

    # ------------------------------- Get grism data and then match with photometry ------------------------------- #
    grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = get_data(current_id, current_field)

    if return_code == 0:
        print "Skipping due to an error with the obs data. See the error message just above this one.",
        print "Moving to the next galaxy."
        return None

    threed_ra = phot_cat_3dhst['ra']
    threed_dec = phot_cat_3dhst['dec']

    # Now match
    ra_lim = 0.3/3600  # arcseconds in degrees
    dec_lim = 0.3/3600
    threed_phot_idx = np.where((threed_ra >= current_ra - ra_lim) & (threed_ra <= current_ra + ra_lim) & \
        (threed_dec >= current_dec - dec_lim) & (threed_dec <= current_dec + dec_lim))[0]

    """
    If there are multiple matches with the photometry catalog 
    within 0.5 arseconds then choose the closest one.
    """
    if len(threed_phot_idx) > 1:
        print "Multiple matches found in Photmetry catalog. Choosing the closest one."

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

    elif len(threed_phot_idx) == 0:
        print "Match not found in Photmetry catalog. Exiting."
        sys.exit(0)

    # ------------------------------- Get photometric fluxes and their errors ------------------------------- #
    flam_f435w = get_flam('F435W', phot_cat_3dhst['f_F435W'][threed_phot_idx])
    flam_f606w = get_flam('F606W', phot_cat_3dhst['f_F606W'][threed_phot_idx])
    flam_f775w = get_flam('F775W', phot_cat_3dhst['f_F775W'][threed_phot_idx])
    flam_f850lp = get_flam('F850LP', phot_cat_3dhst['f_F850LP'][threed_phot_idx])
    flam_f125w = get_flam('F125W', phot_cat_3dhst['f_F125W'][threed_phot_idx])
    flam_f140w = get_flam('F140W', phot_cat_3dhst['f_F140W'][threed_phot_idx])
    flam_f160w = get_flam('F160W', phot_cat_3dhst['f_F160W'][threed_phot_idx])

    flam_U = get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['f_U'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac1 = get_flam_nonhst('irac1', phot_cat_3dhst['f_IRAC1'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac2 = get_flam_nonhst('irac2', phot_cat_3dhst['f_IRAC2'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac3 = get_flam_nonhst('irac3', phot_cat_3dhst['f_IRAC3'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac4 = get_flam_nonhst('irac4', phot_cat_3dhst['f_IRAC4'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

    ferr_f435w = get_flam('F435W', phot_cat_3dhst['e_F435W'][threed_phot_idx])
    ferr_f606w = get_flam('F606W', phot_cat_3dhst['e_F606W'][threed_phot_idx])
    ferr_f775w = get_flam('F775W', phot_cat_3dhst['e_F775W'][threed_phot_idx])
    ferr_f850lp = get_flam('F850LP', phot_cat_3dhst['e_F850LP'][threed_phot_idx])
    ferr_f125w = get_flam('F125W', phot_cat_3dhst['e_F125W'][threed_phot_idx])
    ferr_f140w = get_flam('F140W', phot_cat_3dhst['e_F140W'][threed_phot_idx])
    ferr_f160w = get_flam('F160W', phot_cat_3dhst['e_F160W'][threed_phot_idx])

    ferr_U = get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['e_U'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac1 = get_flam_nonhst('irac1', phot_cat_3dhst['e_IRAC1'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac2 = get_flam_nonhst('irac2', phot_cat_3dhst['e_IRAC2'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac3 = get_flam_nonhst('irac3', phot_cat_3dhst['e_IRAC3'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac4 = get_flam_nonhst('irac4', phot_cat_3dhst['e_IRAC4'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

    # ------------------------------- Apply aperture correction ------------------------------- #
    # First interpolate the given filter curve on to the wavelength frid of the grism data
    # You only need the F775W filter here since you're only using this filter to get the 
    # aperture correction factor.
    f775w_filt_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f775w_filt_curve.txt', \
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
    # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/#keywords
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
    avg_dlam = old_ref.get_avg_dlam(grism_lam_obs)

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

    # ------------- Call fitting function for photo-z ------------- #
    print "Computing photo-z now."

    zp_minchi2, zp, zp_zerr_low, zp_zerr_up, zp_min_chi2, zp_bestalpha, zp_model_idx, zp_age, zp_tau, zp_av = \
    do_photoz_fitting_lookup(phot_fluxes_arr, phot_errors_arr, phot_lam, \
        model_lam_grid_withlines, total_models, model_comp_spec_withlines, start,\
        current_id, current_field, all_model_flam, phot_fin_idx, current_specz, savedir_photoz, \
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr)

    # ------------- Call fitting function for SPZ ------------- #
    print "\n", "Photo-z done. Moving on to SPZ computation now."

    zspz_minchi2, zspz, zspz_zerr_low, zspz_zerr_up, zspz_min_chi2, zspz_bestalpha, zspz_model_idx, zspz_age, zspz_tau, zspz_av = \
    do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_fluxes_arr, phot_errors_arr, phot_lam, \
        lsf_to_use, resampling_lam_grid, len(resampling_lam_grid), all_model_flam, phot_fin_idx, \
        model_lam_grid_withlines, total_models, model_comp_spec_withlines, start, current_id, current_field, current_specz, zp, \
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, \
        use_broadband=True, single_galaxy=False, for_loop_method='parallel')

    # ------------- Call fitting function for grism-z ------------- #
    # Essentially just calls the same function as above but switches off broadband for the fit
    print "\n", "SPZ done. Moving on to grism-z computation now."
    
    zg_minchi2, zg, zg_zerr_low, zg_zerr_up, zg_min_chi2, zg_bestalpha, zg_model_idx, zg_age, zg_tau, zg_av = \
    do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_fluxes_arr, phot_errors_arr, phot_lam, \
        lsf_to_use, resampling_lam_grid, len(resampling_lam_grid), all_model_flam, phot_fin_idx, \
        model_lam_grid_withlines, total_models, model_comp_spec_withlines, start, current_id, current_field, current_specz, zp, \
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, \
        use_broadband=False, single_galaxy=False, for_loop_method='parallel')

    # Append to existing numpy arrays and save
    id_arr = np.append(id_arr, current_id)
    field_arr = np.append(field_arr, current_field)
    zs_arr = np.append(zs_arr, current_specz)

    zp_minchi2_arr = np.append(zp_minchi2_arr, zp_minchi2)
    zp_arr = np.append(zp_arr, zp)
    zp_zerr_low_arr = np.append(zp_zerr_low_arr, zp_zerr_low)
    zp_zerr_up_arr = np.append(zp_zerr_up_arr, zp_zerr_up)
    zp_min_chi2_arr = np.append(zp_min_chi2_arr, zp_min_chi2)
    zp_bestalpha_arr = np.append(zp_bestalpha_arr, zp_bestalpha)
    zp_model_idx_arr = np.append(zp_model_idx_arr, zp_model_idx)
    zp_age_arr = np.append(zp_age_arr, zp_age)
    zp_tau_arr = np.append(zp_tau_arr, zp_tau)
    zp_av_arr = np.append(zp_av_arr, zp_av)

    zspz_minchi2_arr = np.append(zspz_minchi2_arr, zspz_minchi2)
    zspz_arr = np.append(zspz_arr, zspz)
    zspz_zerr_low_arr = np.append(zspz_zerr_low_arr, zspz_zerr_low)
    zspz_zerr_up_arr = np.append(zspz_zerr_up_arr, zspz_zerr_up)
    zspz_min_chi2_arr = np.append(zspz_min_chi2_arr, zspz_min_chi2)
    zspz_bestalpha_arr = np.append(zspz_bestalpha_arr, zspz_bestalpha)
    zspz_model_idx_arr = np.append(zspz_model_idx_arr, zspz_model_idx)
    zspz_age_arr = np.append(zspz_age_arr, zspz_age)
    zspz_tau_arr = np.append(zspz_tau_arr, zspz_tau)
    zspz_av_arr = np.append(zspz_av_arr, zspz_av)

    zg_minchi2_arr = np.append(zg_minchi2_arr, zg_minchi2)
    zg_arr = np.append(zg_arr, zg)
    zg_zerr_low_arr = np.append(zg_zerr_low_arr, zg_zerr_low)
    zg_zerr_up_arr = np.append(zg_zerr_up_arr, zg_zerr_up)
    zg_min_chi2_arr = np.append(zg_min_chi2_arr, zg_min_chi2)
    zg_bestalpha_arr = np.append(zg_bestalpha_arr, zg_bestalpha)
    zg_model_idx_arr = np.append(zg_model_idx_arr, zg_model_idx)
    zg_age_arr = np.append(zg_age_arr, zg_age)
    zg_tau_arr = np.append(zg_tau_arr, zg_tau)
    zg_av_arr = np.append(zg_av_arr, zg_av)

    np.save(savedir_photoz + 'id_arr.npy', id_arr)
    np.save(savedir_photoz + 'field_arr.npy', field_arr)
    np.save(savedir_photoz + 'zs_arr.npy', zs_arr)

    np.save(savedir_photoz + 'zp_minchi2_arr.npy', zp_minchi2_arr)
    np.save(savedir_photoz + 'zp_arr.npy', zp_arr)
    np.save(savedir_photoz + 'zp_zerr_low_arr.npy', zp_zerr_low_arr)
    np.save(savedir_photoz + 'zp_zerr_up_arr.npy', zp_zerr_up_arr)
    np.save(savedir_photoz + 'zp_min_chi2_arr.npy', zp_min_chi2_arr)
    np.save(savedir_photoz + 'zp_bestalpha_arr.npy', zp_bestalpha_arr)
    np.save(savedir_photoz + 'zp_model_idx_arr.npy', zp_model_idx_arr)
    np.save(savedir_photoz + 'zp_age_arr.npy', zp_age_arr)
    np.save(savedir_photoz + 'zp_tau_arr.npy', zp_tau_arr)
    np.save(savedir_photoz + 'zp_av_arr.npy', zp_av_arr)

    np.save(savedir_spz + 'zspz_minchi2_arr.npy', zspz_minchi2_arr)
    np.save(savedir_spz + 'zspz_arr.npy', zspz_arr)
    np.save(savedir_spz + 'zspz_zerr_low_arr.npy', zspz_zerr_low_arr)
    np.save(savedir_spz + 'zspz_zerr_up_arr.npy', zspz_zerr_up_arr)
    np.save(savedir_spz + 'zspz_min_chi2_arr.npy', zspz_min_chi2_arr)
    np.save(savedir_spz + 'zspz_bestalpha_arr.npy', zspz_bestalpha_arr)
    np.save(savedir_spz + 'zspz_model_idx_arr.npy', zspz_model_idx_arr)
    np.save(savedir_spz + 'zspz_age_arr.npy', zspz_age_arr)
    np.save(savedir_spz + 'zspz_tau_arr.npy', zspz_tau_arr)
    np.save(savedir_spz + 'zspz_av_arr.npy', zspz_av_arr)

    np.save(savedir_grismz + 'zg_minchi2_arr.npy', zg_minchi2_arr)
    np.save(savedir_grismz + 'zg_arr.npy', zg_arr)
    np.save(savedir_grismz + 'zg_zerr_low_arr.npy', zg_zerr_low_arr)
    np.save(savedir_grismz + 'zg_zerr_up_arr.npy', zg_zerr_up_arr)
    np.save(savedir_grismz + 'zg_min_chi2_arr.npy', zg_min_chi2_arr)
    np.save(savedir_grismz + 'zg_bestalpha_arr.npy', zg_bestalpha_arr)
    np.save(savedir_grismz + 'zg_model_idx_arr.npy', zg_model_idx_arr)
    np.save(savedir_grismz + 'zg_age_arr.npy', zg_age_arr)
    np.save(savedir_grismz + 'zg_tau_arr.npy', zg_tau_arr)
    np.save(savedir_grismz + 'zg_av_arr.npy', zg_av_arr)

    print "Intermediate result arrays saved."

    return None

def main():

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # ------------------------------- Get correct directories ------------------------------- #
    figs_data_dir = '/Volumes/Bhavins_backup/bc03_models_npy_spectra/'
    threedhst_datadir = "/Volumes/Bhavins_backup/3dhst_data/"
    cspout = "/Volumes/Bhavins_backup/bc03_models_npy_spectra/cspout_2016updated_galaxev/"
    # This is if working on the laptop. 
    # Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(figs_data_dir):
        figs_data_dir = figs_dir  # this path only exists on firstlight
        threedhst_datadir = home + "/Desktop/3dhst_data/"  # this path only exists on firstlight
        cspout = home + '/Documents/galaxev_bc03_2016update/bc03/src/cspout_2016updated_galaxev/'
        if not os.path.isdir(figs_data_dir):
            print "Model files not found. Exiting..."
            sys.exit(0)

    # ------------------------------- Get catalog for final sample ------------------------------- #
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample.txt', dtype=None, names=True)

    # ------------------------------ Get models ------------------------------ #
    # read in entire model set
    # To see how these arrays were created check the code:
    # $HOME/Desktop/test-codes/shared_memory_multiprocessing/shmem_parallel_proc.py
    # This part will fail if the arrays dont already exist.
    total_models = 37761 # get_total_extensions(bc03_all_spec_hdulist)

    log_age_arr = np.load(figs_data_dir + 'log_age_arr.npy', mmap_mode='r')
    metal_arr = np.load(figs_data_dir + 'metal_arr.npy', mmap_mode='r')
    nlyc_arr = np.load(figs_data_dir + 'nlyc_arr.npy', mmap_mode='r')
    tau_gyr_arr = np.load(figs_data_dir + 'tau_gyr_arr.npy', mmap_mode='r')
    tauv_arr = np.load(figs_data_dir + 'tauv_arr.npy', mmap_mode='r')
    ub_col_arr = np.load(figs_data_dir + 'ub_col_arr.npy', mmap_mode='r')
    bv_col_arr = np.load(figs_data_dir + 'bv_col_arr.npy', mmap_mode='r')
    vj_col_arr = np.load(figs_data_dir + 'vj_col_arr.npy', mmap_mode='r')
    ms_arr = np.load(figs_data_dir + 'ms_arr.npy', mmap_mode='r')
    mgal_arr = np.load(figs_data_dir + 'mgal_arr.npy', mmap_mode='r')

    model_lam_grid_withlines_mmap = np.load(figs_data_dir + 'model_lam_grid_withlines.npy', mmap_mode='r')
    model_comp_spec_withlines_mmap = np.load(figs_data_dir + 'model_comp_spec_withlines.npy', mmap_mode='r')

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

    # ---------------------------------- Read in look-up tables for model mags ------------------------------------- #
    # Using the look-up table now since it should be much faster
    # Again check the code --
    # $HOME/Desktop/test-codes/shared_memory_multiprocessing/shmem_parallel_proc.py
    # to see how this was created
    # This part will fail if the array does not already exist.
    all_model_flam_mmap = np.load(figs_data_dir + 'all_model_flam.npy', mmap_mode='r')

    # ------------------------------- Read in photometry catalogs ------------------------------- #
    # GOODS-N from 3DHST
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
    'f_U', 'e_U', 'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
    'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), skip_header=3)

    # Read in Vega spectrum and get it in the appropriate forms
    vega = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/' + 'vega_reference.dat', dtype=None, \
        names=['wav', 'flam'], skip_header=7)

    vega_lam = vega['wav']
    vega_spec_flam = vega['flam']
    vega_nu = speed_of_light / vega_lam
    vega_spec_fnu = vega_lam**2 * vega_spec_flam / speed_of_light

    # ------------------------------- Looping over each object ------------------------------- #
    total_final_sample = len(final_sample)
    galaxy_count = 0
    """
    num_cores = 3
    print "Running on", num_cores, "cores."
    for j in range(44):  # doing the first 176 galaxies on firstlight --> 176 / 4 = 44

        print "Galaxies done so far:", galaxy_count
        print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."

        processes = [mp.Process(target=get_all_redshifts, args=(final_sample['pearsid'][u], final_sample['field'][u], final_sample['ra'][u], final_sample['dec'][u], 
            final_sample['specz'][u], goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam, \
            model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap, all_model_flam_mmap, total_models, start, \
            log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr)) for u in xrange(num_cores*j, num_cores*(j+1))]
        for p in processes:
            p.start()
            print "Current process ID:", p.pid
        for p in processes:
            p.join()

        #zp_minchi2, zp, zp_zerr_low, zp_zerr_up, zp_min_chi2, zp_bestalpha, zp_model_idx, zp_age, zp_tau, zp_av = \
        #get_all_redshifts(current_id, current_field, \
        #final_sample['ra'][j], final_sample['dec'][j], \
        #final_sample['specz'][j], goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam, \
        #bc03_all_spec_hdulist, model_lam_grid_withlines, model_comp_spec_withlines, all_model_flam, total_models, start)

        # This is trigerred if the return code from ngp.get_data() is 0. i.e. excess contamination or incomplete wav array.
        # This is also trigerred if the LSF isn't found and the function above returns prematurely
        #if zp == -99.0:
        #    continue

        galaxy_count += num_cores
    """

    for j in range(total_final_sample):

        print "Galaxies done so far:", galaxy_count
        print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."

        get_all_redshifts(final_sample['pearsid'][j], final_sample['field'][j], final_sample['ra'][j], final_sample['dec'][j], 
            final_sample['specz'][j], goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam, \
            model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap, all_model_flam_mmap, total_models, start, \
            log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr)

        galaxy_count += 1

    print "All done."
    print "Final number within sample:", galaxy_count

    return None

if __name__ == '__main__':
    main()
    sys.exit()