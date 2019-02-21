from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.integrate import simps

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
import check_single_galaxy_fitting_spz_photoz as chk
from new_refine_grismz_gridsearch_parallel import get_data
from fullfitting_grism_broadband_emlines import get_flam, get_flam_nonhst
import dn4000_catalog as dc
from spz_photoz_grismz_comparison import get_z_errors

speed_of_light = 299792458e10  # angstroms per second

def get_arrays_for_plotting():

    # Read in arrays from Firstlight (fl) and Jet (jt) and combine them
    zp_results_dir = savedir_photoz
    zg_results_dir = savedir_grismz
    spz_results_dir = savedir_spz

    # ----- Firstlight -----
    id_arr_fl = np.load(zp_results_dir + 'firstlight_id_arr.npy')
    field_arr_fl = np.load(zp_results_dir + 'firstlight_field_arr.npy')
    zs_arr_fl = np.load(zp_results_dir + 'firstlight_zs_arr.npy')

    # Length checks
    assert len(id_arr_fl) == len(field_arr_fl)
    assert len(id_arr_fl) == len(zs_arr_fl)

    # Redshift and Error arrays
    zp_arr_fl = np.zeros(id_arr_fl.shape[0])
    zg_arr_fl = np.zeros(id_arr_fl.shape[0])
    zspz_arr_fl = np.zeros(id_arr_fl.shape[0])

    zp_low_bound_fl = np.zeros(id_arr_fl.shape[0])
    zp_high_bound_fl = np.zeros(id_arr_fl.shape[0])

    zg_low_bound_fl = np.zeros(id_arr_fl.shape[0])
    zg_high_bound_fl = np.zeros(id_arr_fl.shape[0])

    zspz_low_bound_fl = np.zeros(id_arr_fl.shape[0])
    zspz_high_bound_fl = np.zeros(id_arr_fl.shape[0])

    # min chi2 values
    zp_min_chi2_fl = np.load(zp_results_dir + 'firstlight_zp_min_chi2_arr.npy')
    zg_min_chi2_fl = np.load(zg_results_dir + 'firstlight_zg_min_chi2_arr.npy')
    zspz_min_chi2_fl = np.load(spz_results_dir + 'firstlight_zspz_min_chi2_arr.npy')

    # Make sure you're getting the exact redshift corresponding to the peak of the p(z) curve
    for u in range(len(id_arr_fl)):
        zp_pz = np.load(zp_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_photoz_pz.npy')
        zp_zarr = np.load(zp_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_photoz_z_arr.npy')
        zp_arr_fl[u] = zp_zarr[np.argmax(zp_pz)]

        zspz_pz = np.load(spz_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_spz_pz.npy')
        zspz_zarr = np.load(spz_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_spz_z_arr.npy')
        zspz_arr_fl[u] = zspz_zarr[np.argmax(zspz_pz)]

        zg_pz = np.load(zg_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_zg_pz.npy')
        zg_zarr = np.load(zg_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_zg_z_arr.npy')
        zg_arr_fl[u] = zg_zarr[np.argmax(zg_pz)]

        # Get errors and save them to a file
        zp_low_bound_fl[u], zp_high_bound_fl[u] = get_z_errors(zp_zarr, zp_pz)
        zg_low_bound_fl[u], zg_high_bound_fl[u] = get_z_errors(zg_zarr, zg_pz)
        zspz_low_bound_fl[u], zspz_high_bound_fl[u] = get_z_errors(zspz_zarr, zspz_pz)

    # Best fit model idx and alpha
    zp_model_idx_arr_fl = np.load(zp_results_dir + 'firstlight_zp_model_idx_arr.npy')
    zp_bestalpha_arr_fl = np.load(zp_results_dir + 'firstlight_zp_bestalpha_arr.npy')

    zg_model_idx_arr_fl = np.load(zg_results_dir + 'firstlight_zg_model_idx_arr.npy')
    zg_bestalpha_arr_fl = np.load(zg_results_dir + 'firstlight_zg_bestalpha_arr.npy')

    zspz_model_idx_arr_fl = np.load(spz_results_dir + 'firstlight_zspz_model_idx_arr.npy')
    zspz_bestalpha_arr_fl = np.load(spz_results_dir + 'firstlight_zspz_bestalpha_arr.npy')

    # Best fit params
    zp_age_arr_fl = np.load(zp_results_dir + 'firstlight_zp_age_arr.npy')
    zp_tau_arr_fl = np.load(zp_results_dir + 'firstlight_zp_tau_arr.npy')
    zp_av_arr_fl = np.load(zp_results_dir + 'firstlight_zp_av_arr.npy')

    zg_age_arr_fl = np.load(zg_results_dir + 'firstlight_zg_age_arr.npy')
    zg_tau_arr_fl = np.load(zg_results_dir + 'firstlight_zg_tau_arr.npy')
    zg_av_arr_fl = np.load(zg_results_dir + 'firstlight_zg_av_arr.npy')

    zspz_age_arr_fl = np.load(spz_results_dir + 'firstlight_zspz_age_arr.npy')
    zspz_tau_arr_fl = np.load(spz_results_dir + 'firstlight_zspz_tau_arr.npy')
    zspz_av_arr_fl = np.load(spz_results_dir + 'firstlight_zspz_av_arr.npy')

    # ----- Jet ----- 
    id_arr_jt = np.load(zp_results_dir + 'jet_id_arr.npy')
    field_arr_jt = np.load(zp_results_dir + 'jet_field_arr.npy')
    zs_arr_jt = np.load(zp_results_dir + 'jet_zs_arr.npy')

    # Length checks
    assert len(id_arr_jt) == len(field_arr_jt)
    assert len(id_arr_jt) == len(zs_arr_jt)

    # Redshift and Error arrays
    zp_arr_jt = np.zeros(id_arr_jt.shape[0])
    zg_arr_jt = np.zeros(id_arr_jt.shape[0])
    zspz_arr_jt = np.zeros(id_arr_jt.shape[0])

    zp_low_bound_jt = np.zeros(id_arr_jt.shape[0])
    zp_high_bound_jt = np.zeros(id_arr_jt.shape[0])

    zg_low_bound_jt = np.zeros(id_arr_jt.shape[0])
    zg_high_bound_jt = np.zeros(id_arr_jt.shape[0])

    zspz_low_bound_jt = np.zeros(id_arr_jt.shape[0])
    zspz_high_bound_jt = np.zeros(id_arr_jt.shape[0])

    # min chi2 values
    zp_min_chi2_jt = np.load(zp_results_dir + 'jet_zp_min_chi2_arr.npy')
    zg_min_chi2_jt = np.load(zg_results_dir + 'jet_zg_min_chi2_arr.npy')
    zspz_min_chi2_jt = np.load(spz_results_dir + 'jet_zspz_min_chi2_arr.npy')

    # Make sure you're getting the exact redshift corresponding to the peak of the p(z) curve
    for v in range(len(id_arr_jt)):
        zp_pz = np.load(zp_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_photoz_pz.npy')
        zp_zarr = np.load(zp_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_photoz_z_arr.npy')
        zp_arr_jt[v] = zp_zarr[np.argmax(zp_pz)]

        zspz_pz = np.load(spz_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_spz_pz.npy')
        zspz_zarr = np.load(spz_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_spz_z_arr.npy')
        zspz_arr_jt[v] = zspz_zarr[np.argmax(zspz_pz)]

        zg_pz = np.load(zg_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_zg_pz.npy')
        zg_zarr = np.load(zg_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_zg_z_arr.npy')
        zg_arr_jt[v] = zg_zarr[np.argmax(zg_pz)]

        # Get errors and save them to a file
        zp_low_bound_jt[u], zp_high_bound_jt[u] = get_z_errors(zp_zarr, zp_pz)
        zg_low_bound_jt[u], zg_high_bound_jt[u] = get_z_errors(zg_zarr, zg_pz)
        zspz_low_bound_jt[u], zspz_high_bound_jt[u] = get_z_errors(zspz_zarr, zspz_pz)

    # Best fit model idx and alpha
    zp_model_idx_arr_jt = np.load(zp_results_dir + 'jet_zp_model_idx_arr.npy')
    zp_bestalpha_arr_jt = np.load(zp_results_dir + 'jet_zp_bestalpha_arr.npy')

    zg_model_idx_arr_jt = np.load(zg_results_dir + 'jet_zg_model_idx_arr.npy')
    zg_bestalpha_arr_jt = np.load(zg_results_dir + 'jet_zg_bestalpha_arr.npy')

    zspz_model_idx_arr_jt = np.load(spz_results_dir + 'jet_zspz_model_idx_arr.npy')
    zspz_bestalpha_arr_jt = np.load(spz_results_dir + 'jet_zspz_bestalpha_arr.npy')

    # Best fit params
    zp_age_arr_jt = np.load(zp_results_dir + 'jet_zp_age_arr.npy')
    zp_tau_arr_jt = np.load(zp_results_dir + 'jet_zp_tau_arr.npy')
    zp_av_arr_jt = np.load(zp_results_dir + 'jet_zp_av_arr.npy')

    zg_age_arr_jt = np.load(zg_results_dir + 'jet_zg_age_arr.npy')
    zg_tau_arr_jt = np.load(zg_results_dir + 'jet_zg_tau_arr.npy')
    zg_av_arr_jt = np.load(zg_results_dir + 'jet_zg_av_arr.npy')

    zspz_age_arr_jt = np.load(spz_results_dir + 'jet_zspz_age_arr.npy')
    zspz_tau_arr_jt = np.load(spz_results_dir + 'jet_zspz_tau_arr.npy')
    zspz_av_arr_jt = np.load(spz_results_dir + 'jet_zspz_av_arr.npy')

    # ----- Concatenate -----
    # check for any accidental overlaps
    # I'm just doing an explicit for loop because I need to compare both ID and field
    min_len = len(id_arr_fl)  # Since firstlight went through fewer galaxies
    common_indices_jt = []
    for j in range(min_len):

        id_to_search = id_arr_fl[j]
        field_to_search = field_arr_fl[j]

        """
        Note the order of the two if statements below. 
        if (id_to_search in id_arr_jt) and (field_to_search in field_arr_jt)
        WILL NOT WORK! 
        This is because the second condition there is always true.
        """
        if (id_to_search in id_arr_jt):
            jt_idx = int(np.where(id_arr_jt == id_to_search)[0])
            if (field_arr_jt[jt_idx] == field_to_search):
                common_indices_jt.append(jt_idx)

    # Delete common galaxies from Jet arrays 
    # ONly delete from one of the set of arrays since you want these galaxies included only once
    # ----- Jet arrays with common galaxies deleted ----- 
    id_arr_jt = np.delete(id_arr_jt, common_indices_jt, axis=None)
    field_arr_jt = np.delete(field_arr_jt, common_indices_jt, axis=None)
    zs_arr_jt = np.delete(zs_arr_jt, common_indices_jt, axis=None)

    zp_arr_jt = np.delete(zp_arr_jt, common_indices_jt, axis=None)
    zg_arr_jt = np.delete(zg_arr_jt, common_indices_jt, axis=None)
    zspz_arr_jt = np.delete(zspz_arr_jt, common_indices_jt, axis=None)

    # min chi2 values
    zp_min_chi2_jt = np.delete(zp_min_chi2_jt, common_indices_jt, axis=None)
    zg_min_chi2_jt = np.delete(zg_min_chi2_jt, common_indices_jt, axis=None)
    zspz_min_chi2_jt = np.delete(zspz_min_chi2_jt, common_indices_jt, axis=None)

    # Best fit model idx and alpha
    zp_model_idx_arr_jt = np.delete(zp_model_idx_arr_jt, common_indices_jt, axis=None)
    zp_bestalpha_arr_jt = np.delete(zp_bestalpha_arr_jt, common_indices_jt, axis=None)

    zg_model_idx_arr_jt = np.delete(zg_model_idx_arr_jt, common_indices_jt, axis=None)
    zg_bestalpha_arr_jt = np.delete(zg_bestalpha_arr_jt, common_indices_jt, axis=None)

    zspz_model_idx_arr_jt = np.delete(zspz_model_idx_arr_jt, common_indices_jt, axis=None)
    zspz_bestalpha_arr_jt = np.delete(zspz_bestalpha_arr_jt, common_indices_jt, axis=None)

    # Best fit params
    zp_age_arr_jt = np.delete(zp_age_arr_jt, common_indices_jt, axis=None)
    zp_tau_arr_jt = np.delete(zp_tau_arr_jt, common_indices_jt, axis=None)
    zp_av_arr_jt = np.delete(zp_av_arr_jt, common_indices_jt, axis=None)

    zg_age_arr_jt = np.delete(zg_age_arr_jt, common_indices_jt, axis=None)
    zg_tau_arr_jt = np.delete(zg_tau_arr_jt, common_indices_jt, axis=None)
    zg_av_arr_jt = np.delete(zg_av_arr_jt, common_indices_jt, axis=None)

    zspz_age_arr_jt = np.delete(zspz_age_arr_jt, common_indices_jt, axis=None)
    zspz_tau_arr_jt = np.delete(zspz_tau_arr_jt, common_indices_jt, axis=None)
    zspz_av_arr_jt = np.delete(zspz_av_arr_jt, common_indices_jt, axis=None)

    # --- Lower and Upper bounds
    zp_low_bound_jt = np.delete(zp_low_bound_jt, common_indices_jt, axis=None)
    zp_high_bound_jt = np.delete(zp_high_bound_jt, common_indices_jt, axis=None)

    zg_low_bound_jt = np.delete(zg_low_bound_jt, common_indices_jt, axis=None)
    zg_high_bound_jt = np.delete(zg_high_bound_jt, common_indices_jt, axis=None)

    zspz_low_bound_jt = np.delete(zspz_low_bound_jt, common_indices_jt, axis=None)
    zspz_high_bound_jt = np.delete(zspz_high_bound_jt, common_indices_jt, axis=None)

    # -------------------- Actual concatenation
    # The order while concatenating is important! 
    # Stay consistent! fl is before jt
    all_ids = np.concatenate((id_arr_fl, id_arr_jt))
    all_fields = np.concatenate((field_arr_fl, field_arr_jt))

    zs = np.concatenate((zs_arr_fl, zs_arr_jt))
    zp = np.concatenate((zp_arr_fl, zp_arr_jt))
    zg = np.concatenate((zg_arr_fl, zg_arr_jt))
    zspz = np.concatenate((zspz_arr_fl, zspz_arr_jt))

    zp_chi2 = np.concatenate((zp_min_chi2_fl, zp_min_chi2_jt))
    zg_chi2 = np.concatenate((zg_min_chi2_fl, zg_min_chi2_jt))
    zspz_chi2 = np.concatenate((zspz_min_chi2_fl, zspz_min_chi2_jt))

    zp_model_idx_arr = np.concatenate((zp_model_idx_arr_fl, zp_model_idx_arr_jt))
    zg_model_idx_arr = np.concatenate((zg_model_idx_arr_fl, zg_model_idx_arr_jt))
    zspz_model_idx_arr = np.concatenate((zspz_model_idx_arr_fl, zspz_model_idx_arr_jt))

    zp_bestalpha_arr = np.concatenate((zp_bestalpha_arr_fl, zp_bestalpha_arr_jt))
    zg_bestalpha_arr = np.concatenate((zg_bestalpha_arr_fl, zg_bestalpha_arr_jt))
    zspz_bestalpha_arr = np.concatenate((zspz_bestalpha_arr_fl, zspz_bestalpha_arr_jt))

    zp_age_arr = np.concatenate((zp_age_arr_fl, zp_age_arr_jt))
    zp_tau_arr = np.concatenate((zp_tau_arr_fl, zp_tau_arr_jt))
    zp_av_arr = np.concatenate((zp_av_arr_fl, zp_av_arr_jt))

    zg_age_arr = np.concatenate((zg_age_arr_fl, zg_age_arr_jt))
    zg_tau_arr = np.concatenate((zg_tau_arr_fl, zg_tau_arr_jt))
    zg_av_arr = np.concatenate((zg_av_arr_fl, zg_av_arr_jt))

    zspz_age_arr = np.concatenate((zspz_age_arr_fl, zspz_age_arr_jt))
    zspz_tau_arr = np.concatenate((zspz_tau_arr_fl, zspz_tau_arr_jt))
    zspz_av_arr = np.concatenate((zspz_av_arr_fl, zspz_av_arr_jt))

    zp_low_bound_arr = np.concatenate((zp_low_bound_fl, zp_low_bound_jt))
    zp_high_bound_arr = np.concatenate((zp_high_bound_fl, zp_high_bound_jt))
    zg_low_bound_arr = np.concatenate((zg_low_bound_fl, zg_low_bound_jt))
    zg_high_bound_arr = np.concatenate((zg_high_bound_fl, zg_high_bound_jt))
    zspz_low_bound_arr = np.concatenate((zspz_low_bound_fl, zspz_low_bound_jt))
    zspz_high_bound_arr = np.concatenate((zspz_high_bound_fl, zspz_high_bound_jt))

    # ----- Get D4000 -----
    # Now loop over all galaxies to get D4000 and netsig
    all_d4000_list = []
    all_netsig_list = []
    for i in range(len(all_ids)):
        current_id = all_ids[i]
        current_field = all_fields[i]

        # Get data
        grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = get_data(current_id, current_field)

        if return_code == 0:
            print current_id, current_field
            print "Return code should not have been 0. Exiting."
            sys.exit(0)

        # Get D4000 at specz
        current_specz = zs[i]
        lam_em = grism_lam_obs / (1 + current_specz)
        flam_em = grism_flam_obs * (1 + current_specz)
        ferr_em = grism_ferr_obs * (1 + current_specz)

        d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)
        all_d4000_list.append(d4000)
        all_netsig_list.append(netsig_chosen)

    # Convert to arrays and return
    all_d4000 = np.array(all_d4000_list)
    all_netsig = np.array(all_netsig_list)

    return all_ids, all_fields, zs, zp, zg, zspz, all_d4000, all_netsig, zp_chi2, zg_chi2, zspz_chi2, \
    zp_model_idx_arr, zg_model_idx_arr, zspz_model_idx_arr, zp_bestalpha_arr, zg_bestalpha_arr, zspz_bestalpha_arr, \
    zp_age_arr, zp_tau_arr, zp_av_arr, zg_age_arr, zg_tau_arr, zg_av_arr, zspz_age_arr, zspz_tau_arr, zspz_av_arr, \
    zp_low_bound_arr, zp_high_bound_arr, zg_low_bound_arr, zg_high_bound_arr, zspz_low_bound_arr, zspz_high_bound_arr

def plotfit(current_id, current_field, ids, fields, zs_arr, zp_arr, zg_arr, zspz_arr, d4000, netsig, zp_chi2, zg_chi2, zspz_chi2, \
    zp_model_idx_arr, zg_model_idx_arr, zspz_model_idx_arr, zp_bestalpha_arr, zg_bestalpha_arr, zspz_bestalpha_arr, \
    zp_age_arr, zp_tau_arr, zp_av_arr, zg_age_arr, zg_tau_arr, zg_av_arr, zspz_age_arr, zspz_tau_arr, zspz_av_arr, \
    zp_low_bound_arr, zp_high_bound_arr, zg_low_bound_arr, zg_high_bound_arr, zspz_low_bound_arr, zspz_high_bound_arr, \
    final_sample, total_models, model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap, all_model_flam_mmap, \
    goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, vega):

    # Get RA and DEC
    sample_idx = np.where((final_sample['pearsid'] == current_id) & (final_sample['field'] == current_field))[0]
    current_ra = final_sample['ra'][sample_idx]
    current_dec = final_sample['dec'][sample_idx]

    vega_lam = vega['wav']
    vega_spec_flam = vega['flam']
    vega_nu = speed_of_light / vega_lam
    vega_spec_fnu = vega_lam**2 * vega_spec_flam / speed_of_light

    # Assign catalogs 
    if current_field == 'GOODS-N':
        phot_cat_3dhst = goodsn_phot_cat_3dhst
    elif current_field == 'GOODS-S':
        phot_cat_3dhst = goodss_phot_cat_3dhst

    # ------------------------------- Get grism data and then match with photometry ------------------------------- #
    grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = get_data(current_id, current_field)

    threed_ra = phot_cat_3dhst['ra']
    threed_dec = phot_cat_3dhst['dec']

    print current_ra, current_dec
    print threed_ra.shape, threed_dec.shape
    print current_ra.shape, current_dec.shape
    return

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
    modify_lsf = True
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

    # ------------------------------- Now get other info and plot  ------------------------------- #
    # find match
    match_idx = int(np.where((ids == current_id) & (fields == current_field))[0])

    current_specz = zs_arr[match_idx]
    current_d4000 = d4000[match_idx]

    # Get other fitting results
    # --- Peak (i.e. min chi2) redshifts
    zp = zp_arr[match_idx]
    zg = zg_arr[match_idx]
    zspz = zspz_arr[match_idx]

    # --- Model index for best model
    zp_model_idx = zp_model_idx_arr[match_idx]
    zg_model_idx = zg_model_idx_arr[match_idx]
    zspz_model_idx = zspz_model_idx_arr[match_idx]

    # --- Alpha for best model
    zp_bestalpha = zp_bestalpha_arr[match_idx]
    zg_bestalpha = zg_bestalpha_arr[match_idx]
    zspz_bestalpha = zspz_bestalpha_arr[match_idx]

    # --- Params
    zp_min_chi2 = zp_chi2[match_idx]
    zp_age = zp_age_arr[match_idx]
    zp_tau = zp_tau_arr[match_idx]
    zp_av = zp_av_arr[match_idx]
    
    zg_min_chi2 = zg_chi2[match_idx]
    zg_age = zg_age_arr[match_idx]
    zg_tau = zg_tau_arr[match_idx]
    zg_av = zg_av_arr[match_idx]

    zspz_min_chi2 = zspz_chi2[match_idx]
    zspz_age = zspz_age_arr[match_idx]
    zspz_tau = zspz_tau_arr[match_idx]
    zspz_av = zspz_av_arr[match_idx]

    # --- Lower and Upper bounds
    zp_zerr_low = zp_low_bound_arr[match_idx]
    zp_zerr_up = zp_high_bound_arr[match_idx]

    zg_zerr_low = zg_low_bound_arr[match_idx]
    zg_zerr_up = zg_high_bound_arr[match_idx]

    zspz_zerr_low = zspz_low_bound_arr[match_idx]
    zspz_zerr_up = zspz_high_bound_arr[match_idx]

    # ------------------------------- Get best fit model for plotting ------------------------------- #
    # Will have to do this at the photo-z and SPZ separtely otherwise the plots will not look right
    # ------------ Get best fit model for photo-z ------------ #
    zp_best_fit_model_fullres = model_comp_spec_withlines_mmap[zp_model_idx]
    zp_best_fit_model_fullres = zp_best_fit_model_fullres.ravel()  # This is more of a consistency forcing condition 
    zp_all_filt_flam_bestmodel = chk.get_photometry_best_fit_model(zp, zp_model_idx, phot_fin_idx, all_model_flam_mmap, total_models)

    # ------------ Get best fit model for SPZ ------------ #
    zspz_best_fit_model_in_objlamgrid, zspz_all_filt_flam_bestmodel, zspz_best_fit_model_fullres = \
    chk.get_best_fit_model_spz(resampling_lam_grid, len(resampling_lam_grid), model_lam_grid_withlines_mmap, \
        model_comp_spec_withlines_mmap, grism_lam_obs, zspz, zspz_model_idx, phot_fin_idx, all_model_flam_mmap, lsf_to_use, total_models)
    # Again these reshaping  to 1D calls seem to be needed with the mmapped arrays
    zspz_best_fit_model_fullres = zspz_best_fit_model_fullres.ravel()
    zspz_best_fit_model_in_objlamgrid = zspz_best_fit_model_in_objlamgrid.ravel()

    # ------------ Get best fit model for grism-z ------------ #
    zg_best_fit_model_in_objlamgrid, zg_best_fit_model_fullres = \
    chk.get_best_fit_model_grismz(resampling_lam_grid, len(resampling_lam_grid), model_lam_grid_withlines_mmap, \
        model_comp_spec_withlines_mmap, grism_lam_obs, zg, zg_model_idx, lsf_to_use, total_models)

    # ---------------- Now actual plotting ---------------- #
    chk.plot_photoz_fit(phot_lam, phot_fluxes_arr, phot_errors_arr, model_lam_grid_withlines_mmap, \
    zp_best_fit_model_fullres, zp_all_filt_flam_bestmodel, zp_bestalpha, \
    current_id, current_field, current_specz, zp, \
    zp_zerr_low, zp_zerr_up, zp_min_chi2, zp_age, zp_tau, zp_av, netsig_chosen, current_d4000, savedir_photoz)

    chk.plot_spz_fit(grism_lam_obs, grism_flam_obs, grism_ferr_obs, phot_lam, phot_fluxes_arr, phot_errors_arr, \
    model_lam_grid_withlines_mmap, zspz_best_fit_model_fullres, zspz_best_fit_model_in_objlamgrid, \
    zspz_all_filt_flam_bestmodel, zspz_bestalpha, current_id, current_field, current_specz, \
    zp_zerr_low, zp_zerr_up, zp, zspz_zerr_low, zspz_zerr_up, zspz, zspz_min_chi2, \
    zspz_age, zspz_tau, zspz_av, netsig_chosen, current_d4000, savedir_spz)

    chk.plot_grismz_fit(grism_lam_obs, grism_flam_obs, grism_ferr_obs, \
    model_lam_grid_withlines_mmap, zg_best_fit_model_fullres, zg_best_fit_model_in_objlamgrid, \
    zg_bestalpha, current_id, current_field, current_specz, zp_zerr_low, zp_zerr_up, zp, zg_zerr_low, zg_zerr_up, zg, \
    zg_min_chi2, zg_age, zg_tau, zg_av, netsig_chosen, current_d4000, savedir_grismz)

    return None

def sort_and_save(galaxies_to_plot):

    # SOrt and resave txt file accoring to field and ID
    # Sort all goods-n
    goodsn_idx = np.where(galaxies_to_plot['Field'] == 'GOODS-N')
    goodsn_sortidx = np.argsort(galaxies_to_plot['ID'][goodsn_idx])

    # Sort all goods-s
    goodss_idx = np.where(galaxies_to_plot['Field'] == 'GOODS-S')
    goodss_sortidx = np.argsort(galaxies_to_plot['ID'][goodss_idx])

    # Now save line by line:
    with open(massive_galaxies_dir + 'd4000_1p4_to_1p6_sorted.txt', 'w') as fh:
        fh.write("# All galaxies in bin:   1.4 <= D4000 < 1.6" + "\n")
        fh.write("ID        Field      zspec    zphot    zspz     NetSig    D4000   res_zphot    res_zspz    iABmag" + "\n")

        for u in range(len(goodsn_sortidx)):
            line_idx = goodsn_sortidx[u]

            current_id_to_write = str(galaxies_to_plot['ID'][goodsn_idx][line_idx])
            current_field_to_write = str(galaxies_to_plot['Field'][goodsn_idx][line_idx])
            current_specz_to_write = str(galaxies_to_plot['zspec'][goodsn_idx][line_idx])
            zp_to_write = str(galaxies_to_plot['zphot'][goodsn_idx][line_idx])
            zspz_to_write = str(galaxies_to_plot['zspz'][goodsn_idx][line_idx])
            current_netsig_to_write = str(galaxies_to_plot['NetSig'][goodsn_idx][line_idx])
            d4000_to_write = str(galaxies_to_plot['D4000'][goodsn_idx][line_idx])
            current_res_zphot_to_write = str(galaxies_to_plot['res_zphot'][goodsn_idx][line_idx])
            current_res_zspz_to_write = str(galaxies_to_plot['res_zspz'][goodsn_idx][line_idx])
            current_imag_to_write = str(galaxies_to_plot['iABmag'][goodsn_idx][line_idx])

            if len(current_id_to_write) == 5:
                current_id_to_write += ' '

            if len(current_specz_to_write) == 4:
                current_specz_to_write += '  '
            elif len(current_specz_to_write) == 5:
                current_specz_to_write += ' '

            if len(zp_to_write) == 3:
                zp_to_write += ' '

            if len(zspz_to_write) == 3:
                zspz_to_write += ' '

            if len(current_netsig_to_write) == 4:
                current_netsig_to_write += '  '
            elif len(current_netsig_to_write) == 5:
                current_netsig_to_write += ' '

            if len(d4000_to_write) == 3:
                d4000_to_write += ' '

            if current_res_zphot_to_write[0] != '-':
                current_res_zphot_to_write = '+' + current_res_zphot_to_write

            if len(current_res_zphot_to_write) == 4:
                current_res_zphot_to_write += '  '
            elif len(current_res_zphot_to_write) == 5:
                current_res_zphot_to_write += ' '

            if current_res_zspz_to_write[0] != '-':
                current_res_zspz_to_write = '+' + current_res_zspz_to_write

            if len(current_res_zspz_to_write) == 4:
                current_res_zspz_to_write += '  '
            elif len(current_res_zspz_to_write) == 5:
                current_res_zspz_to_write += ' '

            fh.write(current_id_to_write + "    ")
            fh.write(current_field_to_write + "    ")
            fh.write(current_specz_to_write + "   ")
            fh.write(zp_to_write + "     ")
            fh.write(zspz_to_write + "     ")
            fh.write(current_netsig_to_write + "    ")
            fh.write(d4000_to_write + "    ")
            fh.write(current_res_zphot_to_write + "       ")
            fh.write(current_res_zspz_to_write + "      ")
            fh.write(current_imag_to_write + "\n")

        for v in range(len(goodss_sortidx)):
            line_idx = goodss_sortidx[v]

            current_id_to_write = str(galaxies_to_plot['ID'][goodss_idx][line_idx])
            current_field_to_write = str(galaxies_to_plot['Field'][goodss_idx][line_idx])
            current_specz_to_write = str(galaxies_to_plot['zspec'][goodss_idx][line_idx])
            zp_to_write = str(galaxies_to_plot['zphot'][goodss_idx][line_idx])
            zspz_to_write = str(galaxies_to_plot['zspz'][goodss_idx][line_idx])
            current_netsig_to_write = str(galaxies_to_plot['NetSig'][goodss_idx][line_idx])
            d4000_to_write = str(galaxies_to_plot['D4000'][goodss_idx][line_idx])
            current_res_zphot_to_write = str(galaxies_to_plot['res_zphot'][goodss_idx][line_idx])
            current_res_zspz_to_write = str(galaxies_to_plot['res_zspz'][goodss_idx][line_idx])
            current_imag_to_write = str(galaxies_to_plot['iABmag'][goodss_idx][line_idx])

            if len(current_id_to_write) == 5:
                current_id_to_write += ' '

            if len(current_specz_to_write) == 4:
                current_specz_to_write += '  '
            elif len(current_specz_to_write) == 5:
                current_specz_to_write += ' '

            if len(zp_to_write) == 3:
                zp_to_write += ' '

            if len(zspz_to_write) == 3:
                zspz_to_write += ' '

            if len(current_netsig_to_write) == 4:
                current_netsig_to_write += '  '
            elif len(current_netsig_to_write) == 5:
                current_netsig_to_write += ' '

            if len(d4000_to_write) == 3:
                d4000_to_write += ' '

            if current_res_zphot_to_write[0] != '-':
                current_res_zphot_to_write = '+' + current_res_zphot_to_write

            if len(current_res_zphot_to_write) == 4:
                current_res_zphot_to_write += '  '
            elif len(current_res_zphot_to_write) == 5:
                current_res_zphot_to_write += ' '

            if current_res_zspz_to_write[0] != '-':
                current_res_zspz_to_write = '+' + current_res_zspz_to_write

            if len(current_res_zspz_to_write) == 4:
                current_res_zspz_to_write += '  '
            elif len(current_res_zspz_to_write) == 5:
                current_res_zspz_to_write += ' '

            fh.write(current_id_to_write + "    ")
            fh.write(current_field_to_write + "    ")
            fh.write(current_specz_to_write + "   ")
            fh.write(zp_to_write + "     ")
            fh.write(zspz_to_write + "     ")
            fh.write(current_netsig_to_write + "    ")
            fh.write(d4000_to_write + "    ")
            fh.write(current_res_zphot_to_write + "       ")
            fh.write(current_res_zspz_to_write + "      ")
            fh.write(current_imag_to_write + "\n")

    return None

def main():

    # ------------- Code basically copied from run_final_sample.py
    # and from single galaxy checking code.
    # --------------------------
    galaxies_to_plot = np.genfromtxt(massive_galaxies_dir + 'galaxies_with_zg_lt_0p5.txt', dtype=None, names=True, skip_header=1)
    #sort_and_save(galaxies_to_plot)
    #sys.exit()

    # first get information arrays
    ids, fields, zs_arr, zp_arr, zg_arr, zspz_arr, d4000, netsig, zp_chi2, zg_chi2, zspz_chi2, \
    zp_model_idx_arr, zg_model_idx_arr, zspz_model_idx_arr, zp_bestalpha_arr, zg_bestalpha_arr, zspz_bestalpha_arr, \
    zp_age_arr, zp_tau_arr, zp_av_arr, zg_age_arr, zg_tau_arr, zg_av_arr, zspz_age_arr, zspz_tau_arr, zspz_av_arr, \
    zp_low_bound_arr, zp_high_bound_arr, zg_low_bound_arr, zg_high_bound_arr, zspz_low_bound_arr, zspz_high_bound_arr = \
    get_arrays_for_plotting()

    # ------------------------------- Get catalog for final sample ------------------------------- #
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample.txt', dtype=None, names=True)

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

    # ------------------------------- Read in models ------------------------------- #
    total_models = 37761
    model_lam_grid_withlines_mmap = np.load(figs_data_dir + 'model_lam_grid_withlines.npy', mmap_mode='r')
    model_comp_spec_withlines_mmap = np.load(figs_data_dir + 'model_comp_spec_withlines.npy', mmap_mode='r')

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
        usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), \
        skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), \
        skip_header=3)

    # Read in Vega spectrum and get it in the appropriate forms
    vega = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/' + 'vega_reference.dat', dtype=None, \
        names=['wav', 'flam'], skip_header=7)

    # Just making sure that all returned arrays have the same length.
    # Essential since I'm doing "where" operations below.
    assert len(ids) == len(fields)
    assert len(ids) == len(zs_arr)
    assert len(ids) == len(zp_arr)
    assert len(ids) == len(zg_arr)
    assert len(ids) == len(zspz_arr)
    assert len(ids) == len(d4000)
    assert len(ids) == len(netsig)
    assert len(ids) == len(zp_chi2)
    assert len(ids) == len(zg_chi2)
    assert len(ids) == len(zspz_chi2)

    # plotting all galaxies in above plot now
    # If the low and upper bound files dont exist you'll have to 
    # run spz_photoz_grismz_comparison.py to generate them.
    for k in range(len(galaxies_to_plot)):

        current_id = int(galaxies_to_plot['ID'][k])
        current_field = str(galaxies_to_plot['Field'][k])

        if os.path.isfile(savedir_spz + current_field + '_' + str(current_id) + '_spz_fit.pdf'):
            print "At:", current_id, current_field, "   File exists. Skipping."
            continue

        print "Plotting:", current_id, current_field

        plotfit(current_id, current_field, ids, fields, zs_arr, zp_arr, zg_arr, zspz_arr, d4000, netsig, zp_chi2, zg_chi2, zspz_chi2, \
            zp_model_idx_arr, zg_model_idx_arr, zspz_model_idx_arr, zp_bestalpha_arr, zg_bestalpha_arr, zspz_bestalpha_arr, \
            zp_age_arr, zp_tau_arr, zp_av_arr, zg_age_arr, zg_tau_arr, zg_av_arr, zspz_age_arr, zspz_tau_arr, zspz_av_arr, \
            zp_low_bound_arr, zp_high_bound_arr, zg_low_bound_arr, zg_high_bound_arr, zspz_low_bound_arr, zspz_high_bound_arr, \
            final_sample, total_models, model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap, all_model_flam_mmap, \
            goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, vega)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)