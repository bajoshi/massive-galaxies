from __future__ import division

import numpy as np
from astropy.stats import mad_std
from scipy.integrate import simps
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

import os
import sys
import glob

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
spz_results_dir = massive_figures_dir + 'spz_run_jan2019/'
zp_results_dir = massive_figures_dir + 'photoz_run_jan2019/'
zg_results_dir = massive_figures_dir + 'grismz_run_jan2019/'

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import mocksim_results as mr
from new_refine_grismz_gridsearch_parallel import get_data
import dn4000_catalog as dc

def get_z_errors(zarr, pz, z_minchi2):

    # Since some p(z) curves have all nans in them
    if np.isnan(pz).any():
        return -99.0, -99.0

    # Get the width of the p(z) curve when 68% of the area is covered and save the error on both sides
    # Interpolate and get finer arrays
    zarray = np.arange(0.3, 1.495, 0.005)
    pz_curve = griddata(points=zarr, values=pz, xi=zarray, method='linear')

    total_area = simps(pz_curve, zarray)

    # Now go outward from the peak of the p(z) curve and compute area again
    # Keep iterating until you reach 68% of the total area
    # Redshift limits to integrate over are brought outward each time
    # in both directions by 0.005

    # Get zpeak first
    # zpeak = zarray[np.argmax(pz_curve)]

    # zstep while iterating
    zstep = 0.005
    # Starting redshifts
    zlow = z_minchi2 - zstep
    zhigh = z_minchi2 + zstep

    while True:
        # Find indices and broaden the pz and z curves
        low_idx = np.argmin(abs(zarray - zlow))
        high_idx = np.argmin(abs(zarray - zhigh))

        new_pz_curve = pz_curve[low_idx: high_idx+1]
        new_zarray = zarray[low_idx: high_idx+1]
        
        # Compute area and fraction again
        area_now = simps(new_pz_curve, new_zarray)
        area_frac = area_now / total_area

        if area_frac >= 0.68:
            break
        else:
            zlow_prev = zlow
            zhigh_prev = zhigh
            area_frac_prev = area_frac

            zlow -= zstep
            zhigh += zstep
        
    # If the prevous step was closer to 68% area then choose that
    # This ensures that I get as close as possible to the 68% limits
    # in cases where the p(z) curve is steep and the area changes quickly.

    # In case it gets out on the first iteration then area_frac_prev 
    # will not be defined. So only do the comparison if it is defined
    # otherwise simply return the bounds at the current iteration.
    zlow_bound = zlow
    zhigh_bound = zhigh
    if 'area_frac_prev' in locals():
        if abs(area_frac_prev - 0.68) < abs(area_frac - 0.68):
            zlow_bound = zlow_prev
            zhigh_bound = zhigh_prev

    return zlow_bound, zhigh_bound

def get_arrays_to_plot_v2():

    #spz_outdir = massive_figures_dir + 'cluster_results_covmat_with_fixed_lsflength_June2019/'
    spz_outdir = massive_figures_dir + 'cluster_results_covmat_3lsfsigma_June2019/'

    # ------------------------------- Get catalog for final sample ------------------------------- #
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample.txt', dtype=None, names=True)

    # Read in master catalogs to get i-band mag
    # ------------------------------- Read PEARS cats ------------------------------- #
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    
    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['pearsdec'] = pears_ncat['pearsdec'] - dec_offset_goodsn_v19

    # Lists for storing results
    all_ids = []
    all_fields = []

    zs = []
    zp = []
    zg = []
    zspz = []

    zp_low_bound = []
    zp_high_bound = []
    zg_low_bound = []
    zg_high_bound = []
    zspz_low_bound = []
    zspz_high_bound = []

    # Loop over all result files and save to the empty lists defined above
    for fl in glob.glob(spz_outdir + 'redshift_fitting_results*.txt'):
        current_result = np.genfromtxt(fl, dtype=None, names=True, skip_header=1)

        all_ids.append(current_result['PearsID'])
        all_fields.append(current_result['Field'])

        zs.append(current_result['zspec'])
        zp.append(current_result['zp_minchi2'])
        zspz.append(current_result['zspz_minchi2'])
        zg.append(current_result['zg_minchi2'])

        zp_low_bound.append(current_result['zp_zerr_low'])
        zp_high_bound.append(current_result['zp_zerr_up'])
        zspz_low_bound.append(current_result['zspz_zerr_low'])
        zspz_high_bound.append(current_result['zspz_zerr_up'])
        zg_low_bound.append(current_result['zg_zerr_low'])
        zg_high_bound.append(current_result['zg_zerr_up'])

    # Create empty lists again and now get D4000 and netsig
    # Loop again
    all_ids_list = []
    all_fields_list = []
    zs_list = []
    zp_list = []
    zg_list = []
    zspz_list = []
    all_d4000_list = []
    all_d4000_err_list = []
    all_netsig_list = []
    imag_list = []

    for i in range(len(all_ids)):
        current_id = all_ids[i]
        current_field = all_fields[i]

        sample_idx = np.where(final_sample['pearsid'] == current_id)[0]
        if not sample_idx.size:
            continue
        if len(sample_idx) == 1:
            if final_sample['field'][sample_idx] != current_field:
                continue

        # Check z_spec quality
        # Make sure to get the spec_z qual for the exact match
        if len(sample_idx) == 2:
            sample_idx_nested = int(np.where(final_sample['field'][sample_idx] == current_field)[0])
            current_specz_qual = final_sample['zspec_qual'][sample_idx[sample_idx_nested]]
        else:
            current_specz_qual = final_sample['zspec_qual'][int(sample_idx)]

        if current_specz_qual == '1' or current_specz_qual == 'C':
            #print "Skipping", current_field, current_id, "due to spec-z quality:", current_specz_qual
            continue

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

        # Get i_mag
        if current_field == 'GOODS-N':
            master_cat_idx = int(np.where(pears_ncat['id'] == current_id)[0])
            current_imag = pears_ncat['imag'][master_cat_idx]
        elif current_field == 'GOODS-S':
            master_cat_idx = int(np.where(pears_scat['id'] == current_id)[0])
            current_imag = pears_scat['imag'][master_cat_idx]

        # Append all arrays 
        all_ids_list.append(current_id)
        all_fields_list.append(current_field)
        zs_list.append(current_specz)
        zp_list.append(zp[i])
        zg_list.append(zg[i])
        zspz_list.append(zspz[i])
        all_d4000_list.append(d4000)
        all_d4000_err_list.append(d4000_err)
        all_netsig_list.append(netsig_chosen)
        imag_list.append(current_imag)

    return np.array(all_ids_list), np.array(all_fields_list), np.array(zs_list), np.array(zp_list), np.array(zg_list), np.array(zspz_list), \
    np.array(all_d4000_list), np.array(all_d4000_err_list), np.array(all_netsig_list), np.array(imag_list)

def get_arrays_to_plot():
    # Read in arrays from Firstlight (fl) and Jet (jt) and combine them
    """
    Im' not reading in any chi2 value arrays here.
    I checked this already. Seems like making a cut on the 
    reduced chi2 does not really affect the final results too much.
    """

    # ------------------------------- Get catalog for final sample ------------------------------- #
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample.txt', dtype=None, names=True)

    # ------------- Firstlight -------------
    id_arr_fl = np.load(zp_results_dir + 'firstlight_id_arr.npy')
    field_arr_fl = np.load(zp_results_dir + 'firstlight_field_arr.npy')
    zs_arr_fl = np.load(zp_results_dir + 'firstlight_zs_arr.npy')

    zp_minchi2_fl = np.load(zp_results_dir + 'firstlight_zp_minchi2_arr.npy')
    zg_minchi2_fl = np.load(zg_results_dir + 'firstlight_zg_minchi2_arr.npy')
    zspz_minchi2_fl = np.load(spz_results_dir + 'firstlight_zspz_minchi2_arr.npy')

    # Length checks
    assert len(id_arr_fl) == len(field_arr_fl)
    assert len(id_arr_fl) == len(zs_arr_fl)
    assert len(id_arr_fl) == len(zp_minchi2_fl)
    assert len(id_arr_fl) == len(zg_minchi2_fl)
    assert len(id_arr_fl) == len(zspz_minchi2_fl)

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

    # Make sure you're getting the exact redshift corresponding to the peak of the p(z) curve
    for u in range(len(id_arr_fl)):
        zp_pz = np.load(zp_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_photoz_pz.npy')
        zp_zarr = np.load(zp_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_photoz_z_arr.npy')
        #zp_arr_fl[u] = zp_zarr[np.argmax(zp_pz)]
        zp_arr_fl[u] = zp_minchi2_fl[u]

        zg_pz = np.load(zg_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_zg_pz.npy')
        zg_zarr = np.load(zg_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_zg_z_arr.npy')
        #zg_arr_fl[u] = zg_zarr[np.argmax(zg_pz)]
        zg_arr_fl[u] = zg_minchi2_fl[u]

        zspz_pz = np.load(spz_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_spz_pz.npy')
        zspz_zarr = np.load(spz_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_spz_z_arr.npy')
        #zspz_arr_fl[u] = zspz_zarr[np.argmax(zspz_pz)]
        zspz_arr_fl[u] = zspz_minchi2_fl[u]

        # Get errors and save them to a file
        zp_low_bound_fl[u], zp_high_bound_fl[u] = get_z_errors(zp_zarr, zp_pz, zp_minchi2_fl[u])
        zg_low_bound_fl[u], zg_high_bound_fl[u] = get_z_errors(zg_zarr, zg_pz, zg_minchi2_fl[u])
        zspz_low_bound_fl[u], zspz_high_bound_fl[u] = get_z_errors(zspz_zarr, zspz_pz, zspz_minchi2_fl[u])

    # ------------- Jet -------------
    id_arr_jt = np.load(zp_results_dir + 'jet_id_arr.npy')
    field_arr_jt = np.load(zp_results_dir + 'jet_field_arr.npy')
    zs_arr_jt = np.load(zp_results_dir + 'jet_zs_arr.npy')

    zp_minchi2_jt = np.load(zp_results_dir + 'jet_zp_minchi2_arr.npy')
    zg_minchi2_jt = np.load(zg_results_dir + 'jet_zg_minchi2_arr.npy')
    zspz_minchi2_jt = np.load(spz_results_dir + 'jet_zspz_minchi2_arr.npy')

    # Length checks
    assert len(id_arr_jt) == len(field_arr_jt)
    assert len(id_arr_jt) == len(zs_arr_jt)
    assert len(id_arr_jt) == len(zp_minchi2_jt)
    assert len(id_arr_jt) == len(zg_minchi2_jt)
    assert len(id_arr_jt) == len(zspz_minchi2_jt)

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

    # Make sure you're getting the exact redshift corresponding to the peak of the p(z) curve
    for v in range(len(id_arr_jt)):
        zp_pz = np.load(zp_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_photoz_pz.npy')
        zp_zarr = np.load(zp_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_photoz_z_arr.npy')
        #zp_arr_jt[v] = zp_zarr[np.argmax(zp_pz)]
        zp_arr_jt[v] = zp_minchi2_jt[v]

        zg_pz = np.load(zg_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_zg_pz.npy')
        zg_zarr = np.load(zg_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_zg_z_arr.npy')
        #zg_arr_jt[v] = zg_zarr[np.argmax(zg_pz)]
        zg_arr_jt[v] = zg_minchi2_jt[v]

        zspz_pz = np.load(spz_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_spz_pz.npy')
        zspz_zarr = np.load(spz_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_spz_z_arr.npy')
        #zspz_arr_jt[v] = zspz_zarr[np.argmax(zspz_pz)]
        zspz_arr_jt[v] = zspz_minchi2_jt[v]

        # Get errors and save them to a file
        zp_low_bound_jt[v], zp_high_bound_jt[v] = get_z_errors(zp_zarr, zp_pz, zp_minchi2_jt[v])
        zg_low_bound_jt[v], zg_high_bound_jt[v] = get_z_errors(zg_zarr, zg_pz, zg_minchi2_jt[v])
        zspz_low_bound_jt[v], zspz_high_bound_jt[v] = get_z_errors(zspz_zarr, zspz_pz, zspz_minchi2_jt[v])

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

    zp_low_bound_jt = np.delete(zp_low_bound_jt, common_indices_jt, axis=None)
    zp_high_bound_jt = np.delete(zp_high_bound_jt, common_indices_jt, axis=None)
    zg_low_bound_jt = np.delete(zg_low_bound_jt, common_indices_jt, axis=None)
    zg_high_bound_jt = np.delete(zg_high_bound_jt, common_indices_jt, axis=None)
    zspz_low_bound_jt = np.delete(zspz_low_bound_jt, common_indices_jt, axis=None)
    zspz_high_bound_jt = np.delete(zspz_high_bound_jt, common_indices_jt, axis=None)

    # ---------------------------------------------------------------
    # Read in emission line catalogs (Pirzkal 2013 and Straughn 2009)
    pirzkal2013 = np.genfromtxt(massive_galaxies_dir + 'pirzkal_2013_emline.cat', \
        dtype=None, names=['field', 'pearsid'], skip_header=30, usecols=(0,1))
    straughn2009 = np.genfromtxt(massive_galaxies_dir + 'straughn_2009_emline.cat', \
        dtype=None, names=['pearsid'], skip_header=46, usecols=(0))

    pirzkal2013_emline_ids = np.unique(pirzkal2013['pearsid'])
    straughn2009_emline_ids = np.unique(straughn2009['pearsid'])

    straughn2009_emline_ids = straughn2009_emline_ids.astype(np.int)

    # assign north and south ids
    pirzkal2013_north_emline_ids = []
    pirzkal2013_south_emline_ids = []

    for i in  range(len(pirzkal2013_emline_ids)):
        if 'n' == pirzkal2013_emline_ids[i][0]:
            pirzkal2013_north_emline_ids.append(pirzkal2013_emline_ids[i][1:])
        elif 's' == pirzkal2013_emline_ids[i][0]:
            pirzkal2013_south_emline_ids.append(pirzkal2013_emline_ids[i][1:])

    pirzkal2013_north_emline_ids = np.asarray(pirzkal2013_north_emline_ids, dtype=np.int)
    pirzkal2013_south_emline_ids = np.asarray(pirzkal2013_south_emline_ids, dtype=np.int)

    chuck_em_line_galaxies = False

    # ----- Get D4000 -----
    # Now loop over all galaxies to get D4000 and netsig
    all_ids_list = []
    all_fields_list = []
    zs_list = []
    zp_list = []
    zg_list = []
    zspz_list = []
    all_d4000_list = []
    all_d4000_err_list = []
    all_netsig_list = []
    imag_list = []

    # I need to concatenate these arrays for hte purposes of looping and appending in hte loop below
    all_ids = np.concatenate((id_arr_fl, id_arr_jt))
    all_fields = np.concatenate((field_arr_fl, field_arr_jt))

    zs = np.concatenate((zs_arr_fl, zs_arr_jt))
    zp = np.concatenate((zp_arr_fl, zp_arr_jt))
    zg = np.concatenate((zg_arr_fl, zg_arr_jt))
    zspz = np.concatenate((zspz_arr_fl, zspz_arr_jt))

    zp_low_bound = np.concatenate((zp_low_bound_fl, zp_low_bound_jt))
    zp_high_bound = np.concatenate((zp_high_bound_fl, zp_high_bound_jt))
    zg_low_bound = np.concatenate((zg_low_bound_fl, zg_low_bound_jt))
    zg_high_bound = np.concatenate((zg_high_bound_fl, zg_high_bound_jt))
    zspz_low_bound = np.concatenate((zspz_low_bound_fl, zspz_low_bound_jt))
    zspz_high_bound = np.concatenate((zspz_high_bound_fl, zspz_high_bound_jt))

    # Read in master catalogs to get i-band mag
    # ------------------------------- Read PEARS cats ------------------------------- #
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    
    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['pearsdec'] = pears_ncat['pearsdec'] - dec_offset_goodsn_v19

    # Comment this print statement out if out don't want to actually print this list on paper
    do_print = False
    if do_print:
        print "ID      Field      zspec    zphot    zg     zspz    NetSig    D4000   res_zphot    res_zgrism    res_zspz    iABmag"

    for i in range(len(all_ids)):
        current_id = all_ids[i]
        current_field = all_fields[i]

        sample_idx = np.where(final_sample['pearsid'] == current_id)[0]
        if not sample_idx.size:
            continue
        if len(sample_idx) == 1:
            if final_sample['field'][sample_idx] != current_field:
                continue

        # Check z_spec quality
        # Make sure to get the spec_z qual for the exact match
        if len(sample_idx) == 2:
            sample_idx_nested = int(np.where(final_sample['field'][sample_idx] == current_field)[0])
            current_specz_qual = final_sample['zspec_qual'][sample_idx[sample_idx_nested]]
        else:
            current_specz_qual = final_sample['zspec_qual'][int(sample_idx)]

        if current_specz_qual == '1' or current_specz_qual == 'C':
            #print "Skipping", current_field, current_id, "due to spec-z quality:", current_specz_qual
            continue

        # check if it is an emission line galaxy. If it is then skip
        # Be carreful changing this check. I think it is correct as it is.
        # I don think you can simply do:
        # if (int(current_id) in pirzkal2013_emline_ids) or (int(current_id) in straughn2009_emline_ids):
        #     continue
        # This can mix up north and south IDs because the IDs are not unique in north and south.
        if chuck_em_line_galaxies:
            if current_field == 'GOODS-N':
                if int(current_id) in pirzkal2013_north_emline_ids:
                    continue
            elif current_field == 'GOODS-S':
                if (int(current_id) in pirzkal2013_south_emline_ids) or (int(current_id) in straughn2009_emline_ids):
                    continue

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

        # Get i_mag
        if current_field == 'GOODS-N':
            master_cat_idx = int(np.where(pears_ncat['id'] == current_id)[0])
            current_imag = pears_ncat['imag'][master_cat_idx]
        elif current_field == 'GOODS-S':
            master_cat_idx = int(np.where(pears_scat['id'] == current_id)[0])
            current_imag = pears_scat['imag'][master_cat_idx]

        # Append all arrays 
        all_ids_list.append(current_id)
        all_fields_list.append(current_field)
        zs_list.append(current_specz)
        zp_list.append(zp[i])
        zg_list.append(zg[i])
        zspz_list.append(zspz[i])
        all_d4000_list.append(d4000)
        all_d4000_err_list.append(d4000_err)
        all_netsig_list.append(netsig_chosen)
        imag_list.append(current_imag)

        if do_print:
            if d4000 > 1.1 and d4000 < 1.2:
                # Some formatting stuff just to make it easier to read on the screen
                current_id_to_print = str(current_id)
                if len(current_id_to_print) == 5:
                    current_id_to_print += ' '

                current_specz_to_print = str(current_specz)
                if len(current_specz_to_print) == 4:
                    current_specz_to_print += '  '
                elif len(current_specz_to_print) == 5:
                    current_specz_to_print += ' '

                current_netsig_to_print = str("{:.2f}".format(netsig_chosen))
                if len(current_netsig_to_print) == 5:
                    current_netsig_to_print += ' '

                current_res_zphot = (zp[i] - current_specz) / (1 + current_specz)
                current_res_zgrism = (zg[i] - current_specz) / (1 + current_specz)
                current_res_zspz = (zspz[i] - current_specz) / (1 + current_specz)

                current_res_zphot_to_print = str("{:.3f}".format(current_res_zphot))
                if current_res_zphot_to_print[0] != '-':
                    current_res_zphot_to_print = '+' + current_res_zphot_to_print
                current_res_zgrism_to_print = str("{:.3f}".format(current_res_zgrism))
                if current_res_zgrism_to_print[0] != '-':
                    current_res_zgrism_to_print = '+' + current_res_zgrism_to_print
                current_res_zspz_to_print = str("{:.3f}".format(current_res_zspz))
                if current_res_zspz_to_print[0] != '-':
                    current_res_zspz_to_print = '+' + current_res_zspz_to_print

                print current_id_to_print, "  ",
                print current_field, "  ",
                print "{:.3f}".format(current_specz), "  ",
                print "{:.2f}".format(zp[i]), "  ",
                print "{:.2f}".format(zg[i]), "  ",
                print "{:.2f}".format(zspz[i]), "  ",
                print current_netsig_to_print, "  ",
                print "{:.2f}".format(d4000), "  ",
                print current_res_zphot_to_print, "     ",
                print current_res_zgrism_to_print, "     ",
                print current_res_zspz_to_print, "    ",
                print "{:.2f}".format(current_imag)

    return np.array(all_ids_list), np.array(all_fields_list), np.array(zs_list), np.array(zp_list), np.array(zg_list), np.array(zspz_list), \
    np.array(all_d4000_list), np.array(all_d4000_err_list), np.array(all_netsig_list), np.array(imag_list)

def line_func(x, slope, intercept):
    return slope*x + intercept

def make_d4000_sig_vs_imag_plot(d4000_sig, imag):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$i_\mathrm{AB}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{\frac{D4000 - 1.0}{\sigma_{D4000}}}$', fontsize=15)

    ax.scatter(imag, d4000_sig, s=5, color='k', zorder=2)
    
    #ax.axhline(y=0.0, ls='--', color='k', zorder=2)
    ax.axhline(y=3.0, ls='--', color='k', zorder=2)

    # Now figure out at what magnitude are there more galaxies
    # with D4000_sig < 3.0 than at D4000_sig >= 3.0.
    # i.e. fainter than this magnitude you will get crappier 
    # D4000 measurements.
    imag_check_list = np.arange(18.5, 26.5, 0.1)
    for i in range(len(imag_check_list)):
        # Get magnitude to check
        imag_to_check = imag_check_list[i]

        # Get all galaxies fainter than the magnitude you're checking
        faint_idx = np.where(imag > imag_to_check)[0]

        # Of all the galaxies chosen above
        # how many have significant D4000 measurements
        # compared to how many that don't.
        high_sig_idx = np.where(d4000_sig[faint_idx] >= 3.0)[0]
        low_sig_idx = np.where(d4000_sig[faint_idx] < 3.0)[0]

        if len(low_sig_idx) / len(high_sig_idx) > 1.1:  # i.e., 10% more galaxies in the lower significance bin
            print "Magnitude cut off:", imag_to_check
            break

    # Shade region fainter than mag cut off
    xmin, xmax = ax.get_xlim()
    #ax.axvspan(imag_to_check, xmax, color='gray', alpha=0.5)
    ax.set_xlim(xmin, xmax)

    ax.set_xticklabels(ax.get_xticks().tolist(), size=12)
    ax.set_yticklabels(ax.get_yticks().tolist(), size=12)

    fig.savefig(massive_figures_dir + 'd4000_sig_vs_imag.pdf', dpi=300, bbox_inches='tight')

    return imag_to_check

def make_residual_vs_imag_plots(resid_zp, resid_zg, resid_zspz, imag_for_zp, imag_for_zg, imag_for_zspz, \
    d4000_low, d4000_high):
    
    # Define figure
    fig = plt.figure(figsize=(10, 2.5))
    gs = gridspec.GridSpec(10,28)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.25)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:, :8])
    ax2 = fig.add_subplot(gs[:, 10:18])
    ax3 = fig.add_subplot(gs[:, 20:])

    # Plot
    ax1.plot(imag_for_zp, resid_zp, 'o', markersize=2, color='k', markeredgecolor='k')
    ax2.plot(imag_for_zg, resid_zg, 'o', markersize=2, color='k', markeredgecolor='k')
    ax3.plot(imag_for_zspz, resid_zspz, 'o', markersize=2, color='k', markeredgecolor='k')

    # Lines
    ax1.axhline(y=0.0, ls='--', color='gray')
    ax2.axhline(y=0.0, ls='--', color='gray')
    ax3.axhline(y=0.0, ls='--', color='gray')

    # Limits
    ax1.set_xlim(19.5, 25.5)
    ax2.set_xlim(19.5, 25.5)
    ax3.set_xlim(19.5, 25.5)

    ax1.set_ylim(-0.35, 0.35)
    ax2.set_ylim(-0.35, 0.35)
    ax3.set_ylim(-0.35, 0.35)

    # Axis labels
    ax1.set_xlabel(r'$i_\mathrm{AB}$', fontsize=13)
    ax2.set_xlabel(r'$i_\mathrm{AB}$', fontsize=13)
    ax3.set_xlabel(r'$i_\mathrm{AB}$', fontsize=13)

    ax1.set_ylabel(r'$\mathrm{(z_{p} - z_{s}) / (1+z_{s})}$', fontsize=13)
    ax2.set_ylabel(r'$\mathrm{(z_{g} - z_{s}) / (1+z_{s})}$', fontsize=13)
    ax3.set_ylabel(r'$\mathrm{(z_{spz} - z_{s}) / (1+z_{s})}$', fontsize=13)

    # Tick Labels
    ax1.set_xticklabels(ax1.get_xticks().tolist(), size=10)
    ax1.set_yticklabels(ax1.get_yticks().tolist(), size=10)

    ax2.set_xticklabels(ax2.get_xticks().tolist(), size=10)
    ax2.set_yticklabels(ax2.get_yticks().tolist(), size=10)

    ax3.set_xticklabels(ax3.get_xticks().tolist(), size=10)
    ax3.set_yticklabels(ax3.get_yticks().tolist(), size=10)

    # Save figure
    fig.savefig(massive_figures_dir + 'residuals_vs_mag' + \
        str(d4000_low).replace('.','p') + 'to' + str(d4000_high).replace('.','p') + '.pdf', \
        dpi=300, bbox_inches='tight')

    return None

def make_plots(resid_zp, resid_zg, resid_zspz, zp, zs_for_zp, zg, zs_for_zg, zspz, zs_for_zspz, \
    mean_zphot, nmad_zphot, mean_zgrism, nmad_zgrism, mean_zspz, nmad_zspz, \
    d4000_low, d4000_high, outlier_idx_zp, outlier_idx_zg, outlier_idx_zspz, \
    outlier_frac_zp, outlier_frac_zg, outlier_frac_zspz):

    # Define figure
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(10,28)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.25)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:7, :8])
    ax2 = fig.add_subplot(gs[7:, :8])

    ax3 = fig.add_subplot(gs[:7, 10:18])
    ax4 = fig.add_subplot(gs[7:, 10:18])

    ax5 = fig.add_subplot(gs[:7, 20:])
    ax6 = fig.add_subplot(gs[7:, 20:])

    # Plot stuff
    ax1.plot(zs_for_zp, zp, 'o', markersize=2, color='k', markeredgecolor='k')
    ax1.scatter(zs_for_zp[outlier_idx_zp], zp[outlier_idx_zp], s=20, facecolor='white', edgecolors='gray', zorder=5)
    ax2.plot(zs_for_zp, resid_zp, 'o', markersize=2, color='k', markeredgecolor='k')
    ax2.scatter(zs_for_zp[outlier_idx_zp], resid_zp[outlier_idx_zp], s=20, facecolor='white', edgecolors='gray', zorder=5)

    ax3.plot(zs_for_zg, zg, 'o', markersize=2, color='k', markeredgecolor='k')
    ax3.scatter(zs_for_zg[outlier_idx_zg], zg[outlier_idx_zg], s=20, facecolor='white', edgecolors='gray', zorder=5)
    ax4.plot(zs_for_zg, resid_zg, 'o', markersize=2, color='k', markeredgecolor='k')
    ax4.scatter(zs_for_zg[outlier_idx_zg], resid_zg[outlier_idx_zg], s=20, facecolor='white', edgecolors='gray', zorder=5)

    ax5.plot(zs_for_zspz, zspz, 'o', markersize=2, color='k', markeredgecolor='k')
    ax5.scatter(zs_for_zspz[outlier_idx_zspz], zspz[outlier_idx_zspz], s=20, facecolor='white', edgecolors='gray', zorder=5)
    ax6.plot(zs_for_zspz, resid_zspz, 'o', markersize=2, color='k', markeredgecolor='k')
    ax6.scatter(zs_for_zspz[outlier_idx_zspz], resid_zspz[outlier_idx_zspz], s=20, facecolor='white', edgecolors='gray', zorder=5)

    # Limits
    ax1.set_xlim(0.6, 1.24)
    ax1.set_ylim(0.6, 1.24)

    ax2.set_xlim(0.6, 1.24)
    ax2.set_ylim(-0.15, 0.15)

    ax3.set_xlim(0.6, 1.24)
    ax3.set_ylim(0.6, 1.24)

    ax4.set_xlim(0.6, 1.24)
    ax4.set_ylim(-0.15, 0.15)

    ax5.set_xlim(0.6, 1.24)
    ax5.set_ylim(0.6, 1.24)

    ax6.set_xlim(0.6, 1.24)
    ax6.set_ylim(-0.15, 0.15)

    # Other lines on plot
    ax2.axhline(y=0.0, ls='-', color='gray')
    ax4.axhline(y=0.0, ls='-', color='gray')
    ax6.axhline(y=0.0, ls='-', color='gray')

    # do the fit with scipy
    popt_zp, pcov_zp = curve_fit(line_func, zs_for_zp, zp, p0=[1.0, 0.6])
    popt_zg, pcov_zg = curve_fit(line_func, zs_for_zg, zg, p0=[1.0, 0.6])
    popt_zspz, pcov_zspz = curve_fit(line_func, zs_for_zspz, zspz, p0=[1.0, 0.6])

    # plot line fit
    x_plot = np.arange(0.2,1.5,0.01)

    zp_mean_line = line_func(x_plot, popt_zp[0], popt_zp[1])
    zg_mean_line = line_func(x_plot, popt_zg[0], popt_zg[1])
    zspz_mean_line = line_func(x_plot, popt_zspz[0], popt_zspz[1])

    ax1.plot(x_plot, x_plot, '-', color='gray')
    ax1.plot(x_plot, zp_mean_line, '--', color='darkblue', lw=1)
    ax1.plot(x_plot, (1+nmad_zphot)*popt_zp[0]*x_plot + nmad_zphot + popt_zp[1], ls='--', color='red', lw=1)
    ax1.plot(x_plot, (1-nmad_zphot)*popt_zp[0]*x_plot - nmad_zphot + popt_zp[1], ls='--', color='red', lw=1)

    ax3.plot(x_plot, x_plot, '-', color='gray')
    ax3.plot(x_plot, zg_mean_line, '--', color='darkblue', lw=1)
    ax3.plot(x_plot, (1+nmad_zgrism)*popt_zg[0]*x_plot + nmad_zgrism + popt_zg[1], ls='--', color='red', lw=1)
    ax3.plot(x_plot, (1-nmad_zgrism)*popt_zg[0]*x_plot - nmad_zgrism + popt_zg[1], ls='--', color='red', lw=1)

    ax5.plot(x_plot, x_plot, '-', color='gray')
    ax5.plot(x_plot, zspz_mean_line, '--', color='darkblue', lw=1)
    ax5.plot(x_plot, (1+nmad_zspz)*popt_zspz[0]*x_plot + nmad_zspz + popt_zspz[1], ls='--', color='red', lw=1)
    ax5.plot(x_plot, (1-nmad_zspz)*popt_zspz[0]*x_plot - nmad_zspz + popt_zspz[1], ls='--', color='red', lw=1)

    ax2.axhline(y=mean_zphot, ls='--', color='darkblue', lw=1)
    ax2.axhline(y=mean_zphot + nmad_zphot, ls='--', color='red', lw=1)
    ax2.axhline(y=mean_zphot - nmad_zphot, ls='--', color='red', lw=1)

    ax4.axhline(y=mean_zgrism, ls='--', color='darkblue', lw=1)
    ax4.axhline(y=mean_zgrism + nmad_zgrism, ls='--', color='red', lw=1)
    ax4.axhline(y=mean_zgrism - nmad_zgrism, ls='--', color='red', lw=1)

    ax6.axhline(y=mean_zspz, ls='--', color='darkblue', lw=1)
    ax6.axhline(y=mean_zspz + nmad_zspz, ls='--', color='red', lw=1)
    ax6.axhline(y=mean_zspz - nmad_zspz, ls='--', color='red', lw=1)

    # Make tick labels larger
    ax1.set_xticklabels(ax1.get_xticks().tolist(), size=10)
    ax1.set_yticklabels(ax1.get_yticks().tolist(), size=10)

    ax2.set_xticklabels(ax2.get_xticks().tolist(), size=10)
    ax2.set_yticklabels(ax2.get_yticks().tolist(), size=10)

    ax3.set_xticklabels(ax3.get_xticks().tolist(), size=10)
    ax3.set_yticklabels(ax3.get_yticks().tolist(), size=10)

    ax4.set_xticklabels(ax4.get_xticks().tolist(), size=10)
    ax4.set_yticklabels(ax4.get_yticks().tolist(), size=10)

    ax5.set_xticklabels(ax5.get_xticks().tolist(), size=10)
    ax5.set_yticklabels(ax5.get_yticks().tolist(), size=10)

    ax6.set_xticklabels(ax6.get_xticks().tolist(), size=10)
    ax6.set_yticklabels(ax6.get_yticks().tolist(), size=10)

    # Get rid of Xaxis tick labels on top subplot
    ax1.set_xticklabels([])
    ax3.set_xticklabels([])
    ax5.set_xticklabels([])

    # Minor ticks
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    ax4.minorticks_on()
    ax5.minorticks_on()
    ax6.minorticks_on()

    # Axis labels
    ax1.set_ylabel(r'$\mathrm{z_{p}}$', fontsize=13)
    ax2.set_xlabel(r'$\mathrm{z_{s}}$', fontsize=13)
    ax2.set_ylabel(r'$\mathrm{(z_{p} - z_{s}) / (1+z_{s})}$', fontsize=13)

    ax3.set_ylabel(r'$\mathrm{z_{g}}$', fontsize=13)
    ax4.set_xlabel(r'$\mathrm{z_{s}}$', fontsize=13)
    ax4.set_ylabel(r'$\mathrm{(z_{g} - z_{s}) / (1+z_{s})}$', fontsize=13)

    ax5.set_ylabel(r'$\mathrm{z_{spz}}$', fontsize=13)
    ax6.set_xlabel(r'$\mathrm{z_{s}}$', fontsize=13)
    ax6.set_ylabel(r'$\mathrm{(z_{spz} - z_{s}) / (1+z_{s})}$', fontsize=13)

    # Text on figures
    # print D4000 range
    ax3.text(0.34, 0.11, "{:.1f}".format(d4000_low) + r"$\, \leq \mathrm{D4000} < \,$" + "{:.1f}".format(d4000_high), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=15)

    # print N, mean, nmad, outlier frac
    ax1.text(0.05, 0.97, r'$\mathrm{N = }$' + str(len(resid_zp)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)
    ax1.text(0.04, 0.89, r'${\left< \Delta \right>}_{\mathrm{Photo-z}} = $' + mr.convert_to_sci_not(mean_zphot), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)
    ax1.text(0.05, 0.79, r'$\mathrm{\sigma^{NMAD}_{Photo-z}} = $' + mr.convert_to_sci_not(nmad_zphot), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)
    ax1.text(0.05, 0.7, r'$\mathrm{Out\ frac\, =\, }$' + str("{:.2f}".format(outlier_frac_zp)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)

    ax3.text(0.05, 0.97, r'$\mathrm{N = }$' + str(len(resid_zg)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12)
    ax3.text(0.04, 0.89, r'${\left< \Delta \right>}_{\mathrm{Grism-z}} = $' + mr.convert_to_sci_not(mean_zgrism), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12)
    ax3.text(0.05, 0.79, r'$\mathrm{\sigma^{NMAD}_{Grism-z}} = $' + mr.convert_to_sci_not(nmad_zgrism), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12)
    ax3.text(0.05, 0.7, r'$\mathrm{Out\ frac\, =\, }$' + str("{:.2f}".format(outlier_frac_zg)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12)

    ax5.text(0.05, 0.97, r'$\mathrm{N = }$' + str(len(resid_zspz)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax5.transAxes, color='k', size=12)
    ax5.text(0.04, 0.89, r'${\left< \Delta \right>}_{\mathrm{SPZ}} = $' + mr.convert_to_sci_not(mean_zspz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax5.transAxes, color='k', size=12)
    ax5.text(0.05, 0.79, r'$\mathrm{\sigma^{NMAD}_{SPZ}} = $' + mr.convert_to_sci_not(nmad_zspz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax5.transAxes, color='k', size=12)
    ax5.text(0.05, 0.7, r'$\mathrm{Out\ frac\, =\, }$' + str("{:.2f}".format(outlier_frac_zspz)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax5.transAxes, color='k', size=12)

    # save figure and close
    fig.savefig(massive_figures_dir + 'spz_comp_photoz_revised_' + \
        str(d4000_low).replace('.','p') + 'to' + str(d4000_high).replace('.','p') + '.pdf', \
        dpi=300, bbox_inches='tight')

    return None

def main():
    ids, fields, zs, zp, zg, zspz, d4000, d4000_err, netsig, imag = get_arrays_to_plot_v2()

    # Just making sure that all returned arrays have the same length.
    # Essential since I'm doing "where" operations below.
    assert len(ids) == len(fields)
    assert len(ids) == len(zs)
    assert len(ids) == len(zp)
    assert len(ids) == len(zg)
    assert len(ids) == len(zspz)
    assert len(ids) == len(d4000)
    assert len(ids) == len(d4000_err)
    assert len(ids) == len(netsig)
    assert len(ids) == len(imag)

    # Cut on D4000
    d4000_low = 1.1
    d4000_high = 2.0
    d4000_idx = np.where((d4000 >= d4000_low) & (d4000 < d4000_high) & (d4000_err < 0.5))[0]

    print "\n", "D4000 range:   ", d4000_low, "<= D4000 <", d4000_high, "\n"
    print "Galaxies within D4000 range:", len(d4000_idx)

    # Cut on magnitude
    # PUt the limit at 26.0 to include the entire sample with no mag cut
    mag_idx = np.where(imag <= 26.0)[0]
    print "Galaxies that are brighter than 24 mag:", len(mag_idx)
    d4000_idx = reduce(np.intersect1d, (d4000_idx, mag_idx))

    # Apply D4000 and magnitude indices
    zs = zs[d4000_idx]
    zp = zp[d4000_idx]
    zg = zg[d4000_idx]
    zspz = zspz[d4000_idx]

    d4000 = d4000[d4000_idx]
    d4000_err = d4000_err[d4000_idx]
    netsig = netsig[d4000_idx]
    imag = imag[d4000_idx]

    d4000_resid = (d4000 - 1.0) / d4000_err

    #mag_cutoff = make_d4000_sig_vs_imag_plot(d4000_resid, imag)
    #sys.exit()

    # Get residuals 
    resid_zp = (zp - zs) / (1 + zs)
    resid_zg = (zg - zs) / (1 + zs)
    resid_zspz = (zspz - zs) / (1 + zs)

    # Estimate accurate fraction for paper
    # This is only to be done for the full D4000 range
    do_frac = True
    if do_frac:
        print "Based on SPZ:"
        two_percent_idx = np.where(abs(resid_zspz) <= 0.02)[0]
        print len(two_percent_idx), len(resid_zspz)
        f_acc = len(two_percent_idx) / len(resid_zspz)
        print len(two_percent_idx), "out of", len(resid_zspz), "galaxies have SPZ accuracy at 2% or better."
        print "Therefore, fraction of SPZ galaxies with accuracy at 2% or better:", f_acc

        print "Based only on grism-z:"
        tp_idx_grism = np.where(abs(resid_zg) <= 0.02)[0]
        f_acc_g = len(tp_idx_grism) / len(resid_zg)
        print len(tp_idx_grism), "out of", len(resid_zg), "galaxies have grism-z accuracy at 2% or better."
        print "Therefore, fraction of grism-z galaxies with accuracy at 2% or better:", f_acc_g
        sys.exit()

    # Make sure they are finite
    valid_idx1 = np.where(np.isfinite(resid_zp))[0]
    valid_idx2 = np.where(np.isfinite(resid_zg))[0]
    valid_idx3 = np.where(np.isfinite(resid_zspz))[0]

    # apply cut on netsig
    netsig_thresh = 10
    valid_idx_ns = np.where(netsig > netsig_thresh)[0]
    print len(valid_idx_ns), "out of", len(netsig), "galaxies pass NetSig cut of", netsig_thresh

    # Apply indices
    #valid_idx_zp = reduce(np.intersect1d, (valid_idx1, no_catas_fail1))
    resid_zp = resid_zp[valid_idx1]
    zp = zp[valid_idx1]
    zs_for_zp = zs[valid_idx1]

    #valid_idx_zg = reduce(np.intersect1d, (valid_idx2, no_catas_fail2))
    resid_zg = resid_zg[valid_idx2]
    zg = zg[valid_idx2]
    zs_for_zg = zs[valid_idx2]

    #valid_idx_zspz = reduce(np.intersect1d, (valid_idx3, no_catas_fail3))
    resid_zspz = resid_zspz[valid_idx3]
    zspz = zspz[valid_idx3]
    zs_for_zspz = zs[valid_idx3]

    print "Number of galaxies in photo-z plot:", len(valid_idx1)
    print "Number of galaxies in grism-z plot:", len(valid_idx2)
    print "Number of galaxies in SPZ plot:", len(valid_idx3)

    # ---------
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(d4000_resid[valid_idx3], resid_zspz, s=4, color='k')
    ax.axhline(y=0.0, ls='--')
    ax.set_ylim(-0.1, 0.1)
    plt.show()
    sys.exit(0)
    """

    # Print info
    mean_zphot = np.mean(resid_zp)
    std_zphot = np.std(resid_zp)
    nmad_zphot = mad_std(resid_zp)

    mean_zgrism = np.mean(resid_zg)
    std_zgrism = np.std(resid_zg)
    nmad_zgrism = mad_std(resid_zg)

    mean_zspz = np.mean(resid_zspz)
    std_zspz = np.std(resid_zspz)
    nmad_zspz = mad_std(resid_zspz)

    print "Mean, std dev, and Sigma_NMAD for residuals for Photo-z:", \
    "{:.3f}".format(mean_zphot), "{:.3f}".format(std_zphot), "{:.3f}".format(nmad_zphot)
    print "Mean, std dev, and Sigma_NMAD for residuals for Grism-z:", \
    "{:.3f}".format(mean_zgrism), "{:.3f}".format(std_zgrism), "{:.3f}".format(nmad_zgrism)
    print "Mean, std dev, and Sigma_NMAD for residuals for SPZs:", \
    "{:.3f}".format(mean_zspz), "{:.3f}".format(std_zspz), "{:.3f}".format(nmad_zspz)

    # Compute catastrophic failures
    # i.e., How many galaxies are outside +-3-sigma given the sigma above?
    # Photo-z
    outlier_idx_zp = np.where(abs(resid_zp - mean_zphot) > 3*nmad_zphot)[0]
    outlier_idx_zg = np.where(abs(resid_zg - mean_zgrism) > 3*nmad_zgrism)[0]
    outlier_idx_zspz = np.where(abs(resid_zspz - mean_zspz) > 3*nmad_zspz)[0]

    outlier_frac_zp = len(outlier_idx_zp) / len(resid_zp)
    outlier_frac_zg = len(outlier_idx_zg) / len(resid_zg)
    outlier_frac_zspz = len(outlier_idx_zspz) / len(resid_zspz)

    print "Outlier fraction for Photo-z:", outlier_frac_zp
    print "Outlier fraction for Grism-z:", outlier_frac_zg
    print "Outlier fraction for SPZ:", outlier_frac_zspz

    make_plots(resid_zp, resid_zg, resid_zspz, zp, zs_for_zp, zg, zs_for_zg, zspz, zs_for_zspz, \
        mean_zphot, nmad_zphot, mean_zgrism, nmad_zgrism, mean_zspz, nmad_zspz, \
        d4000_low, d4000_high, outlier_idx_zp, outlier_idx_zg, outlier_idx_zspz, \
        outlier_frac_zp, outlier_frac_zg, outlier_frac_zspz)

    sys.exit(0)

    # Now create the residual vs mag plots
    imag_for_zp = imag[valid_idx1]
    imag_for_zg = imag[valid_idx2]
    imag_for_zspz = imag[valid_idx3]

    make_residual_vs_imag_plots(resid_zp, resid_zg, resid_zspz, imag_for_zp, imag_for_zg, imag_for_zspz,\
        d4000_low, d4000_high)

    return None

if __name__ == '__main__':
    main()
    sys.exit()