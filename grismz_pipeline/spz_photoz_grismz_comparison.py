from __future__ import division

import numpy as np
from astropy.stats import mad_std
from scipy.integrate import simps
from scipy.interpolate import griddata

import os
import sys

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

def get_z_errors(zarr, pz):

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
    zpeak = zarray[np.argmax(pz_curve)]

    # zstep while iterating
    zstep = 0.005
    # Starting redshifts
    zlow = zpeak - zstep
    zhigh = zpeak + zstep

    while True:
        # Find indices and shorten the pz and z curves
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

def get_arrays_to_plot():

    # Read in arrays from Firstlight (fl) and Jet (jt) and combine them
    # ----- Firstlight -----
    id_arr_fl = np.load(zp_results_dir + 'firstlight_id_arr.npy')
    field_arr_fl = np.load(zp_results_dir + 'firstlight_field_arr.npy')
    zs_arr_fl = np.load(zp_results_dir + 'firstlight_zs_arr.npy')

    zp_arr_fl = np.zeros(id_arr_fl.shape[0])  #np.load(zp_results_dir + 'firstlight_zp_minchi2_arr.npy')
    zg_arr_fl = np.zeros(id_arr_fl.shape[0])  #np.load(zg_results_dir + 'firstlight_zg_minchi2_arr.npy')
    zspz_arr_fl = np.zeros(id_arr_fl.shape[0])  #np.load(spz_results_dir + 'firstlight_zspz_minchi2_arr.npy')

    # min chi2 values
    zp_min_chi2_fl = np.load(zp_results_dir + 'firstlight_zp_min_chi2_arr.npy')
    zg_min_chi2_fl = np.load(zg_results_dir + 'firstlight_zg_min_chi2_arr.npy')
    zspz_min_chi2_fl = np.load(spz_results_dir + 'firstlight_zspz_min_chi2_arr.npy')

    # Empty error arrays
    zp_low_bound_fl = np.zeros(id_arr_fl.shape[0])
    zp_high_bound_fl = np.zeros(id_arr_fl.shape[0])

    zg_low_bound_fl = np.zeros(id_arr_fl.shape[0])
    zg_high_bound_fl = np.zeros(id_arr_fl.shape[0])

    zspz_low_bound_fl = np.zeros(id_arr_fl.shape[0])
    zspz_high_bound_fl = np.zeros(id_arr_fl.shape[0])

    resave = False

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

    if resave:
        np.save(zp_results_dir + 'firstlight_zp_low_bound.npy', zp_low_bound_fl)
        np.save(zp_results_dir + 'firstlight_zp_high_bound.npy', zp_high_bound_fl)

        np.save(zg_results_dir + 'firstlight_zg_low_bound.npy', zg_low_bound_fl)
        np.save(zg_results_dir + 'firstlight_zg_high_bound.npy', zg_high_bound_fl)

        np.save(spz_results_dir + 'firstlight_zspz_low_bound.npy', zspz_low_bound_fl)
        np.save(spz_results_dir + 'firstlight_zspz_high_bound.npy', zspz_high_bound_fl)

    # ----- Jet ----- 
    id_arr_jt = np.load(zp_results_dir + 'jet_id_arr.npy')
    field_arr_jt = np.load(zp_results_dir + 'jet_field_arr.npy')
    zs_arr_jt = np.load(zp_results_dir + 'jet_zs_arr.npy')

    zp_arr_jt = np.zeros(id_arr_jt.shape[0])  #np.load(zp_results_dir + 'jet_zp_minchi2_arr.npy')
    zg_arr_jt = np.zeros(id_arr_jt.shape[0])  #np.load(zg_results_dir + 'jet_zg_minchi2_arr.npy')
    zspz_arr_jt = np.zeros(id_arr_jt.shape[0])  #np.load(spz_results_dir + 'jet_zspz_minchi2_arr.npy')

    # min chi2 values
    zp_min_chi2_jt = np.load(zp_results_dir + 'jet_zp_min_chi2_arr.npy')
    zg_min_chi2_jt = np.load(zg_results_dir + 'jet_zg_min_chi2_arr.npy')
    zspz_min_chi2_jt = np.load(spz_results_dir + 'jet_zspz_min_chi2_arr.npy')

    # Empty error arrays
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

    if resave:
        np.save(zp_results_dir + 'jet_zp_low_bound.npy', zp_low_bound_jt)
        np.save(zp_results_dir + 'jet_zp_high_bound.npy', zp_high_bound_jt)

        np.save(zg_results_dir + 'jet_zg_low_bound.npy', zg_low_bound_jt)
        np.save(zg_results_dir + 'jet_zg_high_bound.npy', zg_high_bound_jt)

        np.save(spz_results_dir + 'jet_zspz_low_bound.npy', zspz_low_bound_jt)
        np.save(spz_results_dir + 'jet_zspz_high_bound.npy', zspz_high_bound_jt)

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
    # you will have to re-generate the all_d4000_arr.npy and all_netsig_arr.npy arrays every time this check is changed

    # ----- Get D4000 -----
    # Now loop over all galaxies to get D4000 and netsig
    all_ids_list = []
    all_fields_list = []
    zs_list = []
    zp_list = []
    zg_list = []
    zspz_list = []
    zp_chi2_list = []
    zg_chi2_list = []
    zspz_chi2_list = []
    all_d4000_list = []
    all_netsig_list = []
    imag_list = []

    # I need to concatenate these arrays for hte purposes of looping and appending in hte loop below
    all_ids = np.concatenate((id_arr_fl, id_arr_jt))
    all_fields = np.concatenate((field_arr_fl, field_arr_jt))

    zs = np.concatenate((zs_arr_fl, zs_arr_jt))
    zp = np.concatenate((zp_arr_fl, zp_arr_jt))
    zg = np.concatenate((zg_arr_fl, zg_arr_jt))
    zspz = np.concatenate((zspz_arr_fl, zspz_arr_jt))

    zp_chi2 = np.concatenate((zp_min_chi2_fl, zp_min_chi2_jt))
    zg_chi2 = np.concatenate((zg_min_chi2_fl, zg_min_chi2_jt))
    zspz_chi2 = np.concatenate((zspz_min_chi2_fl, zspz_min_chi2_jt))

    # Comment this print statement out if out don't want to actually print this list on paper
    print "ID        Field      zspec    zphot    zg     zspz    NetSig    D4000   res_zphot    res_zspz    iABmag"

    # Read in master catalogs to get i-band mag
    # ------------------------------- Read PEARS cats ------------------------------- #
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    
    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['pearsdec'] = pears_ncat['pearsdec'] - dec_offset_goodsn_v19

    for i in range(len(all_ids)):
        current_id = all_ids[i]
        current_field = all_fields[i]

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
        zp_chi2_list.append(zp_chi2[i])
        zg_chi2_list.append(zg_chi2[i])
        zspz_chi2_list.append(zspz_chi2[i])
        all_d4000_list.append(d4000)
        all_netsig_list.append(netsig_chosen)
        imag_list.append(current_imag)

        if d4000 >= 1.4 and d4000 < 1.6:
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
            current_res_zspz = (zspz[i] - current_specz) / (1 + current_specz)

            current_res_zphot_to_print = str("{:.3f}".format(current_res_zphot))
            if current_res_zphot_to_print[0] != '-':
                current_res_zphot_to_print = '+' + current_res_zphot_to_print
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
            print current_res_zspz_to_print, "    ",
            print "{:.2f}".format(current_imag)

    return np.array(all_ids_list), np.array(all_fields_list), np.array(zs_list), np.array(zp_list), np.array(zg_list), np.array(zspz_list), \
    np.array(all_d4000_list), np.array(all_netsig_list), np.array(zp_chi2_list), np.array(zg_chi2_list), np.array(zspz_chi2_list), np.array(imag_list)

def make_plots(resid_zp, resid_zg, resid_zspz, zp, zs_for_zp, zg, zs_for_zg, zspz, zs_for_zspz, \
    mean_zphot, nmad_zphot, mean_zgrism, nmad_zgrism, mean_zspz, nmad_zspz, \
    d4000_low, d4000_high):

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
    ax1.plot(zs_for_zp, zp, 'o', markersize=3, color='k', markeredgecolor='k')
    ax2.plot(zs_for_zp, resid_zp, 'o', markersize=3, color='k', markeredgecolor='k')

    ax3.plot(zs_for_zg, zg, 'o', markersize=3, color='k', markeredgecolor='k')
    ax4.plot(zs_for_zg, resid_zg, 'o', markersize=3, color='k', markeredgecolor='k')

    ax5.plot(zs_for_zspz, zspz, 'o', markersize=3, color='k', markeredgecolor='k')
    ax6.plot(zs_for_zspz, resid_zspz, 'o', markersize=3, color='k', markeredgecolor='k')

    # Limits
    ax1.set_xlim(0.6, 1.24)
    ax1.set_ylim(0.6, 1.24)

    ax2.set_xlim(0.6, 1.24)
    ax2.set_ylim(-0.2, 0.2)

    ax3.set_xlim(0.6, 1.24)
    ax3.set_ylim(0.6, 1.24)

    ax4.set_xlim(0.6, 1.24)
    ax4.set_ylim(-0.2, 0.2)

    ax5.set_xlim(0.6, 1.24)
    ax5.set_ylim(0.6, 1.24)

    ax6.set_xlim(0.6, 1.24)
    ax6.set_ylim(-0.2, 0.2)

    # Other lines on plot
    ax2.axhline(y=0.0, ls='--', color='gray')
    ax4.axhline(y=0.0, ls='--', color='gray')
    ax6.axhline(y=0.0, ls='--', color='gray')

    linearr = np.arange(0.5, 1.3, 0.001)
    ax1.plot(linearr, linearr, ls='--', color='darkblue')
    ax3.plot(linearr, linearr, ls='--', color='darkblue')
    ax5.plot(linearr, linearr, ls='--', color='darkblue')

    ax2.axhline(y=mean_zphot, ls='-', color='darkblue')
    ax2.axhline(y=mean_zphot + nmad_zphot, ls='-', color='red')
    ax2.axhline(y=mean_zphot - nmad_zphot, ls='-', color='red')

    ax4.axhline(y=mean_zgrism, ls='-', color='darkblue')
    ax4.axhline(y=mean_zgrism + nmad_zgrism, ls='-', color='red')
    ax4.axhline(y=mean_zgrism - nmad_zgrism, ls='-', color='red')

    ax6.axhline(y=mean_zspz, ls='-', color='darkblue')
    ax6.axhline(y=mean_zspz + nmad_zspz, ls='-', color='red')
    ax6.axhline(y=mean_zspz - nmad_zspz, ls='-', color='red')

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

    ax3.text(0.05, 0.97, r'$\mathrm{N = }$' + str(len(resid_zg)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12)
    ax3.text(0.04, 0.89, r'${\left< \Delta \right>}_{\mathrm{Grism-z}} = $' + mr.convert_to_sci_not(mean_zgrism), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12)
    ax3.text(0.05, 0.79, r'$\mathrm{\sigma^{NMAD}_{Grism-z}} = $' + mr.convert_to_sci_not(nmad_zgrism), \
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

    # save figure and close
    fig.savefig(massive_figures_dir + 'spz_comp_photoz_' + \
        str(d4000_low).replace('.','p') + 'to' + str(d4000_high).replace('.','p') + '.pdf', \
        dpi=300, bbox_inches='tight')

    return None

def main():
    ids, fields, zs, zp, zg, zspz, d4000, netsig, zp_chi2, zg_chi2, zspz_chi2, imag = get_arrays_to_plot()

    # Just making sure that all returned arrays have the same length.
    # Essential since I'm doing "where" operations below.
    assert len(ids) == len(fields)
    assert len(ids) == len(zs)
    assert len(ids) == len(zp)
    assert len(ids) == len(zg)
    assert len(ids) == len(zspz)
    assert len(ids) == len(d4000)
    assert len(ids) == len(netsig)
    assert len(ids) == len(zp_chi2)
    assert len(ids) == len(zg_chi2)
    assert len(ids) == len(zspz_chi2)
    assert len(ids) == len(imag)

    # Cut on D4000
    d4000_low = 1.1
    d4000_high = 2.0
    d4000_idx = np.where((d4000 >= d4000_low) & (d4000 < d4000_high))[0]

    print "\n", "D4000 range:   ", d4000_low, "<= D4000 <", d4000_high, "\n"
    print "Galaxies within D4000 range:", len(d4000_idx)
    sys.exit(0)

    # Apply D4000 and magnitude indices
    zs = zs[d4000_idx]
    zp = zp[d4000_idx]
    zg = zg[d4000_idx]
    zspz = zspz[d4000_idx]
    
    zp_chi2 = zp_chi2[d4000_idx]
    zg_chi2 = zg_chi2[d4000_idx]
    zspz_chi2 = zspz_chi2[d4000_idx]

    netsig = netsig[d4000_idx]

    # Get residuals 
    resid_zp = (zp - zs) / (1 + zs)
    resid_zg = (zg - zs) / (1 + zs)
    resid_zspz = (zspz - zs) / (1 + zs)

    # Make sure they are finite
    valid_idx1 = np.where(np.isfinite(resid_zp))[0]
    valid_idx2 = np.where(np.isfinite(resid_zg))[0]
    valid_idx3 = np.where(np.isfinite(resid_zspz))[0]

    # Remove catastrophic failures
    # i.e. only choose the valid ones
    catas_fail_thresh = 0.1
    no_catas_fail1 = np.where(abs(resid_zp) <= catas_fail_thresh)[0]
    no_catas_fail2 = np.where(abs(resid_zg) <= catas_fail_thresh)[0]
    no_catas_fail3 = np.where(abs(resid_zspz) <= catas_fail_thresh)[0]

    outlier_frac_zp = len(np.where(abs(resid_zp) > catas_fail_thresh)[0]) / len(valid_idx1)
    outlier_frac_zg = len(np.where(abs(resid_zg) > catas_fail_thresh)[0]) / len(valid_idx2)
    outlier_frac_spz = len(np.where(abs(resid_zspz) > catas_fail_thresh)[0]) / len(valid_idx3)

    # apply cut on netsig
    netsig_thresh = 10
    valid_idx_ns = np.where(netsig > netsig_thresh)[0]
    print len(valid_idx_ns), "out of", len(netsig), "galaxies pass NetSig cut of", netsig_thresh

    # Apply indices
    valid_idx_zp = reduce(np.intersect1d, (valid_idx1, no_catas_fail1))
    resid_zp = resid_zp[valid_idx_zp]
    zp = zp[valid_idx_zp]
    zs_for_zp = zs[valid_idx_zp]

    valid_idx_zg = reduce(np.intersect1d, (valid_idx2, no_catas_fail2))
    resid_zg = resid_zg[valid_idx_zg]
    zg = zg[valid_idx_zg]
    zs_for_zg = zs[valid_idx_zg]

    valid_idx_zspz = reduce(np.intersect1d, (valid_idx3, no_catas_fail3))
    resid_zspz = resid_zspz[valid_idx_zspz]
    zspz = zspz[valid_idx_zspz]
    zs_for_zspz = zs[valid_idx_zspz]

    print "Number of galaxies in photo-z plot:", len(valid_idx_zp)
    print "Number of galaxies in grism-z plot:", len(valid_idx_zg)
    print "Number of galaxies in SPZ plot:", len(valid_idx_zspz)

    # ---------
    """
    d4000 = d4000[d4000_idx]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.scatter(d4000[valid_idx_spz], resid_zspz, c=zs)
    ax.axhline(y=0.0, ls='--')

    ax.set_ylim(-0.1, 0.1)
    fig.colorbar(cax)

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
    print "Outlier fraction for Photo-z:", outlier_frac_zp
    print "Outlier fraction for Grism-z:", outlier_frac_zg
    print "Outlier fraction for SPZ:", outlier_frac_spz

    make_plots(resid_zp, resid_zg, resid_zspz, zp, zs_for_zp, zg, zs_for_zg, zspz, zs_for_zspz, \
        mean_zphot, nmad_zphot, mean_zgrism, nmad_zgrism, mean_zspz, nmad_zspz, \
        d4000_low, d4000_high)

    return None

if __name__ == '__main__':
    main()
    sys.exit()