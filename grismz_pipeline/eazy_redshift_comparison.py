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

def get_plotting_arrays():

    # ------------- Firstlight -------------
    id_arr_fl = np.load(zp_results_dir + 'firstlight_id_arr.npy')
    field_arr_fl = np.load(zp_results_dir + 'firstlight_field_arr.npy')
    zs_arr_fl = np.load(zp_results_dir + 'firstlight_zs_arr.npy')

    # Length checks
    assert len(id_arr_fl) == len(field_arr_fl)
    assert len(id_arr_fl) == len(zs_arr_fl)

    zspz_arr_fl = np.zeros(id_arr_fl.shape[0])

    # Make sure you're getting the exact redshift corresponding to the peak of the p(z) curve
    for u in range(len(id_arr_fl)):
        zspz_pz = np.load(spz_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_spz_pz.npy')
        zspz_zarr = np.load(spz_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_spz_z_arr.npy')
        zspz_arr_fl[u] = zspz_zarr[np.argmax(zspz_pz)]

    # ------------- Jet -------------
    id_arr_jt = np.load(zp_results_dir + 'jet_id_arr.npy')
    field_arr_jt = np.load(zp_results_dir + 'jet_field_arr.npy')
    zs_arr_jt = np.load(zp_results_dir + 'jet_zs_arr.npy')

    # Length checks
    assert len(id_arr_jt) == len(field_arr_jt)
    assert len(id_arr_jt) == len(zs_arr_jt)

    zspz_arr_jt = np.zeros(id_arr_jt.shape[0])

    # Make sure you're getting the exact redshift corresponding to the peak of the p(z) curve
    for v in range(len(id_arr_jt)):
        zspz_pz = np.load(spz_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_spz_pz.npy')
        zspz_zarr = np.load(spz_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_spz_z_arr.npy')
        zspz_arr_jt[v] = zspz_zarr[np.argmax(zspz_pz)]

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
    zspz_arr_jt = np.delete(zspz_arr_jt, common_indices_jt, axis=None)

    # I need to concatenate these arrays for hte purposes of looping and appending in hte loop below
    all_ids = np.concatenate((id_arr_fl, id_arr_jt))
    all_fields = np.concatenate((field_arr_fl, field_arr_jt))
    zs = np.concatenate((zs_arr_fl, zs_arr_jt))
    zspz_array = np.concatenate((zspz_arr_fl, zspz_arr_jt))

    all_ids_list = []
    all_fields_list = []
    zs_list = []
    zspz_list = []
    all_d4000_list = []
    eazy_redshift_list = []

    # Read in EAZY catalogs
    eazy_ncat = np.genfromtxt(massive_galaxies_dir + 'EAZY_fromSeth/N/OUTPUT/goods_n_ubvizgrism_0.005.zout', dtype=None, names=True)
    eazy_scat = np.genfromtxt(massive_galaxies_dir + 'EAZY_fromSeth/S/OUTPUT/goods_s_ubvizgrismjhk_0.005.zout', dtype=None, names=True)

    # Now get D4000 and EAZY redshift
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

        # Match and get EAZY redshift
        if current_field == 'GOODS-N':
            id_idx = np.where(eazy_ncat['id'] == current_id)[0]
            if len(id_idx) == 1:
                id_idx = int(id_idx)
                eazy_redshift_list.append(eazy_ncat['z_2'][id_idx])
            elif len(id_idx) == 0:  # i.e., no match found
                continue
        elif current_field == 'GOODS-S':
            id_idx = np.where(eazy_scat['id'] == current_id)[0]
            if len(id_idx) == 1:
                id_idx = int(id_idx)
                eazy_redshift_list.append(eazy_scat['z_2'][id_idx])
            elif len(id_idx) == 0:  # i.e., no match found
                continue

        # Append all arrays
        all_ids_list.append(current_id)
        all_fields_list.append(current_field)
        zs_list.append(current_specz)
        zspz_list.append(zspz_array[i])
        all_d4000_list.append(d4000)

    # Convert to numpy array
    all_ids = np.array(all_ids_list)
    all_fields = np.array(all_fields_list)
    zs = np.array(zs_list)
    zspz = np.array(zspz_list)
    all_d4000 = np.array(all_d4000_list)
    eazy_redshift = np.array(eazy_redshift_list)

    return all_ids, all_fields, zs, zspz, eazy_redshift, all_d4000

def main():

    ids, fields, zs, zspz, eazy_z, d4000 = get_plotting_arrays()

    # Just making sure that all returned arrays have the same length.
    # Essential since I'm doing "where" operations below.
    assert len(ids) == len(fields)
    assert len(ids) == len(zs)
    assert len(ids) == len(eazy_z)
    assert len(ids) == len(d4000)

    # Cut on D4000
    d4000_low = 1.6
    d4000_high = 2.0
    d4000_idx = np.where((d4000 >= d4000_low) & (d4000 < d4000_high))[0]

    print "\n", "D4000 range:   ", d4000_low, "<= D4000 <", d4000_high, "\n"
    print "Galaxies within D4000 range:", len(d4000_idx)

    # Apply D4000 index
    zs = zs[d4000_idx]
    zspz = zspz[d4000_idx]
    eazy_z = eazy_z[d4000_idx]

    # Residuals
    resid_zspz = (zspz - zs) / (1 + zs)
    resid_eazy = (eazy_z - zs) / (1 + zs)

    # Make sure they are finite
    valid_idx1 = np.where(np.isfinite(resid_eazy))[0]
    valid_idx3 = np.where(np.isfinite(resid_zspz))[0]

    # Remove catastrophic failures
    # i.e. only choose the valid ones
    catas_fail_thresh = 0.1
    no_catas_fail1 = np.where(abs(resid_eazy) <= catas_fail_thresh)[0]
    no_catas_fail3 = np.where(abs(resid_zspz) <= catas_fail_thresh)[0]

    outlier_frac_eazy = len(np.where(abs(resid_eazy) > catas_fail_thresh)[0]) / len(valid_idx1)
    outlier_frac_spz = len(np.where(abs(resid_zspz) > catas_fail_thresh)[0]) / len(valid_idx3)

    # Apply indices
    valid_idx_eazy = reduce(np.intersect1d, (valid_idx1, no_catas_fail1))
    resid_eazy = resid_eazy[valid_idx_eazy]
    eazy_z = eazy_z[valid_idx_eazy]
    zs_for_eazy = zs[valid_idx_eazy]

    valid_idx_zspz = reduce(np.intersect1d, (valid_idx3, no_catas_fail3))
    resid_zspz = resid_zspz[valid_idx_zspz]
    zspz = zspz[valid_idx_zspz]
    zs_for_zspz = zs[valid_idx_zspz]

    print "\n", "Number of galaxies in EAZY-z plot:", len(valid_idx_eazy)
    print "Number of galaxies in SPZ plot:", len(valid_idx_zspz)

    # Print info
    mean_eazy = np.mean(resid_eazy)
    std_eazy = np.std(resid_eazy)
    nmad_eazy = mad_std(resid_eazy)

    mean_zspz = np.mean(resid_zspz)
    std_zspz = np.std(resid_zspz)
    nmad_zspz = mad_std(resid_zspz)

    print "\n", "Mean, std dev, and Sigma_NMAD for residuals for EAZY-z:", \
    "{:.3f}".format(mean_eazy), "{:.3f}".format(std_eazy), "{:.3f}".format(nmad_eazy)
    print "Mean, std dev, and Sigma_NMAD for residuals for SPZs:", \
    "{:.3f}".format(mean_zspz), "{:.3f}".format(std_zspz), "{:.3f}".format(nmad_zspz)

    print "\n", "Outlier fraction for EAZY-z:", outlier_frac_eazy
    print "Outlier fraction for SPZ:", outlier_frac_spz

    plot_eazy_spz_comparison()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)