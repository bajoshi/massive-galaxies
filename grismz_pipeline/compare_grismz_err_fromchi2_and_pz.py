from __future__ import division

import numpy as np

import os
import sys

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
zp_results_dir = massive_figures_dir + 'photoz_run_jan2019/'
zg_results_dir = massive_figures_dir + 'grismz_run_jan2019/'

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import spz_photoz_grismz_comparison as comp

def main():

    # ------------------------------- Get catalog for final sample ------------------------------- #
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample.txt', dtype=None, names=True)

    # ------------- Firstlight -------------
    id_arr_fl = np.load(zp_results_dir + 'firstlight_id_arr.npy')
    field_arr_fl = np.load(zp_results_dir + 'firstlight_field_arr.npy')

    zg_minchi2_fl = np.load(zg_results_dir + 'firstlight_zg_minchi2_arr.npy')

    # Get errors from chi2 map
    zg_low_bound_chi2_fl = np.load(zg_results_dir + 'firstlight_zg_zerr_low_arr.npy')
    zg_high_bound_chi2_fl = np.load(zg_results_dir + 'firstlight_zg_zerr_up_arr.npy')

    # Length checks
    assert len(id_arr_fl) == len(field_arr_fl)
    assert len(id_arr_fl) == len(zg_minchi2_fl)
    assert len(id_arr_fl) == len(zg_low_bound_chi2_fl)
    assert len(id_arr_fl) == len(zg_high_bound_chi2_fl)

    # Get errors from p(z) curve
    zg_low_bound_pz_fl = np.zeros(id_arr_fl.shape[0])
    zg_high_bound_pz_fl = np.zeros(id_arr_fl.shape[0])

    for u in range(len(id_arr_fl)):
        zg_pz = np.load(zg_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_zg_pz.npy')
        zg_zarr = np.load(zg_results_dir + str(field_arr_fl[u]) + '_' + str(id_arr_fl[u]) + '_zg_z_arr.npy')
        zg_low_bound_pz_fl[u], zg_high_bound_pz_fl[u] = comp.get_z_errors(zg_zarr, zg_pz, zg_minchi2_fl[u])

    # ------------- Jet -------------
    id_arr_jt = np.load(zp_results_dir + 'jet_id_arr.npy')
    field_arr_jt = np.load(zp_results_dir + 'jet_field_arr.npy')

    zg_minchi2_jt = np.load(zg_results_dir + 'jet_zg_minchi2_arr.npy')

    # Get errors from chi2 map
    zg_low_bound_chi2_jt = np.load(zg_results_dir + 'jet_zg_zerr_low_arr.npy')
    zg_high_bound_chi2_jt = np.load(zg_results_dir + 'jet_zg_zerr_up_arr.npy')

    # Length checks
    assert len(id_arr_jt) == len(field_arr_jt)
    assert len(id_arr_jt) == len(zg_minchi2_jt)
    assert len(id_arr_jt) == len(zg_low_bound_chi2_jt)
    assert len(id_arr_jt) == len(zg_high_bound_chi2_jt)

    # Get errors from p(z) curve
    zg_low_bound_pz_jt = np.zeros(id_arr_jt.shape[0])
    zg_high_bound_pz_jt = np.zeros(id_arr_jt.shape[0])

    for v in range(len(id_arr_jt)):
        zg_pz = np.load(zg_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_zg_pz.npy')
        zg_zarr = np.load(zg_results_dir + str(field_arr_jt[v]) + '_' + str(id_arr_jt[v]) + '_zg_z_arr.npy')
        zg_low_bound_pz_jt[v], zg_high_bound_pz_jt[v] = comp.get_z_errors(zg_zarr, zg_pz, zg_minchi2_jt[v])

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
    zg_minchi2_jt = np.delete(zg_minchi2_jt, common_indices_jt, axis=None)

    zg_low_bound_chi2_jt = np.delete(zg_low_bound_chi2_jt, common_indices_jt, axis=None)
    zg_high_bound_chi2_jt = np.delete(zg_high_bound_chi2_jt, common_indices_jt, axis=None)

    zg_low_bound_pz_jt = np.delete(zg_low_bound_pz_jt, common_indices_jt, axis=None)
    zg_high_bound_pz_jt = np.delete(zg_high_bound_pz_jt, common_indices_jt, axis=None)

    # I need to concatenate these arrays for hte purposes of looping and appending in hte loop below
    all_ids = np.concatenate((id_arr_fl, id_arr_jt))
    all_fields = np.concatenate((field_arr_fl, field_arr_jt))

    zg = np.concatenate((zg_minchi2_fl, zg_minchi2_jt))

    zg_low_bound_chi2 = np.concatenate((zg_low_bound_chi2_fl, zg_low_bound_chi2_jt))
    zg_high_bound_chi2 = np.concatenate((zg_high_bound_chi2_fl, zg_high_bound_chi2_jt))
    zg_low_bound_pz = np.concatenate((zg_low_bound_pz_fl, zg_low_bound_pz_jt))
    zg_high_bound_pz = np.concatenate((zg_high_bound_pz_fl, zg_high_bound_pz_jt))

    print "ID      Field      zg    +chi2    -chi2     +pz    -pz"

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

        # print chi2 based error
        current_zg = zg[i]
        zg_low_err_chi2 = current_zg - zg_low_bound_chi2[i]
        zg_high_err_chi2 = zg_high_bound_chi2[i] - current_zg

        # print pz based error
        zg_low_err_pz = current_zg - zg_low_bound_pz[i]
        zg_high_err_pz = zg_high_bound_pz[i] - current_zg

        current_id_to_print = str(current_id)
        if len(current_id_to_print) == 5:
            current_id_to_print += ' '

        print current_id_to_print, "  ",
        print current_field, "  "
        print "{:.2f}".format(current_zg), "  ",
        print "{:.2f}".format(zg_low_err_chi2), "  ",
        print "{:.2f}".format(zg_high_err_chi2), "  ",
        print "{:.2f}".format(zg_low_err_pz), "  ",
        print "{:.2f}".format(zg_high_err_pz)

    return None

if __name__ == '__main__':
    main()
    sys.exit()