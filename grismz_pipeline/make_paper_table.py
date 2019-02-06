from __future__ import division

import numpy as np

import os
import sys

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
spz_results_dir = massive_figures_dir + 'spz_run_jan2019/'
zp_results_dir = massive_figures_dir + 'photoz_run_jan2019/'
zg_results_dir = massive_figures_dir + 'grismz_run_jan2019/'

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
#import spz_photoz_grismz_comparison as comp
from new_refine_grismz_gridsearch_parallel import get_data
import dn4000_catalog as dc

def main():

    # Read in result files for all redshifts
    # Read in arrays from Firstlight (fl) and Jet (jt) and combine them
    # ----- Firstlight -----
    id_arr_fl = np.load(zp_results_dir + 'firstlight_id_arr.npy')
    field_arr_fl = np.load(zp_results_dir + 'firstlight_field_arr.npy')
    zs_arr_fl = np.load(zp_results_dir + 'firstlight_zs_arr.npy')

    zp_arr_fl = np.zeros(id_arr_fl.shape[0])
    zg_arr_fl = np.zeros(id_arr_fl.shape[0])
    zspz_arr_fl = np.zeros(id_arr_fl.shape[0])

    # Empty error arrays
    zp_low_bound_fl = np.load(zp_results_dir + 'firstlight_zp_low_bound.npy')
    zp_high_bound_fl = np.load(zp_results_dir + 'firstlight_zp_high_bound.npy')

    zg_low_bound_fl = np.load(zg_results_dir + 'firstlight_zg_low_bound.npy')
    zg_high_bound_fl = np.load(zg_results_dir + 'firstlight_zg_high_bound.npy')

    zspz_low_bound_fl = np.load(spz_results_dir + 'firstlight_zspz_low_bound.npy')
    zspz_high_bound_fl = np.load(spz_results_dir + 'firstlight_zspz_high_bound.npy')

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

    # ----- Jet -----
    id_arr_jt = np.load(zp_results_dir + 'jet_id_arr.npy')
    field_arr_jt = np.load(zp_results_dir + 'jet_field_arr.npy')
    zs_arr_jt = np.load(zp_results_dir + 'jet_zs_arr.npy')

    zp_arr_jt = np.zeros(id_arr_jt.shape[0])
    zg_arr_jt = np.zeros(id_arr_jt.shape[0])
    zspz_arr_jt = np.zeros(id_arr_jt.shape[0])

    # Empty error arrays
    zp_low_bound_jt = np.load(zp_results_dir + 'jet_zp_low_bound.npy')
    zp_high_bound_jt = np.load(zp_results_dir + 'jet_zp_high_bound.npy')

    zg_low_bound_jt = np.load(zg_results_dir + 'jet_zg_low_bound.npy')
    zg_high_bound_jt = np.load(zg_results_dir + 'jet_zg_high_bound.npy')

    zspz_low_bound_jt = np.load(spz_results_dir + 'jet_zspz_low_bound.npy')
    zspz_high_bound_jt = np.load(spz_results_dir + 'jet_zspz_high_bound.npy')

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

    # I need to concatenate these arrays for hte purposes of looping
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

    # Comment this print statement out if out don't want to actually print this list
    print "ID       RA    DEC       Field      zspec    zphot    zg     zspz    NetSig    D4000   iABmag"

    total_galaxies = 0

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

        # Get ra, dec, i_mag
        if current_field == 'GOODS-N':
            master_cat_idx = int(np.where(pears_ncat['id'] == current_id)[0])
            current_ra = pears_ncat['pearsra'][master_cat_idx]
            current_dec = pears_ncat['pearsdec'][master_cat_idx]
            current_imag = pears_ncat['imag'][master_cat_idx]
        elif current_field == 'GOODS-S':
            master_cat_idx = int(np.where(pears_scat['id'] == current_id)[0])
            current_ra = pears_scat['pearsra'][master_cat_idx]
            current_dec = pears_scat['pearsdec'][master_cat_idx]
            current_imag = pears_scat['imag'][master_cat_idx]

        # Get errors on redshifts 
        high_zperr = zp_high_bound[i] - zp[i]
        low_zperr  = zp[i] - zp_low_bound[i]

        high_zgerr = zg_high_bound[i] - zg[i]
        low_zgerr  = zg[i] - zg_low_bound[i]

        high_zspzerr = zspz_high_bound[i] - zspz[i]
        low_zspzerr  = zspz[i] - zspz_low_bound[i]

        if d4000 >= 1.1 and d4000 < 2.0:
            # Some formatting stuff just to make it easier to read on the screen and the tex file
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
            print "{:.6f}".format(current_ra), "  ",
            print "{:.6f}".format(current_dec), "  ",
            print "{:.3f}".format(current_specz), "  ",
            print str("{:.2f}".format(zp[i])) + \
            "$\substack{+" + str("{:.2f}".format(high_zperr)) + " \\\\ -" + str("{:.2f}".format(low_zperr)) + "}$", "  ",
            print str("{:.2f}".format(zg[i])) + \
            "$\substack{+" + str("{:.2f}".format(high_zgerr)) + " \\\\ -" + str("{:.2f}".format(low_zgerr)) + "}$", "  ",
            print str("{:.2f}".format(zspz[i])) + \
            "$\substack{+" + str("{:.2f}".format(high_zspzerr)) + " \\\\ -" + str("{:.2f}".format(low_zspzerr)) + "}$", "  ",
            print current_netsig_to_print, "  ",
            print "{:.2f}".format(d4000), "  ",
            #print current_res_zphot_to_print, "     ",
            #print current_res_zspz_to_print, "    ",
            print "{:.2f}".format(current_imag)

            total_galaxies += 1

    print "Total galaxies:", total_galaxies

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)