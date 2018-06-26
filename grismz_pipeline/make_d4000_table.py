from __future__ import division

import numpy as np

import sys
import os

import matplotlib.pyplot as plt

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
figs_dir = home + "/Desktop/FIGS/"

if __name__ == '__main__':

    # Read in arrays
    id_arr_gn = np.load(massive_figures_dir + 'full_run/id_list_gn.npy')
    field_arr_gn = np.load(massive_figures_dir + 'full_run/field_list_gn.npy')
    zgrism_arr_gn = np.load(massive_figures_dir + 'full_run/zgrism_list_gn.npy')
    zgrism_lowerr_arr_gn = np.load(massive_figures_dir + 'full_run/zgrism_lowerr_list_gn.npy')
    zgrism_uperr_arr_gn = np.load(massive_figures_dir + 'full_run/zgrism_uperr_list_gn.npy')
    zspec_arr_gn = np.load(massive_figures_dir + 'full_run/zspec_list_gn.npy')
    zphot_arr_gn = np.load(massive_figures_dir + 'full_run/zphot_list_gn.npy')
    chi2_arr_gn = np.load(massive_figures_dir + 'full_run/chi2_list_gn.npy')
    netsig_arr_gn = np.load(massive_figures_dir + 'full_run/netsig_list_gn.npy')
    d4000_arr_gn = np.load(massive_figures_dir + 'full_run/d4000_list_gn.npy')
    d4000_err_arr_gn = np.load(massive_figures_dir + 'full_run/d4000_err_list_gn.npy')

    id_arr_gs = np.load(massive_figures_dir + 'full_run/id_list_gs.npy')
    field_arr_gs = np.load(massive_figures_dir + 'full_run/field_list_gs.npy')
    zgrism_arr_gs = np.load(massive_figures_dir + 'full_run/zgrism_list_gs.npy')
    zgrism_lowerr_arr_gs = np.load(massive_figures_dir + 'full_run/zgrism_lowerr_list_gs.npy')
    zgrism_uperr_arr_gs = np.load(massive_figures_dir + 'full_run/zgrism_uperr_list_gs.npy')
    zspec_arr_gs = np.load(massive_figures_dir + 'full_run/zspec_list_gs.npy')
    zphot_arr_gs = np.load(massive_figures_dir + 'full_run/zphot_list_gs.npy')
    chi2_arr_gs = np.load(massive_figures_dir + 'full_run/chi2_list_gs.npy')
    netsig_arr_gs = np.load(massive_figures_dir + 'full_run/netsig_list_gs.npy')
    d4000_arr_gs = np.load(massive_figures_dir + 'full_run/d4000_list_gs.npy')
    d4000_err_arr_gs = np.load(massive_figures_dir + 'full_run/d4000_err_list_gs.npy')

    # Concatenate
    id_arr = np.concatenate((id_arr_gn, id_arr_gs))
    field_arr = np.concatenate((field_arr_gn, field_arr_gs))
    zgrism_arr = np.concatenate((zgrism_arr_gn, zgrism_arr_gs))
    zgrism_lowerr_arr = np.concatenate((zgrism_lowerr_arr_gn, zgrism_lowerr_arr_gs))
    zgrism_uperr_arr = np.concatenate((zgrism_uperr_arr_gn, zgrism_uperr_arr_gs))
    zspec_arr = np.concatenate((zspec_arr_gn, zspec_arr_gs))
    zphot_arr = np.concatenate((zphot_arr_gn, zphot_arr_gs))
    chi2_arr = np.concatenate((chi2_arr_gn, chi2_arr_gs))
    netsig_arr = np.concatenate((netsig_arr_gn, netsig_arr_gs))
    d4000_arr = np.concatenate((d4000_arr_gn, d4000_arr_gs))
    d4000_err_arr = np.concatenate((d4000_err_arr_gn, d4000_err_arr_gs))

    # Read in matched files to get other info
    cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    allcats = [cat_n, cat_s]

    # Now loop and print
    for i in range(500, len(id_arr)):  # Change this to something like range(600, len(id_arr)) to get goods-s entries

        current_id = id_arr[i]
        current_field = field_arr[i]

        current_zspec = zspec_arr[i]
        current_zphot = zphot_arr[i]
        current_zgrism = zgrism_arr[i]
        current_zgrism_lowerr = zgrism_lowerr_arr[i]
        current_zgrism_uperr = zgrism_uperr_arr[i]
        current_netsig = netsig_arr[i]
        current_d4000 = d4000_arr[i]
        current_d4000_err = d4000_err_arr[i]
        current_chi2red = chi2_arr[i]

        if current_field == 'GOODS-N':
            cat = cat_n
        elif current_field == 'GOODS-S':
            cat = cat_s

        id_idx = np.where(cat['pearsid'] == current_id)[0]

        current_ra = float(cat['pearsra'][id_idx])
        current_dec = float(cat['pearsdec'][id_idx])

        current_zphot_l68 = float(cat['zphot_l68'][id_idx])
        current_zphot_u68 = float(cat['zphot_u68'][id_idx])

        if current_field == 'GOODS-N':
            print current_id, '&', current_field, '&',
            print current_ra, '&', current_dec, '&',
            print current_zspec, '&',
            print '$', '{:.3}'.format(current_zphot), '&',
            print '$', '{:.3}'.format(current_zgrism), \
            '\substack{+', '{:.3}'.format(current_zgrism_lowerr), '\\\ -', "{:.3}".format(current_zgrism_uperr), '}$', '&',
            print '{:.2f}'.format(current_netsig), '&',
            print '$', '{:.2}'.format(current_d4000), '\pm', '{:.2}'.format(current_d4000_err), '$', '&',
            print '{:.2}'.format(current_chi2red), '\\\ '
        else:
            print current_id, '&', current_field, '&',
            print current_ra, '&', current_dec, '&',
            print current_zspec, '&',
            print '$', '{:.3}'.format(current_zphot), \
            '\substack{+', '{:.3}'.format(current_zphot_l68), '\\\ -', "{:.3}".format(current_zphot_u68), '}$', '&',
            print '$', '{:.3}'.format(current_zgrism), \
            '\substack{+', '{:.3}'.format(current_zgrism_lowerr), '\\\ -', "{:.3}".format(current_zgrism_uperr), '}$', '&',
            print '{:.2f}'.format(current_netsig), '&',
            print '$', '{:.2}'.format(current_d4000), '\pm', '{:.2}'.format(current_d4000_err), '$', '&',
            print '{:.2}'.format(current_chi2red), '\\\ '

    sys.exit(0)