from __future__ import division

import numpy as np

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import mag_hist as mh
import new_refine_grismz_gridsearch_parallel as ngp
import dn4000_catalog as dc

if __name__ == '__main__':

    # read master catalog to get magnitude
    pears_master_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag', 'netsig_corr'], usecols=(0,1,2,3,6))
    pears_master_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag', 'netsig_corr'], usecols=(0,1,2,3,6))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_master_ncat['dec'] = pears_master_ncat['dec'] - dec_offset_goodsn_v19

    # read in id arrays and zgrism error arrays
    id_n = np.load(massive_figures_dir + 'full_run/id_list_gn.npy')
    id_s = np.load(massive_figures_dir + 'full_run/id_list_gs.npy')

    zgrism_n = np.load(massive_figures_dir + 'full_run/zgrism_list_gn.npy')
    zgrism_s = np.load(massive_figures_dir + 'full_run/zgrism_list_gs.npy')

    zgrism_lowerr_n = np.load(massive_figures_dir + 'full_run/zgrism_lowerr_list_gn.npy')
    zgrism_lowerr_s = np.load(massive_figures_dir + 'full_run/zgrism_lowerr_list_gs.npy')

    zgrism_uperr_n = np.load(massive_figures_dir + 'full_run/zgrism_uperr_list_gn.npy')
    zgrism_uperr_s = np.load(massive_figures_dir + 'full_run/zgrism_uperr_list_gs.npy')

    # loop over all objects with D4000>1.4 and get an 
    # avg_error on the grism-z and i-band magnitude for them
    avg_error = []
    imag = []
    zgrism_list = []

    all_ids = [id_n, id_s]

    fieldcount = 0
    for id_arr in all_ids:

        if fieldcount == 0:
            fieldname = 'GOODS-N'
            zgrism_arr = zgrism_n
            zgrism_lowerr_arr = zgrism_lowerr_n
            zgrism_uperr_arr = zgrism_uperr_n
            pears_cat = pears_master_ncat

        elif fieldcount == 1:
            fieldname = 'GOODS-S'
            zgrism_arr = zgrism_s
            zgrism_lowerr_arr = zgrism_lowerr_s
            zgrism_uperr_arr = zgrism_uperr_s
            pears_cat = pears_master_scat

        for i in range(len(id_arr)):

            # Now loop over all objects with d4000 >= 1.3
            # get data and d4000 first
            current_id = id_arr[i]
            current_zgrism = float(zgrism_arr[i])
            current_zgrism_lowerr = float(current_zgrism - zgrism_lowerr_arr[i])
            current_zgrism_uperr = float(zgrism_uperr_arr[i] - current_zgrism)

            # get data and then d4000
            lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, fieldname)

            if return_code == 0:
                print current_id, fieldname
                sys.exit(0)

            lam_em = lam_obs / (1 + current_zgrism)
            flam_em = flam_obs * (1 + current_zgrism)
            ferr_em = ferr_obs * (1 + current_zgrism)

            d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

            # check d4000
            d4000_thresh = 1.4
            if d4000 < d4000_thresh:
                continue
            elif d4000 >= d4000_thresh:
                current_avg_err = (current_zgrism_lowerr + current_zgrism_uperr) / 2
                avg_error.append(current_avg_err)
                zgrism_list.append(current_zgrism)

                id_idx = np.where(pears_cat['id'] == current_id)[0]
                current_imag = pears_cat['imag'][id_idx]

                imag.append(current_imag)

        fieldcount += 1

    # Convert to numpy arrays
    avg_error = np.asarray(avg_error)
    imag = np.asarray(imag)
    zgrism_list = np.asarray(zgrism_list)

    avg_error /= (1 + zgrism_list)

    # pLot
    threepercent_idx = np.where(avg_error <= 0.03)[0]
    onepercent_idx = np.where(avg_error <= 0.01)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    print "Total objects that the code ran for", 
    print "(i.e. all within z range and also have D4000>=1.2 + they passed all cuts):", len(zgrism_n) + len(zgrism_s)
    fail_idx = np.where(avg_error >= 0.1)[0]
    print "Number of catastrophic failures:", len(fail_idx), "out of total", len(zgrism_list), 
    print "which have D4000>=", d4000_thresh, "and are within z range."

    imag_threep = imag[threepercent_idx]
    imag_onep = imag[onepercent_idx]

    imag_idx = np.where((imag >= 18) & (imag <= 26))[0]
    total_threep_within_imag_range = len(reduce(np.intersect1d, (threepercent_idx, imag_idx)))
    total_onep_within_imag_range = len(reduce(np.intersect1d, (onepercent_idx, imag_idx)))

    print total_threep_within_imag_range, "have grism redshifts accurate to within <=0.03"
    print total_onep_within_imag_range, "have grism redshifts accurate to within <=0.01"

    # get total bins and plot histogram
    ax.hist(imag_threep, 24, color='k', range=[18,26], ls='-', histtype='step', align='mid', zorder=10, \
        label=r'$\mathrm{\frac{\left<\sigma_{grism}\right>}{1+z_{grism}} \leq 0.03;}$' + r'$\mathrm{N}$' + '=' + str(total_threep_within_imag_range))
    ax.hist(imag_onep, 24, color='k', range=[18,26], ls='--', histtype='step', align='mid', zorder=10, \
        label=r'$\mathrm{\frac{\left<\sigma_{grism}\right>}{1+z_{grism}} \leq 0.01;}$' + r'$\mathrm{N}$' + '=' + str(total_onep_within_imag_range))

    ax.grid(True, color=mh.rgb_to_hex(240, 240, 240))

    ax.legend(loc='upper left', frameon=False, fontsize=13)

    ax.set_xlabel(r'$\mathrm{i_{AB}}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    ax.set_xticklabels(ax.get_xticks().tolist(), size='large')
    ax.set_yticklabels(ax.get_yticks().tolist(), size='large')

    # save figure
    fig.savefig(massive_figures_dir + 'fraction_improved_mag_hist.eps', dpi=300, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    # ----------- histogram of avg zgrism err ------------ # 

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get total bins and plot histogram
    iqr = np.std(avg_error, dtype=np.float64)
    binsize = 2*iqr*np.power(len(avg_error),-1/3)
    totalbins = np.floor((max(avg_error) - min(avg_error))/binsize)

    ax.hist(avg_error, totalbins, color='k', ls='-', histtype='step', align='mid', zorder=10)

    ax.grid(True, color=mh.rgb_to_hex(240, 240, 240))

    ax.set_xlabel(r'$\mathrm{\frac{\left<\sigma_{grism}\right>}{1+z_{grism}}}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    ax.set_xticklabels(ax.get_xticks().tolist(), size='large')
    ax.set_yticklabels(ax.get_yticks().tolist(), size='large')

    # save figure
    fig.savefig(massive_figures_dir + 'zgrism_error_hist.eps', dpi=300, bbox_inches='tight')

    sys.exit(0)
  