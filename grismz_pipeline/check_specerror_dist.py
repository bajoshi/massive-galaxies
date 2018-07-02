from __future__ import division

import numpy as np

import sys
import os

import matplotlib.pyplot as plt

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import mag_hist as mh
import new_refine_grismz_gridsearch_parallel as ngp

if __name__ == '__main__':

    # Read in all output arrays
    # GOODS-N
    id_list_gn = np.load(massive_figures_dir + '/full_run/id_list_gn.npy')
    field_list_gn = np.load(massive_figures_dir + '/full_run/field_list_gn.npy')
    zgrism_list_gn = np.load(massive_figures_dir + '/full_run/zgrism_list_gn.npy')
    zgrism_lowerr_list_gn = np.load(massive_figures_dir + '/full_run/zgrism_lowerr_list_gn.npy')
    zgrism_uperr_list_gn = np.load(massive_figures_dir + '/full_run/zgrism_uperr_list_gn.npy')
    zspec_list_gn = np.load(massive_figures_dir + '/full_run/zspec_list_gn.npy')
    zphot_list_gn = np.load(massive_figures_dir + '/full_run/zphot_list_gn.npy')
    chi2_list_gn = np.load(massive_figures_dir + '/full_run/chi2_list_gn.npy')
    netsig_list_gn = np.load(massive_figures_dir + '/full_run/netsig_list_gn.npy')
    age_list_gn = np.load(massive_figures_dir + '/full_run/age_list_gn.npy')
    tau_list_gn = np.load(massive_figures_dir + '/full_run/tau_list_gn.npy')
    av_list_gn = np.load(massive_figures_dir + '/full_run/av_list_gn.npy')
    d4000_list_gn = np.load(massive_figures_dir + '/full_run/d4000_list_gn.npy')
    d4000_err_list_gn = np.load(massive_figures_dir + '/full_run/d4000_err_list_gn.npy')

    # GOODS-S
    id_list_gs = np.load(massive_figures_dir + '/full_run/id_list_gs.npy')
    field_list_gs = np.load(massive_figures_dir + '/full_run/field_list_gs.npy')
    zgrism_list_gs = np.load(massive_figures_dir + '/full_run/zgrism_list_gs.npy')
    zgrism_lowerr_list_gs = np.load(massive_figures_dir + '/full_run/zgrism_lowerr_list_gs.npy')
    zgrism_uperr_list_gs = np.load(massive_figures_dir + '/full_run/zgrism_uperr_list_gs.npy')
    zspec_list_gs = np.load(massive_figures_dir + '/full_run/zspec_list_gs.npy')
    zphot_list_gs = np.load(massive_figures_dir + '/full_run/zphot_list_gs.npy')
    chi2_list_gs = np.load(massive_figures_dir + '/full_run/chi2_list_gs.npy')
    netsig_list_gs = np.load(massive_figures_dir + '/full_run/netsig_list_gs.npy')
    age_list_gs = np.load(massive_figures_dir + '/full_run/age_list_gs.npy')
    tau_list_gs = np.load(massive_figures_dir + '/full_run/tau_list_gs.npy')
    av_list_gs = np.load(massive_figures_dir + '/full_run/av_list_gs.npy')
    d4000_list_gs = np.load(massive_figures_dir + '/full_run/d4000_list_gs.npy')
    d4000_err_list_gs = np.load(massive_figures_dir + '/full_run/d4000_err_list_gs.npy')

    # Concatenate ID and field arrays
    id_arr = np.concatenate((id_list_gn, id_list_gs))
    field_arr = np.concatenate((field_list_gn, field_list_gs))

    # Empty list for storing average errors
    err_list = []
    new_rand_err_list = []

    # Now get data and check the error
    for i in range(len(id_arr)):

        # Get data
        lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(id_arr[i], field_arr[i])

        # Get current err and append
        current_err = np.nanmean(ferr_obs/flam_obs)
        err_list.append(current_err)

    # convert to numpy array
    err_arr = np.asarray(err_list)

    # Now generate a random array based on this error array
    for j in range(len(err_arr)):
        new_rand_err_list.append(np.random.choice(err_arr))

    # convert to numpy array
    new_rand_err_arr = np.asarray(new_rand_err_list)

    # Now plot histograms for the two to compare them
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.hist(err_arr, 50, range=(0, 1), color='k', histtype='step')
    ax2.hist(new_rand_err_arr, 50, range=(0, 1), color='r', histtype='step')

    plt.show()

    sys.exit(0)