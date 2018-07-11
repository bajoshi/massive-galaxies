from __future__ import division

import numpy as np
from scipy.stats import gaussian_kde

import sys
import os

import matplotlib.pyplot as plt

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import mag_hist as mh
import new_refine_grismz_gridsearch_parallel as ngp
import dn4000_catalog as dc

def d4000_vs_netsig(netsig_arr, d4000_arr, d4000_err_arr):

    # Check D4000 vs netsig
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.scatter(np.log10(netsig_arr), d4000_err_arr, c=d4000_arr, vmin=1.2, vmax=2.5, s=4)

    ax.set_xlim(1,3)
    ax.set_ylim(0,1)

    fig.colorbar(cax)
    
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    return None

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
    #age_list_gn = np.load(massive_figures_dir + '/full_run/age_list_gn.npy')
    #tau_list_gn = np.load(massive_figures_dir + '/full_run/tau_list_gn.npy')
    #av_list_gn = np.load(massive_figures_dir + '/full_run/av_list_gn.npy')
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
    #age_list_gs = np.load(massive_figures_dir + '/full_run/age_list_gs.npy')
    #tau_list_gs = np.load(massive_figures_dir + '/full_run/tau_list_gs.npy')
    #av_list_gs = np.load(massive_figures_dir + '/full_run/av_list_gs.npy')
    d4000_list_gs = np.load(massive_figures_dir + '/full_run/d4000_list_gs.npy')
    d4000_err_list_gs = np.load(massive_figures_dir + '/full_run/d4000_err_list_gs.npy')

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

    # Concatenate arrays
    id_arr = np.concatenate((id_list_gn, id_list_gs))
    field_arr = np.concatenate((field_list_gn, field_list_gs))
    d4000_arr = np.concatenate((d4000_list_gn, d4000_list_gs))
    d4000_err_arr = np.concatenate((d4000_err_list_gn, d4000_err_list_gs))
    netsig_arr = np.concatenate((netsig_list_gn, netsig_list_gs))
    zgrism_arr = np.concatenate((zgrism_list_gn, zgrism_list_gs))
    zspec_arr = np.concatenate((zspec_list_gn, zspec_list_gs))
    zphot_arr = np.concatenate((zphot_list_gn, zphot_list_gs))

    # plot high d4000 and low netsig
    highd4000_lownetsig = np.where((d4000_arr > 1.6) & (netsig_arr < 100))[0]
    lowd4000_highnetsig = np.where((d4000_arr < 1.25) & (netsig_arr > 100))[0]

    """
    for i in range(len(highd4000_lownetsig)):

        current_idx = lowd4000_highnetsig[i]

        current_id = id_arr[current_idx]
        current_field = field_arr[current_idx]

        # Get data
        lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)

        current_zspec = zspec_arr[current_idx]
        current_zgrism = zgrism_arr[current_idx]

        if current_zspec != -99.0:
            redshift = current_zspec
        elif current_zspec == -99.0:
            redshift = current_zgrism

        # Now de-redshift and find D4000
        lam_em = lam_obs / (1 + redshift)
        flam_em = flam_obs * (1 + redshift)
        ferr_em = ferr_obs * (1 + redshift)

        d4000_out, d4000_out_err = dc.get_d4000(lam_em, flam_em, ferr_em)

        print "\n", "Current ID:", current_id, "in", current_field, "with NetSig:", netsig_chosen
        print "D4000:", d4000_arr[current_idx], "meas. D4000:", dc.get_d4000(lam_em, flam_em, ferr_em)
        print "At grism redshift:", current_zgrism, "with spec-z:", current_zspec
        print "Chosen redshift:", redshift, "putting 4000 break at:", (1+redshift)*4000

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(lam_obs, flam_obs)
        ax.fill_between(lam_obs, flam_obs + ferr_obs, flam_obs - ferr_obs, color='lightgray')

        plt.show()

    d4000_vs_netsig(netsig_arr, d4000_arr, d4000_err_arr)

    sys.exit(0)
    """

    # Empty list for storing average errors
    err_list = []
    d4000_list = []
    d4000_err_list = []
    new_rand_err_list = []

    # Now get data and check the error
    for i in range(len(id_arr)):

        current_id = id_arr[i]
        current_field = field_arr[i]

        # check if it is an emission line galaxy. If it is then skip
        # Be carreful changing this check. I think it is correct as it is.
        # I don think you can simply do:
        # if (int(current_id) in pirzkal2013_emline_ids) or (int(current_id) in straughn2009_emline_ids):
        #     continue
        # This can mix up north and south IDs because the IDs are not unique in north and south.
        if current_field == 'GOODS-N':
            if int(current_id) in pirzkal2013_north_emline_ids:
                print "At ID:", id_arr[i], "in", field_arr[i], "at redshift:", redshift
                print "Skipping emission line galaxy"
                continue
        elif current_field == 'GOODS-S':
            if (int(current_id) in pirzkal2013_south_emline_ids) or (int(current_id) in straughn2009_emline_ids):
                print "At ID:", id_arr[i], "in", field_arr[i], "at redshift:", redshift
                print "Skipping emission line galaxy"
                continue

        # Get data
        lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)

        # Get current err and append
        current_err = np.nanmean(ferr_obs/flam_obs)
        err_list.append(current_err)

        # append d4000 after computing using zgrism estimate
        current_zgrism = zgrism_arr[i]
        redshift = current_zgrism

        # Now de-redshift and find D4000
        lam_em = lam_obs / (1 + redshift)
        flam_em = flam_obs * (1 + redshift)
        ferr_em = ferr_obs * (1 + redshift)

        # Check that hte lambda array is not too incomplete 
        # I don't want the D4000 code extrapolating too much.
        # I'm choosing this limit to be 50A
        if np.max(lam_em) < 4200:
            print "At ID:", id_arr[i], "in", field_arr[i], "at redshift:", redshift
            print "Skipping because lambda array is incomplete by too much."
            print "i.e. the max val in rest-frame lambda is less than 4200A."
            continue

        d4000_out, d4000_out_err = dc.get_d4000(lam_em, flam_em, ferr_em)
        d4000_list.append(d4000_out)
        d4000_err_list.append(d4000_out_err)

    # convert to numpy array
    err_arr = np.asarray(err_list)
    d4000_list_arr = np.asarray(d4000_list)
    d4000_err_list_arr = np.asarray(d4000_err_list)

    print "Total galaxies in final array:", len(d4000_list_arr)

    # Now generate a random array based on this error array
    for j in range(len(err_arr)):
        new_rand_err_list.append(np.random.choice(err_arr))

    # convert to numpy array
    new_rand_err_arr = np.asarray(new_rand_err_list)

    # ------------------- plots ------------------ #
    # Check D4000 vs avg err
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # only plot the ones with high significance of measured D4000
    d4000_sig = d4000_list_arr / d4000_err_list_arr
    val_idx = np.where(d4000_sig > 5)[0]

    ax.scatter(err_arr[val_idx], d4000_list_arr[val_idx], s=3)

    ax.set_xlim(0,0.5) # -- lim for err plot
    ax.set_ylim(1,2)
    
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    sys.exit(0)

    # Now plot histograms for the two to compare them
    """
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.hist(err_arr, 50, range=(0, 1), color='k', histtype='step')
    ax2.hist(new_rand_err_arr, 50, range=(0, 1), color='r', histtype='step')

    plt.show()
    """

    sys.exit(0)