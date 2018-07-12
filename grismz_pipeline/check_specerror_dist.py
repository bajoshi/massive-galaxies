from __future__ import division

import numpy as np
import scipy
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

def get_err_and_d4000_arr(id_arr, field_arr, zgrism_arr, \
    pirzkal2013_north_emline_ids, pirzkal2013_south_emline_ids, straughn2009_emline_ids):

    # Empty list for storing average errors
    err_list = []
    d4000_list = []
    d4000_err_list = []

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

    return err_arr, d4000_list_arr, d4000_err_list_arr

def get_resampled_d4000_err(d4000_list_arr, d4000_err_list_arr, err_arr):

    # only plot the ones with high significance of measured D4000
    d4000_sig = d4000_list_arr / d4000_err_list_arr
    val_idx = np.where(d4000_sig >= 5)[0]

    # min and max values for plot and for kernel density estimate
    xmin = 0.0
    xmax = 0.5
    ymin = 1.0
    ymax = 2.0

    # first clip x and y arrays to the specified min and max values
    x = err_arr[val_idx]
    y = d4000_list_arr[val_idx]
    x_idx = np.where((x>=xmin) & (x<=xmax))[0]
    y_idx = np.where((y>=ymin) & (y<=ymax))[0]
    xy_idx = reduce(np.intersect1d, (x_idx, y_idx))
    x = x[xy_idx]
    y = y[xy_idx]

    # now use scipy gaussian kde
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values, bw_method=0.5)

    # resample points
    resample_arr = kernel.resample(size=50000)
    err_resamp = resample_arr[0]
    d4000_resamp = resample_arr[1]

    return err_resamp, d4000_resamp

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

    #err_arr, d4000_list_arr, d4000_err_list_arr = get_err_and_d4000_arr(id_arr, field_arr, zgrism_arr, \
    #pirzkal2013_north_emline_ids, pirzkal2013_south_emline_ids, straughn2009_emline_ids)
    #np.save(massive_figures_dir + 'avg_fobs_errors.npy', err_arr)
    #np.save(massive_figures_dir + 'd4000_list_arr.npy', d4000_list_arr)
    #np.save(massive_figures_dir + 'd4000_err_list_arr.npy', d4000_err_list_arr)

    err_arr = np.load(massive_figures_dir + 'avg_fobs_errors.npy')
    d4000_list_arr = np.load(massive_figures_dir + 'd4000_list_arr.npy')
    d4000_err_list_arr = np.load(massive_figures_dir + 'd4000_err_list_arr.npy')

    print "Total galaxies in final array:", len(d4000_list_arr)

    # Now generate a random array based on this error array
    new_rand_err_list = []
    for j in range(len(err_arr)):
        new_rand_err_list.append(np.random.choice(err_arr))
    # convert to numpy array
    new_rand_err_arr = np.asarray(new_rand_err_list)

    # Now plot histograms for the two to compare them
    """
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.hist(err_arr, 50, range=(0, 1), color='k', histtype='step')
    ax2.hist(new_rand_err_arr, 50, range=(0, 1), color='r', histtype='step')

    plt.show()
    """

    # ------------------- plots ------------------ #
    # Check D4000 vs avg err
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{\left< f^{obs}_{err}\right>}$')
    ax.set_ylabel('D4000')

    # only plot the ones with high significance of measured D4000
    d4000_sig = d4000_list_arr / d4000_err_list_arr
    val_idx = np.where(d4000_sig >= 5)[0]
    print "Galaxies after applying D4000 significance cut:", len(val_idx)

    # min and max values for plot and for kernel density estimate
    xmin = 0.0
    xmax = 0.5
    ymin = 1.0
    ymax = 2.0

    # first clip x and y arrays to the specified min and max values
    x = err_arr[val_idx]
    y = d4000_list_arr[val_idx]
    x_idx = np.where((x>=xmin) & (x<=xmax))[0]
    y_idx = np.where((y>=ymin) & (y<=ymax))[0]
    xy_idx = reduce(np.intersect1d, (x_idx, y_idx))
    x = x[xy_idx]
    y = y[xy_idx]
    print "New total number of galaxies after rejecting galaxies outside believable ranges:", len(x)

    # plot points
    ax.scatter(x, y, s=3, color='k', zorder=10)

    # now use scipy gaussian kde
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values, bw_method=0.5)
    f = np.reshape(kernel(positions).T, xx.shape)

    print "KDE scotts factor:", kernel.scotts_factor()

    # contourf plot
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    # Contour plot
    cset = ax.contour(xx, yy, f, colors='k')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=10)

    # Test the pdf on a set of points
    print kernel.evaluate(np.array([[0.2, 0.5], [1.2, 1.6]]))

    # Also test that the integral over the full range is equal to 1
    print np.sum(kernel.pdf(positions))
    print kernel.integrate_box([xmin, ymin], [xmax, ymax])

    # probability of a point
    eps = 1e-4
    xcen = 1.6
    ycen = 0.1
    print kernel.integrate_box([xcen-eps, ycen-eps], [xcen+eps, ycen+eps])

    # -------------------- plot for random resampling from previous KDE -------------------- #
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.set_xlabel(r'$\mathrm{Resampled\ \left< f^{obs}_{err}\right>}$')
    ax1.set_ylabel('Resampled D4000')

    # plot points
    resample_arr = kernel.resample(size=50000)
    x_resamp = resample_arr[0]
    y_resamp = resample_arr[1]
    ax1.scatter(x_resamp, y_resamp, s=0.003, color='k', zorder=10)

    # now use scipy gaussian kde
    xx_resamp, yy_resamp = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions_resamp = np.vstack([xx_resamp.ravel(), yy_resamp.ravel()])
    values_resamp = np.vstack([x_resamp, y_resamp])
    kernel_resamp = gaussian_kde(values_resamp)
    f_resamp = np.reshape(kernel(positions_resamp).T, xx_resamp.shape)

    # contourf plot
    cfset = ax1.contourf(xx_resamp, yy_resamp, f_resamp, cmap='Blues')
    # Contour plot
    cset = ax1.contour(xx_resamp, yy_resamp, f_resamp, colors='k')
    # Label plot
    ax1.clabel(cset, inline=1, fontsize=10)
    
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    # ----------------- Choosing err based on D4000 ----------------- #
    # This part of the code will spit out a choice for the error value 
    # to insert in the mock spectra given it D4000 value. THe way it 
    # does this is by first finding the D4000 values in a close range 
    # to the D4000 of the mock spectrum. Within this small range of D4000
    # it then picks a point from the resampled (see above) distribution
    # of D4000 vs ferr and the error value of htat point is now hte 
    # average error given to the mock spectrum. It also asssumes 
    # that every point in the resapled distribution within the thin 
    # strip around the chosen D4000 is equally likely.

    err_resamp = resample_arr[0]
    d4000_resamp = resample_arr[1]

    mock_d4000 = 1.2

    eps_d4000 = 0.001

    # The two while loops here make sure that the code doesn't break
    # in stupid ways, like because of my arbitrary choice of eps_d4000
    while True:

        d4000_resamp_idx = np.where((d4000_resamp >= mock_d4000 - eps_d4000) & (d4000_resamp <= mock_d4000 + eps_d4000))[0]

        if len(d4000_resamp_idx) == 0:
            print "Did not find any re-samples within given D4000 range. Increasing search width."
            eps_d4000 = 5*eps_d4000
            continue

        if len(d4000_resamp_idx) > 0:
            print "Number of resampled D4000 values within the specified range:", len(d4000_resamp_idx)
            break

    err_in_choice_range = err_resamp[d4000_resamp_idx]

    while True:
        # i.e. keep doing it until it chooses a number which is not exactly zero
        # I know that it is highly unlikely to choose exactly zero ever but I just
        # wanted to make sure that this part never failed.
        chosen_err = np.random.choice(err_in_choice_range)

        if chosen_err != 0:
            break

    if chosen_err < 0:
        chosen_err = np.abs(chosen_err)

    print "Chosen error for mock spectrum", chosen_err

    sys.exit(0)
