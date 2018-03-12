from __future__ import division

import numpy as np
from functools import reduce

import os
import sys

import matplotlib.pyplot as plt
import matplotlib
print matplotlib.matplotlib_fname()

from matplotlib import rcParams
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
"""
I've set the x and y axis tick direction to be 'in'
in the matplotlibrc file but it is not working for
some reason. So I have to explicitly set these here.
Need to figure this out.
"""

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
import grid_coadd as gd
import mag_hist as mh
import new_refine_grismz_gridsearch_parallel as ngp
import dn4000_catalog as dc

def get_all_d4000_speczqual(id_arr, field_arr, zspec_arr, specz_goodsn, specz_goodss):

    d4000_list = []
    d4000_err_list = []
    specz_qual_list = []
    specz_source_list = []

    # loop over all objects
    for i in range(len(id_arr)):

        # get obs data, deredshift and get d4000
        lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(id_arr[i], field_arr[i])
        #print "At ID", id_arr[i], "in", field_arr[i], "Return Code was:", return_code

        lam_em = lam_obs / (1 + zspec_arr[i])
        flam_em = flam_obs * (1 + zspec_arr[i])
        ferr_em = ferr_obs * (1 + zspec_arr[i])

        d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

        # now get specz quality
        if field_arr[i] == 'GOODS-N':
            idx = np.where(specz_goodsn['pearsid'] == id_arr[i])[0]
            specz_qual = specz_goodsn['specz_qual'][idx]
            specz_source = specz_goodsn['specz_source'][idx]
        elif field_arr[i] == 'GOODS-S':
            idx = np.where(specz_goodss['pearsid'] == id_arr[i])[0]
            specz_qual = specz_goodss['specz_qual'][idx]
            specz_source = specz_goodss['specz_source'][idx]

        # append
        d4000_list.append(d4000)
        d4000_err_list.append(d4000_err)
        specz_qual_list.append(specz_qual)
        specz_source_list.append(specz_source)

    # convert to numpy arrays
    d4000_list = np.asarray(d4000_list)
    d4000_err_list = np.asarray(d4000_err_list)
    specz_qual_list = np.asarray(specz_qual_list)
    specz_source_list = np.asarray(specz_source_list)

    specz_qual_list = specz_qual_list.ravel()
    specz_source_list = specz_source_list.ravel()
    # added these lines to ravel after checking htat the returned shape was (N, 1) instead of just (N)

    return d4000_list, d4000_err_list, specz_qual_list, specz_source_list

if __name__ == '__main__':
    
    # Read in arrays
    id_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/id_list.npy')
    field_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/field_list.npy')
    zgrism_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/zgrism_list.npy')
    zspec_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/zspec_list.npy')
    zphot_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/zphot_list.npy')
    chi2_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/chi2_list.npy')
    netsig_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/netsig_list.npy')
    print "Code ran for", len(zgrism_arr), "galaxies."

    # get d4000 and also specz quality for all galaxies
    # Read in Specz comparison catalogs
    specz_goodsn = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-N.txt', dtype=None, names=True)
    specz_goodss = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-S.txt', dtype=None, names=True)

    d4000_arr, d4000_err_arr, specz_qual_arr, specz_source_arr = \
    get_all_d4000_speczqual(id_arr, field_arr, zspec_arr, specz_goodsn, specz_goodss)

    # Place some more cuts
    valid_idx1 = np.where((zgrism_arr >= 0.6) & (zgrism_arr <= 1.235))[0]
    valid_idx2 = np.where(chi2_arr < 2.0)[0]
    valid_idx3 = np.where(d4000_arr >= 1.5)[0]
    valid_idx4 = np.where((specz_qual_arr != '4') & (specz_qual_arr != 'D'))[0]
    valid_idx = reduce(np.intersect1d, (valid_idx1, valid_idx2, valid_idx3, valid_idx4))
    # I'm not making a cut on netsig 

    id_plot = id_arr[valid_idx]
    field_plot = field_arr[valid_idx]
    zgrism_plot = zgrism_arr[valid_idx]
    zspec_plot = zspec_arr[valid_idx]
    zphot_plot = zphot_arr[valid_idx]
    chi2_plot = chi2_arr[valid_idx]
    netsig_plot = netsig_arr[valid_idx]

    d4000_arr = d4000_arr[valid_idx]
    d4000_err_arr = d4000_err_arr[valid_idx]
    specz_qual_arr = specz_qual_arr[valid_idx]
    specz_source_arr = specz_source_arr[valid_idx]

    print len(zgrism_plot), "galaxies in plot."
    #print "Only", len(zspec_plot), "galaxies within the", len(spec_res_cat), 
    #print "pass the D4000, NetSig, and overall error constraints."

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # define colors
    myblue = mh.rgb_to_hex(0, 100, 180)
    myred = mh.rgb_to_hex(214, 39, 40)  # tableau 20 red

    grism_resid_hist_arr = (zspec_plot - zgrism_plot)/(1+zspec_plot)
    photz_resid_hist_arr = (zspec_plot - zphot_plot)/(1+zspec_plot)

    large_diff_idx = np.where(abs(grism_resid_hist_arr) > 0.04)[0]
    np.set_printoptions(precision=2, suppress=True)
    print len(large_diff_idx)
    print id_plot[large_diff_idx]
    print field_plot[large_diff_idx]
    #print zgrism_plot[large_diff_idx]
    #print zspec_plot[large_diff_idx]
    #print zphot_plot[large_diff_idx]
    #print netsig_plot[large_diff_idx]
    print chi2_plot[large_diff_idx]
    print specz_qual_arr[large_diff_idx]
    print d4000_arr[large_diff_idx]
    print specz_source_arr[large_diff_idx]
    #print len(np.where(specz_qual_arr == 'Z')[0])

    fullrange = True
    if fullrange:
        # If you don't want to restrict the range
        photz_min = np.min(photz_resid_hist_arr)
        photz_max = np.max(photz_resid_hist_arr)
        grismz_min = np.min(grism_resid_hist_arr)
        grismz_max = np.max(grism_resid_hist_arr)

        data_min = np.min([photz_min, grismz_min])
        data_max = np.max([photz_max, grismz_max])

        binwidth = 0.005

        ax.hist(photz_resid_hist_arr, bins=np.arange(data_min, data_max+binwidth, binwidth), \
            histtype='step', color=myred, zorder=10)
        ax.hist(grism_resid_hist_arr, bins=np.arange(data_min, data_max+binwidth, binwidth), \
            histtype='step', color=myblue, zorder=10)
    else:
        ax.hist(photz_resid_hist_arr, 20, range=[-0.05,0.05], histtype='step', color=myred, zorder=10)
        ax.hist(grism_resid_hist_arr, 20, range=[-0.05,0.05], histtype='step', color=myblue, zorder=10)

    # this plot needs an alpha channel if you fill in the face color
    # otherwise you wont see that the photo-z histogram under the grism-z histogram
    # is actually fatter around 0 whereas the grism-z histogram is thinner.
    # It does not need an alpha channel if you do histtype='step'

    ax.axvline(x=0.0, ls='--', color='k')

    ax.text(0.72, 0.97, r'$\mathrm{Grism{-}z}$' + '\n' + r'$\mathrm{residuals}$',\
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color=myblue, size=10)
    ax.text(0.835, 0.96, r'$\mathrm{\equiv \frac{z_s - z_g}{1 + z_s}}$',\
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color=myblue, size=14)

    ax.text(0.72, 0.87, r'$\mathrm{Photo{-}z}$' + '\n' + r'$\mathrm{residuals}$',\
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color=myred, size=10)
    ax.text(0.835, 0.86, r'$\mathrm{\equiv \frac{z_s - z_p}{1 + z_s}}$',\
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color=myred, size=14)

    ax.minorticks_on()

    if fullrange:
        fig.savefig(massive_figures_dir + \
            'new_specz_sample_fits/residual_histogram_netsig_10_fullrange.png', \
            dpi=300, bbox_inches='tight')
    else:
        fig.savefig(massive_figures_dir + \
            'new_specz_sample_fits/residual_histogram_netsig_10.png', \
            dpi=300, bbox_inches='tight')

    plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    sys.exit(0)