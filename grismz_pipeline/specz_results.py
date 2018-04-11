from __future__ import division

import numpy as np
from functools import reduce

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

def line_func(x, slope, intercept):
    return slope*x + intercept

def make_zspec_comparison_plot(z_spec, z_grism, z_phot):

    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    from scipy.optimize import curve_fit

    mpl.rcParams['axes.grid'] = True

    # z_grism vs z_phot vs z_spec plot
    gs = gridspec.GridSpec(15,34)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=1.0, hspace=0.0)

    fig_gs = plt.figure(figsize=(12.8, 9.6))
    ax1 = fig_gs.add_subplot(gs[:10,:10])
    ax2 = fig_gs.add_subplot(gs[10:,:10])
    ax3 = fig_gs.add_subplot(gs[:10,12:22])
    ax4 = fig_gs.add_subplot(gs[10:,12:22])
    ax5 = fig_gs.add_subplot(gs[:10,24:])
    ax6 = fig_gs.add_subplot(gs[10:,24:])

    # ------------------------------------------------------------------------------------------------- #
    # first panel # z_spec vs z_grism
    ax1.plot(z_spec, z_grism, 'o', markersize=5.0, color='k', markeredgecolor='k', zorder=10)
    ax1.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r', linewidth=2.0)

    ax1.set_xlim(0.6, 1.24)
    ax1.set_ylim(0.6, 1.24)

    ax1.set_ylabel(r'$\mathrm{z_g}$', fontsize=18, labelpad=1)

    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels(['', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2'], fontsize='x-large', rotation=45)

    # do the fit with scipy
    popt, pcov = curve_fit(line_func, z_spec, z_grism, p0=[1.0, 0.6])
    #print popt
    #print pcov

    # Find stddev for the residuals
    resid = (z_spec - z_grism)/(1+z_spec)
    mu = np.mean(resid)
    sigma = np.std(resid)

    x_plot = np.arange(0.2,1.5,0.01)

    ax1.plot(x_plot, (mu+sigma) + (1+mu+sigma)*x_plot, '-', color='lightblue', linewidth=2.0)
    ax1.plot(x_plot, (mu-sigma) + (1+mu-sigma)*x_plot, '-', color='lightblue', linewidth=2.0)

    # residuals for first panel
    ax2.plot(z_spec, (z_spec - z_grism)/(1+z_spec), 'o', markersize=5.0, color='k', markeredgecolor='k', zorder=10)
    ax2.axhline(y=0, linestyle='--', color='r')

    ax2.axhline(y=sigma, ls='-', color='lightblue', linewidth=2.0)
    ax2.axhline(y=-1*sigma, ls='-', color='lightblue', linewidth=2.0)

    ax2.set_xlim(0.6, 1.24)
    ax2.set_ylim(-0.1, 0.1)

    ax2.set_xticklabels(ax2.get_xticks().tolist(), size='x-large', rotation=45)
    ax2.set_yticklabels(ax2.get_yticks().tolist(), size='x-large', rotation=45)

    ax2.set_xlabel(r'$\mathrm{z_s}$', fontsize=18)
    ax2.set_ylabel(r'$(\mathrm{z_s - z_g})/(1+\mathrm{z_s})$', fontsize=18, labelpad=-2)

    # ------------------------------------------------------------------------------------------------- #
    # second panel # z_spec vs z_phot
    ax3.plot(z_spec, z_phot, 'o', markersize=5.0, color='k', markeredgecolor='k', zorder=10)
    ax3.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r', linewidth=2.0)

    ax3.set_xlim(0.6, 1.24)
    ax3.set_ylim(0.6, 1.24)

    ax3.set_ylabel(r'$\mathrm{z_p}$', fontsize=18, labelpad=1)

    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels(['', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2'], fontsize='x-large', rotation=45)

    # Find stddev for the residuals
    resid = (z_spec - z_phot)/(1+z_spec)
    mu = np.mean(resid)
    sigma = np.std(resid)

    ax3.plot(x_plot, (mu+sigma) + (1+mu+sigma)*x_plot, '-', color='lightblue', linewidth=2.0)
    ax3.plot(x_plot, (mu-sigma) + (1+mu-sigma)*x_plot, '-', color='lightblue', linewidth=2.0)

    # residuals for second panel
    ax4.plot(z_spec, (z_spec - z_phot)/(1+z_spec), 'o', markersize=5.0, color='k', markeredgecolor='k', zorder=10)
    ax4.axhline(y=0, linestyle='--', color='r')

    ax4.axhline(y=sigma, ls='-', color='lightblue', linewidth=2.0)
    ax4.axhline(y=-1*sigma, ls='-', color='lightblue', linewidth=2.0)

    ax4.set_xlim(0.6, 1.24)
    ax4.set_ylim(-0.1, 0.1)

    ax4.set_xticklabels(ax4.get_xticks().tolist(), size='x-large', rotation=45)
    ax4.set_yticklabels(ax4.get_yticks().tolist(), size='x-large', rotation=45)

    ax4.set_xlabel(r'$\mathrm{z_s}$', fontsize=18)
    ax4.set_ylabel(r'$(\mathrm{z_s - z_p})/(1+\mathrm{z_s})$', fontsize=18, labelpad=-2)

    # ------------------------------------------------------------------------------------------------- #
    # third panel # z_spec vs z_phot
    ax5.plot(z_grism, z_phot, 'o', markersize=5.0, color='k', markeredgecolor='k', zorder=10)
    ax5.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r', linewidth=2.0)

    ax5.set_xlim(0.6, 1.24)
    ax5.set_ylim(0.6, 1.24)

    ax5.set_ylabel(r'$\mathrm{z_p}$', fontsize=18, labelpad=1)

    ax5.xaxis.set_ticklabels([])
    ax5.yaxis.set_ticklabels(['', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2'], fontsize='x-large', rotation=45)

    # Find stddev for the residuals
    resid = (z_grism - z_phot)/(1+z_grism)
    sigma = np.std(resid)

    ax5.plot(x_plot, x_plot + (1+x_plot)*sigma, '-', color='lightblue', linewidth=2.0)
    ax5.plot(x_plot, x_plot - (1+x_plot)*sigma, '-', color='lightblue', linewidth=2.0)

    # residuals for third panel
    ax6.plot(z_grism, (z_grism - z_phot)/(1+z_grism), 'o', markersize=5.0, color='k', markeredgecolor='k', zorder=10)
    ax6.axhline(y=0, linestyle='--', color='r')

    ax6.axhline(y=sigma, ls='-', color='lightblue', linewidth=2.0)
    ax6.axhline(y=-1*sigma, ls='-', color='lightblue', linewidth=2.0)

    ax6.set_xlim(0.6, 1.24)
    ax6.set_ylim(-0.1, 0.1)

    ax6.set_xticklabels(ax6.get_xticks().tolist(), size='x-large', rotation=45)
    ax6.set_yticklabels(ax6.get_yticks().tolist(), size='x-large', rotation=45)

    ax6.set_xlabel(r'$\mathrm{z_g}$', fontsize=18)
    ax6.set_ylabel(r'$(\mathrm{z_g - z_p})/(1+\mathrm{z_g})$', fontsize=18, labelpad=-2)

    fig_gs.savefig(massive_figures_dir + "zspec_comparison.eps", dpi=300, bbox_inches='tight')

    return None

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

    # Place some cuts
    chi2_thresh = 2.0
    d4000_thresh = 1.5
    valid_idx1 = np.where((zgrism_arr >= 0.6) & (zgrism_arr <= 1.235))[0]
    valid_idx2 = np.where(chi2_arr < chi2_thresh)[0]
    valid_idx3 = np.where(d4000_arr >= d4000_thresh)[0]
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

    N_gal = len(zgrism_plot)
    print N_gal, "galaxies in plot."
    #print "Only", len(zspec_plot), "galaxies within the", len(spec_res_cat), 
    #print "pass the D4000, NetSig, and overall error constraints."

    make_zspec_comparison_plot(zspec_plot, zgrism_plot, zphot_plot)
    sys.exit(0)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('Residuals', fontsize=15)
    ax.set_ylabel('N', fontsize=15)

    # define colors
    myblue = mh.rgb_to_hex(0, 100, 180)
    myred = mh.rgb_to_hex(214, 39, 40)  # tableau 20 red

    grism_resid_hist_arr = (zspec_plot - zgrism_plot)/(1+zspec_plot)
    photz_resid_hist_arr = (zspec_plot - zphot_plot)/(1+zspec_plot)

    print "Grism-z resid mean:", np.mean(grism_resid_hist_arr)
    print "Grism-z resid std:", np.std(grism_resid_hist_arr)
    print "photo-z resid mean:", np.mean(photz_resid_hist_arr)
    print "photo-z resid std:", np.std(photz_resid_hist_arr)

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
    # this fullrange parameter simply tells the program whether or not 
    # to plot the entire range of residual values.
    if fullrange:
        # If you don't want to restrict the range
        photz_min = np.min(photz_resid_hist_arr)
        photz_max = np.max(photz_resid_hist_arr)
        grismz_min = np.min(grism_resid_hist_arr)
        grismz_max = np.max(grism_resid_hist_arr)

        data_min = np.min([photz_min, grismz_min])
        data_max = np.max([photz_max, grismz_max])

        # changing data min and max to fixed endpoints for 
        # all d4000_thresh just to be able to compare them easily.
        data_min = -0.12
        data_max= 0.12

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

    ax.text(0.72, 0.78, r'$\mathrm{D4000\geq\,}$' + str(d4000_thresh), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='black', size=12)
    ax.text(0.72, 0.73, r'$\mathrm{N=\,}$' + str(N_gal), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='black', size=12)

    ax.minorticks_on()

    # force integer tick labels
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if fullrange:
        fig.savefig(massive_figures_dir + \
            'residual_histogram_netsig_10_fullrange_d4000_' + str(d4000_thresh).replace('.', 'p') + '.eps', \
            dpi=300, bbox_inches='tight')
    else:
        fig.savefig(massive_figures_dir + 'residual_histogram_netsig_10.png', dpi=300, bbox_inches='tight')

    #plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    sys.exit(0)