from __future__ import division

import numpy as np
import scipy.stats as stats

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, AnchoredText

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"

if __name__ == '__main__':

    # read catalog adn rename arrays for convenience
    pears_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-N.txt',\
     dtype=None, names=True, skip_header=1)
    pears_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-S.txt',\
     dtype=None, names=True, skip_header=1)

    z_spec = np.concatenate((pears_cat_n['new_z'], pears_cat_s['new_z']), axis=0)
    z_phot = np.concatenate((pears_cat_n['old_z'], pears_cat_s['old_z']), axis=0)

    old_chi2 = np.concatenate((pears_cat_n['old_chi2'], pears_cat_s['old_chi2']), axis=0)
    old_chi2 = np.log10(old_chi2 / 88)

    new_chi2 = np.concatenate((pears_cat_n['new_chi2'], pears_cat_s['new_chi2']), axis=0)
    new_chi2 = np.log10(new_chi2 / 88)

    z_spec_std = np.concatenate((pears_cat_n['new_z_err'], pears_cat_s['new_z_err']), axis=0)

    # make plots
    # z_spec vs z_phot
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=2.0)

    fig = plt.figure()
    ax1 = fig.add_subplot(gs[:10,:])
    ax2 = fig.add_subplot(gs[10:,:])

    # first subplot
    ax1.plot(z_phot, z_spec, 'o', markersize=1.5, color='k', markeredgecolor='k')
    ax1.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r')

    ax1.set_xlim(0.2, 1.42)
    ax1.set_ylim(0.2, 1.42)

    ax1.set_ylabel(r'$z_\mathrm{phot}$', fontsize=15)

    ax1.xaxis.set_ticklabels([])

    ax1.minorticks_on()
    ax1.tick_params('both', width=1, length=3, which='minor')
    ax1.tick_params('both', width=1, length=4.7, which='major')
    ax1.grid(True)

    # second subplot
    ax2.plot(z_spec, (z_spec - z_phot)/(1+z_spec), 'o', markersize=1.5, color='k', markeredgecolor='k')
    ax2.axhline(y=0, linestyle='--', color='r')

    ax2.set_xlim(0.2, 1.42)
    ax2.set_ylim(-0.5, 0.5)

    ax2.set_xlabel(r'$z_\mathrm{spec}$', fontsize=15)
    ax2.set_ylabel(r'$(z_\mathrm{spec} - z_\mathrm{phot})/(1+z_\mathrm{spec})$', fontsize=15)

    ax2.minorticks_on()
    ax2.tick_params('both', width=1, length=3, which='minor')
    ax2.tick_params('both', width=1, length=4.7, which='major')
    ax2.grid(True)

    fig.savefig(massive_figures_dir + "refined_zspec_vs_zphot.eps", dpi=300, bbox_inches='tight')

    del fig, ax1, ax2

    # histograms of chi2
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get total bins and plot histogram
    #old_chi2_indx = np.where(old_chi2 < 20)
    #new_chi2_indx = np.where(new_chi2 < 20)
    #old_chi2 = old_chi2[old_chi2_indx]
    #new_chi2 = new_chi2[new_chi2_indx]

    new_chi2 = new_chi2[~np.isnan(new_chi2)]
    old_chi2 = old_chi2[~np.isnan(old_chi2)] 

    iqr = np.std(old_chi2, dtype=np.float64)
    binsize = 2*iqr*np.power(len(old_chi2),-1/3)
    totalbins = np.floor((max(old_chi2) - min(old_chi2))/binsize)

    ax.hist(old_chi2, totalbins, facecolor='None', align='mid', linewidth=1, edgecolor='b', histtype='step')

    iqr = np.std(new_chi2, dtype=np.float64)
    binsize = 2*iqr*np.power(len(new_chi2),-1/3)
    totalbins = np.floor((max(new_chi2) - min(new_chi2))/binsize)

    ax.hist(new_chi2, totalbins, facecolor='None', align='mid', linewidth=1, edgecolor='r', histtype='step')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    ax.set_xlabel(r'$\mathrm{log(\chi^2)}$')
    ax.set_ylabel(r'$\mathrm{N}$')

    fig.savefig(massive_figures_dir + "refined_old_new_chi2_hist.eps", dpi=300, bbox_inches='tight')

    # histogram of error in new redshift
    fig = plt.figure()
    ax = fig.add_subplot(111)

    count_zero = 0
    count_nonzero_finite = 0
    norm_z_err_plot = np.empty(len(z_spec_std))
    for i in range(len(z_spec_std)):
        delta_z = z_phot[i] - z_spec[i]
        if delta_z == 0:
            count_zero += 1
        elif (delta_z != 0) & ~np.isinf(delta_z / z_spec_std[i]):
            count_nonzero_finite += 1
        norm_z_err_plot[i] = delta_z / z_spec_std[i]

    print "Total zero values in normalized error of new redshift --", count_zero
    print "Total nonzero and finite values in normalized error of new redshift --", count_nonzero_finite

    rng = 100
    norm_z_err_indx = np.where(abs(norm_z_err_plot) <= rng)[0]
    norm_z_err_plot = norm_z_err_plot[norm_z_err_indx]
    print "Total values in range +-", rng ,"for normalized error of new redshift --", len(norm_z_err_indx)

    ax.hist(norm_z_err_plot[np.isfinite(norm_z_err_plot)], 30, facecolor='None', align='mid', linewidth=1, edgecolor='r', histtype='step')

    #z_phot_err = []
    #for i in range(len(z_phot)):
    #    if z_phot[i] < z_spec[i]:
    #        z_phot_err.append(z_spec[i] - norm_z_err[i] - z_phot[i])
    #    elif z_phot[i] > z_spec[i]:
    #        z_phot_err.append(z_phot[i] - z_spec[i] + norm_z_err[i])
    #    elif z_phot[i] == z_spec[i]:
    #        z_phot_err.append(0.0)
    #z_phot_err = np.asarray(z_phot_err)

    #z_phot_err_indx = np.where(z_phot_err < 0.2)[0]
    #z_phot_err = z_phot_err[z_phot_err_indx]
    #iqr = np.std(z_phot_err, dtype=np.float64)
    #binsize = 2*iqr*np.power(len(z_phot_err),-1/3)
    #totalbins = np.floor((max(z_phot_err) - min(z_phot_err))/binsize)

    #ax.hist(z_phot_err, totalbins, facecolor='None', align='mid', linewidth=1, edgecolor='b', histtype='step')    
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)
    
    ax.set_xlabel(r'$\mathrm{Normalized\ Error\ (\Delta z_{norm} = \Delta z / \sigma_{z_{new}})}$')
    ax.set_ylabel(r'$\mathrm{N}$')

    #ax.set_xlim(-0.1, 0.2)

    print '\n'
    print "Mean of measurement uncertainty in new redshift--", np.mean(z_spec_std)
    print "Median of measurement uncertainty in new redshift--", np.median(z_spec_std)
    print "Mode of measurement uncertainty in new redshift--", stats.mode(z_spec_std)
    print "Total values of measurement uncertainty in new redshift within 3% --", len(np.where(z_spec_std <= 0.03)[0])
    print "Total values of measurement uncertainty in new redshift within 1% --", len(np.where(z_spec_std <= 0.01)[0])
    print '\n'

    print "Mean of normalized error of new redshift --", np.mean(norm_z_err_plot[np.isfinite(norm_z_err_plot)])
    print "Median of normalized error of new redshift --", np.median(norm_z_err_plot[np.isfinite(norm_z_err_plot)])
    print "Mode of normalized error of new redshift --", stats.mode(norm_z_err_plot[np.isfinite(norm_z_err_plot)])
    print "Total values of normalized error of new redshift within 3% --", len(np.where(norm_z_err_plot[np.isfinite(norm_z_err_plot)] <= 0.03)[0])
    print "Total values of normalized error of new redshift within 1% --", len(np.where(norm_z_err_plot[np.isfinite(norm_z_err_plot)] <= 0.01)[0])

    fig.savefig(massive_figures_dir + "refined_norm_z_err_hist_inrange_pm_" + str(rng) + ".eps", dpi=300, bbox_inches='tight')
    del fig, ax

    fig = plt.figure()
    ax = fig.add_subplot(111)

    z_spec_std_indx = np.where(z_spec_std < 0.2)[0]
    z_spec_std = z_spec_std[z_spec_std_indx]

    ax.hist(z_spec_std[np.isfinite(z_spec_std)], 30, facecolor='None', align='mid', linewidth=1, edgecolor='r', histtype='step')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)
    
    ax.set_xlabel(r'$\mathrm{\sigma_{z_{new}}}$')
    ax.set_ylabel(r'$\mathrm{N}$')

    ax.set_xlim(0.0, 0.2)

    fig.savefig(massive_figures_dir + "refined_z_err_hist.eps", dpi=300, bbox_inches='tight')

    sys.exit(0)

