from __future__ import division

import numpy as np

import sys
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
figs_dir = home + "/Desktop/FIGS/"

def line_func(x, slope, intercept):
    return slope*x + intercept

if __name__ == '__main__':

    # Read in results arrays
    d4000_in = np.load(massive_figures_dir + 'model_mockspectra_fits/d4000_in_list_1p2to1p4.npy')
    d4000_out = np.load(massive_figures_dir + 'model_mockspectra_fits/d4000_out_list_1p2to1p4.npy')
    d4000_out_err = np.load(massive_figures_dir + 'model_mockspectra_fits/d4000_out_err_list_1p2to1p4.npy')
    mock_model_index = np.load(massive_figures_dir + 'model_mockspectra_fits/mock_model_index_list_1p2to1p4.npy')
    test_redshift = np.load(massive_figures_dir + 'model_mockspectra_fits/test_redshift_list_1p2to1p4.npy')
    mock_zgrism = np.load(massive_figures_dir + 'model_mockspectra_fits/mock_zgrism_list_1p2to1p4.npy')
    mock_zgrism_lowerr = np.load(massive_figures_dir + 'model_mockspectra_fits/mock_zgrism_lowerr_list_1p2to1p4.npy')
    mock_zgrism_higherr = np.load(massive_figures_dir + 'model_mockspectra_fits/mock_zgrism_higherr_list_1p2to1p4.npy')

    # --------- redshift accuracy comparison ---------- # 
    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[:8,:])
    ax2 = fig.add_subplot(gs[8:,:])

    # ------- z_mock vs test_redshift ------- #
    ax1.plot(test_redshift, mock_zgrism, 'o', markersize=2.0, color='k', markeredgecolor='k', zorder=10)
    ax1.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r', linewidth=2.0)

    ax1.set_xlim(0.6, 1.24)
    ax1.set_ylim(0.6, 1.24)

    ax1.set_ylabel(r'$\mathrm{z_{mock}}$', fontsize=12)

    ax1.xaxis.set_ticklabels([])

    # ------- residuals ------- #
    ax2.plot(test_redshift, (test_redshift - mock_zgrism)/(1+test_redshift), 'o', \
        markersize=2.0, color='k', markeredgecolor='k', zorder=10)
    ax2.axhline(y=0, linestyle='--', color='r')

    ax2.set_xlim(0.6, 1.24)

    ax2.set_xticklabels([])

    ax2.set_xlabel(r'$\mathrm{Test\ redshift}$', fontsize=12)
    ax2.set_ylabel(r'$\mathrm{Residuals}$', fontsize=12, labelpad=-1)

    # Turn on minorticks
    ax1.minorticks_on()
    ax2.minorticks_on()

    # Fitting
    # do the fit with scipy
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(line_func, test_redshift, mock_zgrism, p0=[1.0, 0.6])

    # plot line fit
    x_plot = np.arange(0.2,1.5,0.01)
    ax1.plot(x_plot, line_func(x_plot, popt[0], popt[1]), '-', color='#41ab5d', linewidth=2.0)

    # Find stddev for the residuals
    resid = (test_redshift - mock_zgrism)/(1+test_redshift)
    mu = np.mean(resid)
    sigma_nmad = 1.48 * np.median(abs(((test_redshift - mock_zgrism) - np.median((test_redshift - mock_zgrism))) / (1 + test_redshift)))

    print "Mean and sigma_nmad:", mu, sigma_nmad

    # plot fit to residuals
    ax2.axhline(y=mu + sigma_nmad, ls='-', color='#3690c0', linewidth=1.5)
    ax2.axhline(y=mu - sigma_nmad, ls='-', color='#3690c0', linewidth=1.5)
    ax2.axhline(y=mu, ls='-', color='blue', linewidth=1.5)

    # tick labels
    ax2.yaxis.set_ticklabels(['', '-0.1', '0', ''], fontsize='medium')

    # add text
    ax1.text(0.05, 0.9, r'$\mathrm{\mu=}$' + "{:.3}".format(mu), \
        verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='k', size=12)
    ax1.text(0.05, 0.85, r'$\mathrm{\sigma_{NMAD}=}$' + "{:.3}".format(sigma_nmad), \
        verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='k', size=12)

    fig.savefig(massive_figures_dir + \
        'model_mockspectra_fits/mock_redshift_comparison_d4000_1p2to1p4.eps', dpi=300, bbox_inches='tight')

    sys.exit(0)