from __future__ import division

import numpy as np
from scipy.optimize import curve_fit

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

def plot_panel(ax_main, ax_resid, test_redshift_arr, mock_zgrism_arr, \
    mock_zgrism_lowerr_arr, mock_zgrism_uperr_arr, d4000_range_lowlim):

    # ------- z_mock vs test_redshift ------- #
    ax_main.plot(test_redshift_arr, mock_zgrism_arr, 'o', markersize=2.0, color='k', markeredgecolor='k', zorder=10)
    ax_main.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r', linewidth=2.0)

    # ------- residuals ------- #
    ax_resid.plot(test_redshift_arr, (test_redshift_arr - mock_zgrism_arr)/(1+test_redshift_arr), 'o', \
        markersize=2.0, color='k', markeredgecolor='k', zorder=10)
    ax_resid.axhline(y=0, linestyle='--', color='r')

    # Fitting
    # do the fit with scipy
    popt, pcov = curve_fit(line_func, test_redshift_arr, mock_zgrism_arr, p0=[1.0, 0.6])

    # plot line fit
    x_plot = np.arange(0.2,1.5,0.01)
    ax_main.plot(x_plot, line_func(x_plot, popt[0], popt[1]), '-', color='#41ab5d', linewidth=2.0)

    # Find residual stats
    resid = (test_redshift_arr - mock_zgrism_arr)/(1+test_redshift_arr)
    mu = np.mean(resid)
    sigma_nmad = \
    1.48 * np.median(abs(((test_redshift_arr - mock_zgrism_arr) - np.median(test_redshift_arr - mock_zgrism_arr)) / (1 + test_redshift_arr)))
    stddev = np.std(resid)

    outliers = np.where(resid > 0.05)[0]

    print "\n", d4000_range_lowlim.replace('p', '.'), "<= D4000 <= 1.4"
    print "Residual mean, sigma_nmad, and stddev:", mu, sigma_nmad, stddev
    print "Number of outliers i.e. Residual>5%:", len(outliers)
    out_frac = len(outliers)/len(mock_zgrism_arr)
    print "Outlier fraction:", out_frac

    # plot fit to residuals
    ax_resid.axhline(y=mu + stddev, ls='-', color='#3690c0', linewidth=1.5)
    ax_resid.axhline(y=mu - stddev, ls='-', color='#3690c0', linewidth=1.5)
    ax_resid.axhline(y=mu, ls='-', color='blue', linewidth=1.5)

    # Turn on minorticks
    ax_main.minorticks_on()
    ax_resid.minorticks_on()

    # Limits to axis
    ax_main.set_xlim(0.6, 1.24)
    ax_main.set_ylim(0.6, 1.24)

    ax_resid.set_xlim(0.6, 1.24)
    ax_resid.set_ylim(-0.1, 0.1)

    # Axis labels
    ax_main.set_ylabel(r'$\mathrm{z_{mock}}$', fontsize=12)
    ax_resid.set_xlabel(r'$\mathrm{Test\ redshift}$', fontsize=12)
    ax_resid.set_ylabel(r'$\mathrm{Residuals}$', fontsize=12, labelpad=-1)

    # tick labels
    ax_main.xaxis.set_ticklabels([])
    ax_resid.yaxis.set_ticklabels(['-0.1', '0', ''], fontsize='medium')

    if float(d4000_range_lowlim.replace('p', '.')) < 1.3:
        ax_resid.xaxis.set_ticklabels([])
        ax_resid.set_xlabel('')

    # add text
    ax_main.text(0.05, 0.95, d4000_range_lowlim.replace('p', '.') + r"$\ \leq \mathrm{D}4000 \leq 1.4$", \
        verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, color='k', size=12)
    ax_main.text(0.05, 0.89, r'$\mathrm{\mu =\,}$' + convert_to_sci_not(mu), \
        verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, color='k', size=12)
    ax_main.text(0.05, 0.84, r'$\mathrm{\sigma =\,}$' + convert_to_sci_not(stddev), \
        verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, color='k', size=12)

    ax_main.text(0.05, 0.75, r'$\mathrm{N =\,}$' + str(len(mock_zgrism_arr)),\
        verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, color='k', size=12)
    ax_main.text(0.05, 0.7, r'$\mathrm{Outlier\ fraction =\,}$' + "{:.3f}".format(out_frac),\
        verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, color='k', size=12)

    # Plot avg mock_zgrism error bar on each panel
    avg_lowerr = np.nanmean(mock_zgrism_lowerr_arr)
    avg_uperr = np.nanmean(mock_zgrism_uperr_arr)
    yerrbar = [[avg_lowerr], [avg_uperr]]

    if d4000_range_lowlim == '1p2':
        ax_main.errorbar(np.array([1.2]), np.array([0.82]), yerr=yerrbar, fmt='o', \
            color='k', markeredgecolor='k', capsize=0, markersize=5.0, elinewidth=0.6)
    else:
        ax_main.errorbar(np.array([1.2]), np.array([0.7]), yerr=yerrbar, fmt='o', \
            color='k', markeredgecolor='k', capsize=0, markersize=5.0, elinewidth=0.6)

    return ax_main, ax_resid

def convert_to_sci_not(n):
    """
    I'm not sure how well this function works
    in every possible case. Needs more testing.
    """

    # convert to python string with sci notation
    n_str = "{:.3e}".format(n)

    # split string and assign parts
    n_splt = n_str.split('e')
    decimal = n_splt[0]
    exponent = n_splt[1]

    # strip leading zeros in exponent
    if float(exponent) < 0:
        exponent = exponent.split('-')[1]
        exponent = '-' + exponent.lstrip('0')
    elif float(exponent) > 0:
        exponent = exponent.lstrip('0')

    # create final string with proper TeX sci notation and return
    sci_str = decimal + r'$\times$' + r'$\mathrm{10^{' + exponent + r'}}$'

    return sci_str

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
    gs = gridspec.GridSpec(20,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=2.1, hspace=0.0)

    fig_gs = plt.figure(figsize=(10, 10))
    ax1 = fig_gs.add_subplot(gs[:8,:5])
    ax2 = fig_gs.add_subplot(gs[8:10,:5])
    
    ax3 = fig_gs.add_subplot(gs[:8,5:])
    ax4 = fig_gs.add_subplot(gs[8:10,5:])
    
    ax5 = fig_gs.add_subplot(gs[10:18,:5])
    ax6 = fig_gs.add_subplot(gs[18:,:5])

    ax7 = fig_gs.add_subplot(gs[10:18,5:])
    ax8 = fig_gs.add_subplot(gs[18:,5:])

    # Create arrays for all four panels
    d4000_gtr_1p2_idx = np.where(d4000_out >= 1.2)[0]
    d4000_gtr_1p25_idx = np.where(d4000_out >= 1.25)[0]
    d4000_gtr_1p3_idx = np.where(d4000_out >= 1.3)[0]
    d4000_gtr_1p35_idx = np.where(d4000_out >= 1.35)[0]

    test_redshift_d4000_gtr_1p2 = test_redshift[d4000_gtr_1p2_idx]
    test_redshift_d4000_gtr_1p25 = test_redshift[d4000_gtr_1p25_idx]
    test_redshift_d4000_gtr_1p3 = test_redshift[d4000_gtr_1p3_idx]
    test_redshift_d4000_gtr_1p35 = test_redshift[d4000_gtr_1p35_idx]

    mock_zgrism_d4000_gtr_1p2 = mock_zgrism[d4000_gtr_1p2_idx]
    mock_zgrism_d4000_gtr_1p25 = mock_zgrism[d4000_gtr_1p25_idx]
    mock_zgrism_d4000_gtr_1p3 = mock_zgrism[d4000_gtr_1p3_idx]
    mock_zgrism_d4000_gtr_1p35 = mock_zgrism[d4000_gtr_1p35_idx]

    mock_zgrism_lowerr_d4000_gtr_1p2 = mock_zgrism_lowerr[d4000_gtr_1p2_idx]
    mock_zgrism_lowerr_d4000_gtr_1p25 = mock_zgrism_lowerr[d4000_gtr_1p25_idx]
    mock_zgrism_lowerr_d4000_gtr_1p3 = mock_zgrism_lowerr[d4000_gtr_1p3_idx]
    mock_zgrism_lowerr_d4000_gtr_1p35 = mock_zgrism_lowerr[d4000_gtr_1p35_idx]

    mock_zgrism_higherr_d4000_gtr_1p2 = mock_zgrism_higherr[d4000_gtr_1p2_idx]
    mock_zgrism_higherr_d4000_gtr_1p25 = mock_zgrism_higherr[d4000_gtr_1p25_idx]
    mock_zgrism_higherr_d4000_gtr_1p3 = mock_zgrism_higherr[d4000_gtr_1p3_idx]
    mock_zgrism_higherr_d4000_gtr_1p35 = mock_zgrism_higherr[d4000_gtr_1p35_idx]

    # make panel plots
    ax1, ax2 = plot_panel(ax1, ax2, test_redshift_d4000_gtr_1p2, mock_zgrism_d4000_gtr_1p2, \
        mock_zgrism_lowerr_d4000_gtr_1p2, mock_zgrism_higherr_d4000_gtr_1p2, '1p2')
    ax3, ax4 = plot_panel(ax3, ax4, test_redshift_d4000_gtr_1p25, mock_zgrism_d4000_gtr_1p25, \
        mock_zgrism_lowerr_d4000_gtr_1p25, mock_zgrism_higherr_d4000_gtr_1p25, '1p25')
    ax5, ax6 = plot_panel(ax5, ax6, test_redshift_d4000_gtr_1p3, mock_zgrism_d4000_gtr_1p3, \
        mock_zgrism_lowerr_d4000_gtr_1p3, mock_zgrism_higherr_d4000_gtr_1p3, '1p3')
    ax7, ax8 = plot_panel(ax7, ax8, test_redshift_d4000_gtr_1p35, mock_zgrism_d4000_gtr_1p35, \
        mock_zgrism_lowerr_d4000_gtr_1p35, mock_zgrism_higherr_d4000_gtr_1p35, '1p35')

    # add text only to the first panel
    ax1.axhline(y=0.75, xmin=0.55, xmax=0.67, ls='-', lw=2.0, color='#41ab5d')
    ax1.text(0.68, 0.26, 'Best fit line', verticalalignment='top', horizontalalignment='left', \
        transform=ax1.transAxes, color='k', size=14)

    ax1.axhline(y=0.7, xmin=0.55, xmax=0.67, ls='-', lw=2.0, color='blue')
    ax1.text(0.68, 0.18, 'Residual Mean', verticalalignment='top', horizontalalignment='left', \
        transform=ax1.transAxes, color='k', size=14)

    ax1.axhline(y=0.65, xmin=0.55, xmax=0.67, ls='-', lw=2.0, color='#3690c0')
    ax1.text(0.68, 0.1, r'$\mathrm{\pm 1\ \sigma}$', verticalalignment='top', horizontalalignment='left', \
        transform=ax1.transAxes, color='k', size=14)

    # Save figure 
    fig_gs.savefig(massive_figures_dir + \
        'model_mockspectra_fits/mock_redshift_comparison_d4000_1p2to1p4.eps', dpi=300, bbox_inches='tight')

    sys.exit(0)