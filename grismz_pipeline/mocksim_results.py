from __future__ import division

import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.stats import gaussian_kde

import sys
import os
import glob

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

    outliers = np.where(resid > 0.03)[0]

    # Print info
    if d4000_range_lowlim == '1p5':
        d4000_range_uplim = float(d4000_range_lowlim.replace('p', '.')) + 0.1
    else:
        d4000_range_uplim = float(d4000_range_lowlim.replace('p', '.')) + 0.05

    avg_lowerr = np.nanmean(mock_zgrism_lowerr_arr)
    avg_uperr = np.nanmean(mock_zgrism_uperr_arr)
    yerrbar = [[avg_lowerr], [avg_uperr]]

    print "\n", d4000_range_lowlim.replace('p', '.'), "<= D4000 < ", d4000_range_uplim
    print "Number of objects in bin:", len(resid)
    print "Residual mean, sigma_nmad, and stddev:", '{:.3}'.format(mu), '{:.3}'.format(sigma_nmad), '{:.3}'.format(stddev)
    print "Number of outliers i.e. Residual>3%:", len(outliers)
    out_frac = len(outliers)/len(mock_zgrism_arr)
    print "Outlier fraction:", out_frac
    print "Skewness of redisduals:", '{:.3}'.format(stats.skew(resid))
    print "Magnitude of avg lower and upper error on mock grism redshift:", '{:.3}'.format(avg_lowerr), '{:.3}'.format(avg_uperr),
    print "Avg of the prev two:", '{:.3}'.format(np.mean(yerrbar))

    # Printing for latex table
    print len(resid), "&",
    print "{:.3}".format(np.mean(yerrbar)), "&",
    print "{:.3}".format(mu), "&",
    print "{:.3}".format(stddev)

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
    ax_resid.set_ylim(-0.054, 0.054)

    # Axis labels
    if d4000_range_lowlim != '1p2' and d4000_range_lowlim != '1p3' \
    and d4000_range_lowlim != '1p4' and d4000_range_lowlim != '1p5':
        ax_main.set_ylabel('')
        ax_resid.set_ylabel('')
    else:
        ax_main.set_ylabel(r'$\mathrm{z_{mock}}$', fontsize=12)
        ax_resid.set_ylabel(r'$\mathrm{Residuals}$', fontsize=12, labelpad=-1)

    ax_resid.set_xlabel(r'$\mathrm{Test\ redshift}$', fontsize=12)

    # tick labels
    ax_main.xaxis.set_ticklabels([])
    ax_resid.yaxis.set_ticklabels(['', '-0.05', '0', ''], fontsize='medium')

    if float(d4000_range_lowlim.replace('p', '.')) < 1.5:
        ax_resid.xaxis.set_ticklabels([])
        ax_resid.set_xlabel('')

    # add text
    d4000_label_str = d4000_range_lowlim.replace('p', '.') + r"$\,\leq \mathrm{D}4000 <\,$" + str(d4000_range_uplim)
    ax_main.text(0.05, 0.95, d4000_label_str, \
        verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, color='k', size=10)
    ax_main.text(0.05, 0.86, r'$\mathrm{\mu =\,}$' + convert_to_sci_not(mu), \
        verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, color='k', size=10)
    ax_main.text(0.05, 0.76, r'$\mathrm{\sigma =\,}$' + convert_to_sci_not(stddev), \
        verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, color='k', size=10)

    ax_main.text(0.05, 0.6, r'$\mathrm{N =\,}$' + str(len(mock_zgrism_arr)),\
        verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, color='k', size=10)
    #ax_main.text(0.05, 0.5, r'$\mathrm{Outlier\ }$' + '\n' + r'$\mathrm{fraction} =\,$' + "{:.3f}".format(out_frac),\
    #    verticalalignment='top', horizontalalignment='left', transform=ax_main.transAxes, color='k', size=10)

    # Plot avg mock_zgrism error bar on each panel
    if d4000_range_lowlim == '1p2' or d4000_range_lowlim == '1p0':
        ax_main.errorbar(np.array([1.2]), np.array([0.82]), yerr=yerrbar, fmt='o', \
            color='r', markeredgecolor='r', capsize=0, markersize=1.5, elinewidth=0.6)
    else:
        ax_main.errorbar(np.array([1.2]), np.array([0.7]), yerr=yerrbar, fmt='o', \
            color='r', markeredgecolor='r', capsize=0, markersize=1.5, elinewidth=0.6)

    return ax_main, ax_resid

def convert_to_sci_not(n):
    """
    I'm not sure how well this function works
    in every possible case. Needs more testing.
    """

    # convert to python string with sci notation
    n_str = "{:.2e}".format(n)

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

def dummy_func_code_for9panelplot():

    # first row
    # D4000 > 1.05
    ax1 = fig_gs.add_subplot(gs[:5,:7])
    ax2 = fig_gs.add_subplot(gs[5:7,:7])
    
    # D4000 > 1.1
    ax3 = fig_gs.add_subplot(gs[:5,7:14])
    ax4 = fig_gs.add_subplot(gs[5:7,7:14])
    
    # D4000 > 1.15
    ax5 = fig_gs.add_subplot(gs[:5,14:])
    ax6 = fig_gs.add_subplot(gs[5:7,14:])

    # second row
    # D4000 > 1.2
    ax7 = fig_gs.add_subplot(gs[7:12,:7])
    ax8 = fig_gs.add_subplot(gs[12:14,:7])
    
    # D4000 > 1.25
    ax9 = fig_gs.add_subplot(gs[7:12,7:14])
    ax10 = fig_gs.add_subplot(gs[12:14,7:14])
    
    # D4000 > 1.3
    ax11 = fig_gs.add_subplot(gs[7:12,14:])
    ax12 = fig_gs.add_subplot(gs[12:14,14:])

    # third row
    # D4000 > 1.35
    ax13 = fig_gs.add_subplot(gs[14:19,:7])
    ax14 = fig_gs.add_subplot(gs[19:,:7])
    
    # D4000 > 1.4
    ax15 = fig_gs.add_subplot(gs[14:19,7:14])
    ax16 = fig_gs.add_subplot(gs[19:,7:14])
    
    # D4000 > 1.5
    ax17 = fig_gs.add_subplot(gs[14:19,14:])
    ax18 = fig_gs.add_subplot(gs[19:,14:])

    # Create arrays for all four panels
    valid_chi2_idx = np.where(chi2 < 2.0)[0]
    d4000_sig = d4000_out / d4000_out_err
    valid_d4000_sig_idx = np.where(d4000_sig >= 5)[0]

    d4000_gtr_1p05_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where(d4000_out >= 1.05)[0]))
    d4000_gtr_1p1_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where(d4000_out >= 1.1)[0]))
    d4000_gtr_1p15_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where(d4000_out >= 1.15)[0]))

    d4000_gtr_1p2_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where(d4000_out >= 1.2)[0]))
    d4000_gtr_1p25_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where(d4000_out >= 1.25)[0]))
    d4000_gtr_1p3_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where(d4000_out >= 1.3)[0]))

    d4000_gtr_1p35_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where(d4000_out >= 1.35)[0]))
    d4000_gtr_1p4_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where(d4000_out >= 1.4)[0]))
    d4000_gtr_1p5_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where(d4000_out >= 1.5)[0]))

    # ------
    test_redshift_d4000_gtr_1p05 = test_redshift[d4000_gtr_1p05_idx]
    test_redshift_d4000_gtr_1p1 = test_redshift[d4000_gtr_1p1_idx]
    test_redshift_d4000_gtr_1p15 = test_redshift[d4000_gtr_1p15_idx]

    test_redshift_d4000_gtr_1p2 = test_redshift[d4000_gtr_1p2_idx]
    test_redshift_d4000_gtr_1p25 = test_redshift[d4000_gtr_1p25_idx]
    test_redshift_d4000_gtr_1p3 = test_redshift[d4000_gtr_1p3_idx]

    test_redshift_d4000_gtr_1p35 = test_redshift[d4000_gtr_1p35_idx]
    test_redshift_d4000_gtr_1p4 = test_redshift[d4000_gtr_1p4_idx]
    test_redshift_d4000_gtr_1p5 = test_redshift[d4000_gtr_1p5_idx]

    # ------------------------------
    mock_zgrism_d4000_gtr_1p05 = mock_zgrism[d4000_gtr_1p05_idx]
    mock_zgrism_d4000_gtr_1p1 = mock_zgrism[d4000_gtr_1p1_idx]
    mock_zgrism_d4000_gtr_1p15 = mock_zgrism[d4000_gtr_1p15_idx]

    mock_zgrism_d4000_gtr_1p2 = mock_zgrism[d4000_gtr_1p2_idx]
    mock_zgrism_d4000_gtr_1p25 = mock_zgrism[d4000_gtr_1p25_idx]
    mock_zgrism_d4000_gtr_1p3 = mock_zgrism[d4000_gtr_1p3_idx]

    mock_zgrism_d4000_gtr_1p35 = mock_zgrism[d4000_gtr_1p35_idx]
    mock_zgrism_d4000_gtr_1p4 = mock_zgrism[d4000_gtr_1p4_idx]
    mock_zgrism_d4000_gtr_1p5 = mock_zgrism[d4000_gtr_1p5_idx]

    # ------------------------------
    mock_zgrism_lowerr_d4000_gtr_1p05 = mock_zgrism_lowerr[d4000_gtr_1p05_idx]
    mock_zgrism_lowerr_d4000_gtr_1p1 = mock_zgrism_lowerr[d4000_gtr_1p1_idx]
    mock_zgrism_lowerr_d4000_gtr_1p15 = mock_zgrism_lowerr[d4000_gtr_1p15_idx]

    mock_zgrism_lowerr_d4000_gtr_1p2 = mock_zgrism_lowerr[d4000_gtr_1p2_idx]
    mock_zgrism_lowerr_d4000_gtr_1p25 = mock_zgrism_lowerr[d4000_gtr_1p25_idx]
    mock_zgrism_lowerr_d4000_gtr_1p3 = mock_zgrism_lowerr[d4000_gtr_1p3_idx]

    mock_zgrism_lowerr_d4000_gtr_1p35 = mock_zgrism_lowerr[d4000_gtr_1p35_idx]
    mock_zgrism_lowerr_d4000_gtr_1p4 = mock_zgrism_lowerr[d4000_gtr_1p4_idx]
    mock_zgrism_lowerr_d4000_gtr_1p5 = mock_zgrism_lowerr[d4000_gtr_1p5_idx]

    # ------------------------------
    mock_zgrism_higherr_d4000_gtr_1p05 = mock_zgrism_higherr[d4000_gtr_1p05_idx]
    mock_zgrism_higherr_d4000_gtr_1p1 = mock_zgrism_higherr[d4000_gtr_1p1_idx]
    mock_zgrism_higherr_d4000_gtr_1p15 = mock_zgrism_higherr[d4000_gtr_1p15_idx]

    mock_zgrism_higherr_d4000_gtr_1p2 = mock_zgrism_higherr[d4000_gtr_1p2_idx]
    mock_zgrism_higherr_d4000_gtr_1p25 = mock_zgrism_higherr[d4000_gtr_1p25_idx]
    mock_zgrism_higherr_d4000_gtr_1p3 = mock_zgrism_higherr[d4000_gtr_1p3_idx]

    mock_zgrism_higherr_d4000_gtr_1p35 = mock_zgrism_higherr[d4000_gtr_1p35_idx]
    mock_zgrism_higherr_d4000_gtr_1p4 = mock_zgrism_higherr[d4000_gtr_1p4_idx]
    mock_zgrism_higherr_d4000_gtr_1p5 = mock_zgrism_higherr[d4000_gtr_1p5_idx]

    # make panel plots
    ax1, ax2 = plot_panel(ax1, ax2, test_redshift_d4000_gtr_1p05, mock_zgrism_d4000_gtr_1p05, mock_zgrism_lowerr_d4000_gtr_1p05, mock_zgrism_higherr_d4000_gtr_1p05, '1p05')
    ax3, ax4 = plot_panel(ax3, ax4, test_redshift_d4000_gtr_1p1, mock_zgrism_d4000_gtr_1p1, mock_zgrism_lowerr_d4000_gtr_1p1, mock_zgrism_higherr_d4000_gtr_1p1, '1p1')
    ax5, ax6 = plot_panel(ax5, ax6, test_redshift_d4000_gtr_1p15, mock_zgrism_d4000_gtr_1p15, mock_zgrism_lowerr_d4000_gtr_1p15, mock_zgrism_higherr_d4000_gtr_1p15, '1p15')

    ax7, ax8 = plot_panel(ax7, ax8, test_redshift_d4000_gtr_1p2, mock_zgrism_d4000_gtr_1p2, mock_zgrism_lowerr_d4000_gtr_1p2, mock_zgrism_higherr_d4000_gtr_1p2, '1p2')
    ax9, ax10 = plot_panel(ax9, ax10, test_redshift_d4000_gtr_1p25, mock_zgrism_d4000_gtr_1p25, mock_zgrism_lowerr_d4000_gtr_1p25, mock_zgrism_higherr_d4000_gtr_1p25, '1p25')
    ax11, ax12 = plot_panel(ax11, ax12, test_redshift_d4000_gtr_1p3, mock_zgrism_d4000_gtr_1p3, mock_zgrism_lowerr_d4000_gtr_1p3, mock_zgrism_higherr_d4000_gtr_1p3, '1p3')

    ax13, ax14 = plot_panel(ax13, ax14, test_redshift_d4000_gtr_1p35, mock_zgrism_d4000_gtr_1p35, mock_zgrism_lowerr_d4000_gtr_1p35, mock_zgrism_higherr_d4000_gtr_1p35, '1p35')
    ax15, ax16 = plot_panel(ax15, ax16, test_redshift_d4000_gtr_1p4, mock_zgrism_d4000_gtr_1p4, mock_zgrism_lowerr_d4000_gtr_1p4, mock_zgrism_higherr_d4000_gtr_1p4, '1p4')
    ax17, ax18 = plot_panel(ax17, ax18, test_redshift_d4000_gtr_1p5, mock_zgrism_d4000_gtr_1p5, mock_zgrism_lowerr_d4000_gtr_1p5, mock_zgrism_higherr_d4000_gtr_1p5, '1p5')

    return None

def model_d4000_vs_fluxerr(new_d4000, new_d4000_err, chosen_err):

    # Check D4000 vs avg err
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{\left< f^{obs}_{err}\right>}$')
    ax.set_ylabel('D4000')

    # only plot the ones with high significance of measured D4000
    d4000_sig = new_d4000 / new_d4000_err
    val_idx = np.where(d4000_sig >= 3)[0]
    print "Galaxies after applying D4000 significance cut:", len(val_idx)

    # min and max values for plot and for kernel density estimate
    xmin = 0.0
    xmax = 0.5
    ymin = 1.0
    ymax = 2.0

    # first clip x and y arrays to the specified min and max values
    x = chosen_err[val_idx]
    y = new_d4000[val_idx]
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

    # contourf plot
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    # Contour plot
    cset = ax.contour(xx, yy, f, colors='k')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=10)

    plt.show()

    return None

if __name__ == '__main__':

    # Merge if necessary
    # Comment out this block if not needed
    # It only needs to be run once. Although 
    # it wont hurt if you accidentally run it again.
    """
    for fl in sorted(glob.glob(massive_figures_dir + 'model_mockspectra_fits/' + '*_jet.npy')):
        jet_fl = np.load(fl)
        firstlight_fl = np.load(fl.replace('jet', 'firstlight'))
        merged_arr = np.concatenate((jet_fl, firstlight_fl))
        merged_fl_name = fl.replace('intermediate', 'merged')
        merged_fl_name = merged_fl_name.replace('_jet', '')
        np.save(merged_fl_name, merged_arr)
    """

    # Read in results arrays
    d4000_in = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_d4000_in_list_geq1.npy')
    #d4000_out = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_d4000_out_list_geq1.npy')
    #d4000_out_err = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_d4000_out_err_list_geq1.npy')
    mock_model_index = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_mock_model_index_list_geq1.npy')
    test_redshift = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_test_redshift_list_geq1.npy')
    mock_zgrism = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_mock_zgrism_list_geq1.npy')
    mock_zgrism_lowerr = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_mock_zgrism_lowerr_list_geq1.npy')
    mock_zgrism_higherr = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_mock_zgrism_higherr_list_geq1.npy')
    chi2 = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_chi2_list_geq1.npy')
    chosen_err = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_chosen_error_list_geq1.npy')
    new_d4000 = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_new_d4000_list_geq1.npy')
    new_d4000_err = np.load(massive_figures_dir + 'model_mockspectra_fits/intermediate_new_d4000_err_list_geq1.npy')

    # ONe problem that I noticed is that most of the recovered
    # mock zgrisms are still the same as the test redshift. This
    # is likely due to the still relatively large step in the 
    # redshift grid (0.005) when it searches for the best redshift.

    # Check that both 1D distributions of both D4000 and 
    # the model chosen error are the same as for real galaxies
    #model_d4000_vs_fluxerr(new_d4000, new_d4000_err, chosen_err)

    # --------- redshift accuracy comparison ---------- # 
    gs = gridspec.GridSpec(28,2)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.15, hspace=0.0)

    fig_gs = plt.figure(figsize=(6.5, 8))  # figsize=(width, height)

    ### first row
    # D4000 >= 1.2
    ax1 = fig_gs.add_subplot(gs[:5,:1])
    ax2 = fig_gs.add_subplot(gs[5:7,:1])
    
    # D4000 >= 1.25
    ax3 = fig_gs.add_subplot(gs[:5,1:])
    ax4 = fig_gs.add_subplot(gs[5:7,1:])
    
    # second row
    # D4000 >= 1.3
    ax5 = fig_gs.add_subplot(gs[7:12,:1])
    ax6 = fig_gs.add_subplot(gs[12:14,:1])
    
    # D4000 >= 1.35
    ax7 = fig_gs.add_subplot(gs[7:12,1:])
    ax8 = fig_gs.add_subplot(gs[12:14,1:])
    
    # third row
    # D4000 >= 1.4
    ax9 = fig_gs.add_subplot(gs[14:19,:1])
    ax10 = fig_gs.add_subplot(gs[19:21,:1])
    
    # D4000 >= 1.45
    ax11 = fig_gs.add_subplot(gs[14:19,1:])
    ax12 = fig_gs.add_subplot(gs[19:21,1:])

    # fourth row
    # D4000 >= 1.5
    ax13 = fig_gs.add_subplot(gs[21:26,:1])
    ax14 = fig_gs.add_subplot(gs[26:,:1])
    
    # D4000 >= 1.6
    ax15 = fig_gs.add_subplot(gs[21:26,1:])
    ax16 = fig_gs.add_subplot(gs[26:,1:])

    # ------------------------------
    # Create arrays for all four panels
    valid_chi2_idx = np.where(chi2 < 1.5)[0]
    d4000_sig = new_d4000 / new_d4000_err
    valid_d4000_sig_idx = np.where(d4000_sig >= 3)[0]

    d4000_gtr_1p2_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.2) & (new_d4000 < 1.25))[0]))
    d4000_gtr_1p25_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.25) & (new_d4000 < 1.3))[0]))

    d4000_gtr_1p3_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.3) & (new_d4000 < 1.35))[0]))
    d4000_gtr_1p35_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.35) & (new_d4000 < 1.4))[0]))

    d4000_gtr_1p4_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.4) & (new_d4000 < 1.45))[0]))
    d4000_gtr_1p45_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.45) & (new_d4000 < 1.5))[0]))

    d4000_gtr_1p5_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.5) & (new_d4000 < 1.6))[0]))
    d4000_gtr_1p6_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where(new_d4000 >= 1.6)[0]))

    # ------
    test_redshift_d4000_gtr_1p2 = test_redshift[d4000_gtr_1p2_idx]
    test_redshift_d4000_gtr_1p25 = test_redshift[d4000_gtr_1p25_idx]

    test_redshift_d4000_gtr_1p3 = test_redshift[d4000_gtr_1p3_idx]
    test_redshift_d4000_gtr_1p35 = test_redshift[d4000_gtr_1p35_idx]

    test_redshift_d4000_gtr_1p4 = test_redshift[d4000_gtr_1p4_idx]
    test_redshift_d4000_gtr_1p45 = test_redshift[d4000_gtr_1p45_idx]

    test_redshift_d4000_gtr_1p5 = test_redshift[d4000_gtr_1p5_idx]
    test_redshift_d4000_gtr_1p6 = test_redshift[d4000_gtr_1p6_idx]

    # ------------------------------
    mock_zgrism_d4000_gtr_1p2 = mock_zgrism[d4000_gtr_1p2_idx]
    mock_zgrism_d4000_gtr_1p25 = mock_zgrism[d4000_gtr_1p25_idx]

    mock_zgrism_d4000_gtr_1p3 = mock_zgrism[d4000_gtr_1p3_idx]
    mock_zgrism_d4000_gtr_1p35 = mock_zgrism[d4000_gtr_1p35_idx]

    mock_zgrism_d4000_gtr_1p4 = mock_zgrism[d4000_gtr_1p4_idx]
    mock_zgrism_d4000_gtr_1p45 = mock_zgrism[d4000_gtr_1p45_idx]

    mock_zgrism_d4000_gtr_1p5 = mock_zgrism[d4000_gtr_1p5_idx]
    mock_zgrism_d4000_gtr_1p6 = mock_zgrism[d4000_gtr_1p6_idx]

    # ------------------------------
    mock_zgrism_lowerr_d4000_gtr_1p2 = mock_zgrism_lowerr[d4000_gtr_1p2_idx]
    mock_zgrism_lowerr_d4000_gtr_1p25 = mock_zgrism_lowerr[d4000_gtr_1p25_idx]

    mock_zgrism_lowerr_d4000_gtr_1p3 = mock_zgrism_lowerr[d4000_gtr_1p3_idx]
    mock_zgrism_lowerr_d4000_gtr_1p35 = mock_zgrism_lowerr[d4000_gtr_1p35_idx]

    mock_zgrism_lowerr_d4000_gtr_1p4 = mock_zgrism_lowerr[d4000_gtr_1p4_idx]
    mock_zgrism_lowerr_d4000_gtr_1p45 = mock_zgrism_lowerr[d4000_gtr_1p45_idx]

    mock_zgrism_lowerr_d4000_gtr_1p5 = mock_zgrism_lowerr[d4000_gtr_1p5_idx]
    mock_zgrism_lowerr_d4000_gtr_1p6 = mock_zgrism_lowerr[d4000_gtr_1p6_idx]

    # ------------------------------
    mock_zgrism_higherr_d4000_gtr_1p2 = mock_zgrism_higherr[d4000_gtr_1p2_idx]
    mock_zgrism_higherr_d4000_gtr_1p25 = mock_zgrism_higherr[d4000_gtr_1p25_idx]

    mock_zgrism_higherr_d4000_gtr_1p3 = mock_zgrism_higherr[d4000_gtr_1p3_idx]
    mock_zgrism_higherr_d4000_gtr_1p35 = mock_zgrism_higherr[d4000_gtr_1p35_idx]

    mock_zgrism_higherr_d4000_gtr_1p4 = mock_zgrism_higherr[d4000_gtr_1p4_idx]
    mock_zgrism_higherr_d4000_gtr_1p45 = mock_zgrism_higherr[d4000_gtr_1p45_idx]

    mock_zgrism_higherr_d4000_gtr_1p5 = mock_zgrism_higherr[d4000_gtr_1p5_idx]
    mock_zgrism_higherr_d4000_gtr_1p6 = mock_zgrism_higherr[d4000_gtr_1p6_idx]

    # ------------------------------
    ax1, ax2 = plot_panel(ax1, ax2, test_redshift_d4000_gtr_1p2, mock_zgrism_d4000_gtr_1p2, mock_zgrism_lowerr_d4000_gtr_1p2, mock_zgrism_higherr_d4000_gtr_1p2, '1p2')
    ax3, ax4 = plot_panel(ax3, ax4, test_redshift_d4000_gtr_1p25, mock_zgrism_d4000_gtr_1p25, mock_zgrism_lowerr_d4000_gtr_1p25, mock_zgrism_higherr_d4000_gtr_1p25, '1p25')

    ax5, ax6 = plot_panel(ax5, ax6, test_redshift_d4000_gtr_1p3, mock_zgrism_d4000_gtr_1p3, mock_zgrism_lowerr_d4000_gtr_1p3, mock_zgrism_higherr_d4000_gtr_1p3, '1p3')
    ax7, ax8 = plot_panel(ax7, ax8, test_redshift_d4000_gtr_1p35, mock_zgrism_d4000_gtr_1p35, mock_zgrism_lowerr_d4000_gtr_1p35, mock_zgrism_higherr_d4000_gtr_1p35, '1p35')

    ax9, ax10 = plot_panel(ax9, ax10, test_redshift_d4000_gtr_1p4, mock_zgrism_d4000_gtr_1p4, mock_zgrism_lowerr_d4000_gtr_1p4, mock_zgrism_higherr_d4000_gtr_1p4, '1p4')
    ax11, ax12 = plot_panel(ax11, ax12, test_redshift_d4000_gtr_1p45, mock_zgrism_d4000_gtr_1p45, mock_zgrism_lowerr_d4000_gtr_1p45, mock_zgrism_higherr_d4000_gtr_1p45, '1p45')

    ax13, ax14 = plot_panel(ax13, ax14, test_redshift_d4000_gtr_1p5, mock_zgrism_d4000_gtr_1p5, mock_zgrism_lowerr_d4000_gtr_1p5, mock_zgrism_higherr_d4000_gtr_1p5, '1p5')
    ax15, ax16 = plot_panel(ax15, ax16, test_redshift_d4000_gtr_1p6, mock_zgrism_d4000_gtr_1p6, mock_zgrism_lowerr_d4000_gtr_1p6, mock_zgrism_higherr_d4000_gtr_1p6, '1p6')

    # add text only to the first panel
    ax1.axhline(y=0.75, xmin=0.55, xmax=0.65, ls='-', lw=2.0, color='#41ab5d')
    ax1.text(0.66, 0.28, 'Best fit line', verticalalignment='top', horizontalalignment='left', \
        transform=ax1.transAxes, color='k', size=10)

    ax1.axhline(y=0.7, xmin=0.55, xmax=0.65, ls='-', lw=2.0, color='blue')
    ax1.text(0.66, 0.2, 'Residual Mean', verticalalignment='top', horizontalalignment='left', \
        transform=ax1.transAxes, color='k', size=10)

    ax1.axhline(y=0.65, xmin=0.55, xmax=0.65, ls='-', lw=2.0, color='#3690c0')
    ax1.text(0.66, 0.12, r'$\mathrm{\pm 1\ \sigma}$', verticalalignment='top', horizontalalignment='left', \
        transform=ax1.transAxes, color='k', size=10)

    # Save figure 
    fig_gs.savefig(massive_figures_dir + \
        'model_mockspectra_fits/mock_redshift_comparison_d4000.eps', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    # -------------------------------------------------------------------------------- #
    # ------------------------------ Lower D4000 values ------------------------------ #
    # -------------------------------------------------------------------------------- #
    gs = gridspec.GridSpec(14,2)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.15, hspace=0.0)

    fig_gs = plt.figure()  # figsize=(width, height)

    ### first row
    # D4000 >= 1.0
    ax1 = fig_gs.add_subplot(gs[:5,:1])
    ax2 = fig_gs.add_subplot(gs[5:7,:1])
    
    # D4000 >= 1.05
    ax3 = fig_gs.add_subplot(gs[:5,1:])
    ax4 = fig_gs.add_subplot(gs[5:7,1:])
    
    # second row
    # D4000 >= 1.1
    ax5 = fig_gs.add_subplot(gs[7:12,:1])
    ax6 = fig_gs.add_subplot(gs[12:,:1])
    
    # D4000 >= 1.15
    ax7 = fig_gs.add_subplot(gs[7:12,1:])
    ax8 = fig_gs.add_subplot(gs[12:,1:])

    # ------------------------------
    # Create arrays for all four panels
    valid_chi2_idx = np.where(chi2 < 2.0)[0]
    d4000_sig = new_d4000 / new_d4000_err
    valid_d4000_sig_idx = np.where(d4000_sig >= 3)[0]

    d4000_gtr_1p0_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.0) & (new_d4000 < 1.05))[0]))
    d4000_gtr_1p05_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.05) & (new_d4000 < 1.1))[0]))

    d4000_gtr_1p1_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.1) & (new_d4000 < 1.15))[0]))
    d4000_gtr_1p15_idx = reduce(np.intersect1d, (valid_d4000_sig_idx, valid_chi2_idx, np.where((new_d4000 >= 1.15) & (new_d4000 < 1.2))[0]))

    # ------
    test_redshift_d4000_gtr_1p0 = test_redshift[d4000_gtr_1p0_idx]
    test_redshift_d4000_gtr_1p05 = test_redshift[d4000_gtr_1p05_idx]

    test_redshift_d4000_gtr_1p1 = test_redshift[d4000_gtr_1p1_idx]
    test_redshift_d4000_gtr_1p15 = test_redshift[d4000_gtr_1p15_idx]

    # ------------------------------
    mock_zgrism_d4000_gtr_1p0 = mock_zgrism[d4000_gtr_1p0_idx]
    mock_zgrism_d4000_gtr_1p05 = mock_zgrism[d4000_gtr_1p05_idx]

    mock_zgrism_d4000_gtr_1p1 = mock_zgrism[d4000_gtr_1p1_idx]
    mock_zgrism_d4000_gtr_1p15 = mock_zgrism[d4000_gtr_1p15_idx]

    # ------------------------------
    mock_zgrism_lowerr_d4000_gtr_1p0 = mock_zgrism_lowerr[d4000_gtr_1p0_idx]
    mock_zgrism_lowerr_d4000_gtr_1p05 = mock_zgrism_lowerr[d4000_gtr_1p05_idx]

    mock_zgrism_lowerr_d4000_gtr_1p1 = mock_zgrism_lowerr[d4000_gtr_1p1_idx]
    mock_zgrism_lowerr_d4000_gtr_1p15 = mock_zgrism_lowerr[d4000_gtr_1p15_idx]

    # ------------------------------
    mock_zgrism_higherr_d4000_gtr_1p0 = mock_zgrism_higherr[d4000_gtr_1p0_idx]
    mock_zgrism_higherr_d4000_gtr_1p05 = mock_zgrism_higherr[d4000_gtr_1p05_idx]

    mock_zgrism_higherr_d4000_gtr_1p1 = mock_zgrism_higherr[d4000_gtr_1p1_idx]
    mock_zgrism_higherr_d4000_gtr_1p15 = mock_zgrism_higherr[d4000_gtr_1p15_idx]

    # ------------------------------
    ax1, ax2 = plot_panel(ax1, ax2, test_redshift_d4000_gtr_1p0, mock_zgrism_d4000_gtr_1p0, mock_zgrism_lowerr_d4000_gtr_1p0, mock_zgrism_higherr_d4000_gtr_1p0, '1p0')
    ax3, ax4 = plot_panel(ax3, ax4, test_redshift_d4000_gtr_1p05, mock_zgrism_d4000_gtr_1p05, mock_zgrism_lowerr_d4000_gtr_1p05, mock_zgrism_higherr_d4000_gtr_1p05, '1p05')

    ax5, ax6 = plot_panel(ax5, ax6, test_redshift_d4000_gtr_1p1, mock_zgrism_d4000_gtr_1p1, mock_zgrism_lowerr_d4000_gtr_1p1, mock_zgrism_higherr_d4000_gtr_1p1, '1p1')
    ax7, ax8 = plot_panel(ax7, ax8, test_redshift_d4000_gtr_1p15, mock_zgrism_d4000_gtr_1p15, mock_zgrism_lowerr_d4000_gtr_1p15, mock_zgrism_higherr_d4000_gtr_1p15, '1p15')

    # add text only to the first panel
    ax1.axhline(y=0.75, xmin=0.55, xmax=0.65, ls='-', lw=2.0, color='#41ab5d')
    ax1.text(0.66, 0.28, 'Best fit line', verticalalignment='top', horizontalalignment='left', \
        transform=ax1.transAxes, color='k', size=10)

    ax1.axhline(y=0.7, xmin=0.55, xmax=0.65, ls='-', lw=2.0, color='blue')
    ax1.text(0.66, 0.2, 'Residual Mean', verticalalignment='top', horizontalalignment='left', \
        transform=ax1.transAxes, color='k', size=10)

    ax1.axhline(y=0.65, xmin=0.55, xmax=0.65, ls='-', lw=2.0, color='#3690c0')
    ax1.text(0.66, 0.12, r'$\mathrm{\pm 1\ \sigma}$', verticalalignment='top', horizontalalignment='left', \
        transform=ax1.transAxes, color='k', size=10)

    # Save figure 
    fig_gs.savefig(massive_figures_dir + \
        'model_mockspectra_fits/mock_redshift_comparison_lowd4000.eps', dpi=300, bbox_inches='tight')

    sys.exit(0)