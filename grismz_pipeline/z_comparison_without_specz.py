from __future__ import division

import numpy as np
from astropy.stats import mad_std
from scipy.integrate import simps
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
zg_results_dir = massive_figures_dir + 'grismz_run_jan2019/'
zp_results_dir = massive_figures_dir + 'photoz_run_jan2019/'

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import mocksim_results as mr
import spz_photoz_grismz_comparison as comp

def make_z_comparison_plot(resid_zpzg, resid_zpzspz, zp_for_zg, zg_for_zp, zp_for_zspz, zspz_for_zp, \
    mean_zpzg, nmad_zpzg, mean_zpzspz, nmad_zpzspz, d4000_low, d4000_high, \
    outlier_idx_zpzg, outlier_idx_zpzspz, outlier_frac_zpzg, outlier_frac_zpzspz):

    # Define figure
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(10,28)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:7, :12])
    ax2 = fig.add_subplot(gs[7:, :12])

    ax3 = fig.add_subplot(gs[:7, 16:])
    ax4 = fig.add_subplot(gs[7:, 16:])

    # Plot stuff
    ax1.plot(zp_for_zg, zg_for_zp, 'o', markersize=2, color='k', markeredgecolor='k')
    ax1.scatter(zp_for_zg[outlier_idx_zpzg], zg_for_zp[outlier_idx_zpzg], s=20, facecolor='white', edgecolors='gray', zorder=5)
    ax2.plot(zp_for_zg, resid_zpzg, 'o', markersize=2, color='k', markeredgecolor='k')
    ax2.scatter(zp_for_zg[outlier_idx_zpzg], resid_zpzg[outlier_idx_zpzg], s=20, facecolor='white', edgecolors='gray', zorder=5)

    ax3.plot(zspz_for_zp, zp_for_zspz, 'o', markersize=2, color='k', markeredgecolor='k')
    ax3.scatter(zspz_for_zp[outlier_idx_zpzspz], zp_for_zspz[outlier_idx_zpzspz], s=20, facecolor='white', edgecolors='gray', zorder=5)
    ax4.plot(zspz_for_zp, resid_zpzspz, 'o', markersize=2, color='k', markeredgecolor='k')
    ax4.scatter(zspz_for_zp[outlier_idx_zpzspz], resid_zpzspz[outlier_idx_zpzspz], s=20, facecolor='white', edgecolors='gray', zorder=5)

    # Other lines on plot
    ax2.axhline(y=0.0, ls='-', color='gray')
    ax4.axhline(y=0.0, ls='-', color='gray')

    # do the fit with scipy
    popt_zpzg, pcov_zpzg = curve_fit(comp.line_func, zp_for_zg, zg_for_zp, p0=[1.0, 0.6])
    popt_zpzspz, pcov_zpzspz = curve_fit(comp.line_func, zspz_for_zp, zp_for_zspz, p0=[1.0, 0.6])

    # plot line fit
    x_plot = np.arange(0.1,1.61,0.01)

    zpzg_mean_line = comp.line_func(x_plot, popt_zpzg[0], popt_zpzg[1])
    zpzspz_mean_line = comp.line_func(x_plot, popt_zpzspz[0], popt_zpzspz[1])

    ax1.plot(x_plot, x_plot, '-', color='gray')
    ax1.plot(x_plot, zpzg_mean_line, '--', color='darkblue', lw=1)
    ax1.plot(x_plot, (1+nmad_zpzg)*popt_zpzg[0]*x_plot + nmad_zpzg + popt_zpzg[1], ls='--', color='red', lw=1)
    ax1.plot(x_plot, (1-nmad_zpzg)*popt_zpzg[0]*x_plot - nmad_zpzg + popt_zpzg[1], ls='--', color='red', lw=1)

    ax3.plot(x_plot, x_plot, '-', color='gray')
    ax3.plot(x_plot, zpzspz_mean_line, '--', color='darkblue', lw=1)
    ax3.plot(x_plot, (1+nmad_zpzspz)*popt_zpzspz[0]*x_plot + nmad_zpzspz + popt_zpzspz[1], ls='--', color='red', lw=1)
    ax3.plot(x_plot, (1-nmad_zpzspz)*popt_zpzspz[0]*x_plot - nmad_zpzspz + popt_zpzspz[1], ls='--', color='red', lw=1)

    ax2.axhline(y=mean_zpzg, ls='--', color='darkblue', lw=1)
    ax2.axhline(y=mean_zpzg + nmad_zpzg, ls='--', color='red', lw=1)
    ax2.axhline(y=mean_zpzg - nmad_zpzg, ls='--', color='red', lw=1)

    ax4.axhline(y=mean_zpzspz, ls='--', color='darkblue', lw=1)
    ax4.axhline(y=mean_zpzspz + nmad_zpzspz, ls='--', color='red', lw=1)
    ax4.axhline(y=mean_zpzspz - nmad_zpzspz, ls='--', color='red', lw=1)

    # Get rid of Xaxis tick labels on top subplot
    ax1.set_xticklabels([])
    ax3.set_xticklabels([])

    # Minor ticks
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    ax4.minorticks_on()

    # Limits
    ax1.set_xlim(0.1, 1.6)
    ax1.set_ylim(0.1, 1.6)

    ax2.set_xlim(0.1, 1.6)
    ax2.set_ylim(-0.25, 0.25)

    ax3.set_xlim(0.1, 1.6)
    ax3.set_ylim(0.1, 1.6)

    ax4.set_xlim(0.1, 1.6)
    ax4.set_ylim(-0.25, 0.25)

    # Axis labels
    ax1.set_ylabel(r'$\mathrm{z_{g}}$', fontsize=13)
    ax2.set_xlabel(r'$\mathrm{z_{p}}$', fontsize=13)
    ax2.set_ylabel(r'$\mathrm{(z_{p} - z_{g}) / (1+z_{p})}$', fontsize=13)

    ax3.set_ylabel(r'$\mathrm{z_{p}}$', fontsize=13)
    ax4.set_xlabel(r'$\mathrm{z_{spz}}$', fontsize=13)
    ax4.set_ylabel(r'$\mathrm{(z_{p} - z_{spz}) / (1+z_{spz})}$', fontsize=13)
 
    # print D4000 range
    ax1.text(0.45, 0.11, "{:.1f}".format(d4000_low) + r"$\, \leq \mathrm{D4000} < \,$" + "{:.1f}".format(d4000_high), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=15, zorder=10)

    # print N, mean, nmad, outlier frac
    ax1.text(0.05, 0.97, r'$\mathrm{N = }$' + str(len(resid_zpzg)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12, zorder=10)
    ax1.text(0.04, 0.89, r'${\left< \Delta \right>} = $' + mr.convert_to_sci_not(mean_zpzg), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12, zorder=10)
    ax1.text(0.05, 0.79, r'$\mathrm{\sigma^{NMAD}} = $' + mr.convert_to_sci_not(nmad_zpzg), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12, zorder=10)
    ax1.text(0.05, 0.7, r'$\mathrm{Out\ frac\, =\, }$' + str("{:.2f}".format(outlier_frac_zpzg)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12, zorder=10)

    ax3.text(0.05, 0.97, r'$\mathrm{N = }$' + str(len(resid_zpzspz)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12, zorder=10)
    ax3.text(0.04, 0.89, r'${\left< \Delta \right>} = $' + mr.convert_to_sci_not(mean_zpzspz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12, zorder=10)
    ax3.text(0.05, 0.79, r'$\mathrm{\sigma^{NMAD}} = $' + mr.convert_to_sci_not(nmad_zpzspz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12, zorder=10)
    ax3.text(0.05, 0.7, r'$\mathrm{Out\ frac\, =\, }$' + str("{:.2f}".format(outlier_frac_zpzspz)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12, zorder=10)

    # Save figure
    fig.savefig(massive_figures_dir + 'photoz_nospecz_comp_' + \
        str(d4000_low).replace('.','p') + 'to' + str(d4000_high).replace('.','p') + '.pdf', \
        dpi=300, bbox_inches='tight')

    return None

def main():

    ids, fields, zs, zp, zg, zspz, d4000, d4000_err, netsig, imag = comp.get_arrays_to_plot()

    # Just making sure that all returned arrays have the same length.
    # Essential since I'm doing "where" operations below.
    assert len(ids) == len(fields)
    assert len(ids) == len(zs)
    assert len(ids) == len(zp)
    assert len(ids) == len(zg)
    assert len(ids) == len(zspz)
    assert len(ids) == len(d4000)
    assert len(ids) == len(d4000_err)
    assert len(ids) == len(netsig)
    assert len(ids) == len(imag)

    # Cut on D4000
    d4000_low = 1.6
    d4000_high = 2.0
    d4000_idx = np.where((d4000 >= d4000_low) & (d4000 < d4000_high) & (d4000_err < 0.5))[0]

    print "\n", "D4000 range:   ", d4000_low, "<= D4000 <", d4000_high, "\n"
    print "Galaxies within D4000 range:", len(d4000_idx)

    # Apply D4000 and magnitude indices
    #zs = zs[d4000_idx]  # This code does not need the specz so I'm not going to use that from now on.
    zp = zp[d4000_idx]
    zg = zg[d4000_idx]
    zspz = zspz[d4000_idx]

    d4000 = d4000[d4000_idx]
    d4000_err = d4000_err[d4000_idx]
    netsig = netsig[d4000_idx]

    d4000_resid = (d4000 - 1.0) / d4000_err

    # Get residuals 
    resid_zpzg = (zp - zg) / (1 + zp)
    resid_zpzspz = (zp - zspz) / (1 + zspz)

    # Make sure they are finite
    valid_idx1 = np.where(np.isfinite(resid_zpzg))[0]
    valid_idx2 = np.where(np.isfinite(resid_zpzspz))[0]

    # Apply indices
    resid_zpzg = resid_zpzg[valid_idx1]
    zp_for_zg = zp[valid_idx1]
    zg_for_zp = zg[valid_idx1]

    resid_zpzspz = resid_zpzspz[valid_idx2]
    zp_for_zspz = zp[valid_idx2]
    zspz_for_zp = zspz[valid_idx2]

    print "Number of galaxies in photo-z vs grism-z plot:", len(valid_idx1)
    print "Number of galaxies in photo-z vs SPZ plot:", len(valid_idx2)

    # Print info
    mean_zpzg = np.mean(resid_zpzg)
    std_zpzg = np.std(resid_zpzg)
    nmad_zpzg = mad_std(resid_zpzg)

    mean_zpzspz = np.mean(resid_zpzspz)
    std_zpzspz = np.std(resid_zpzspz)
    nmad_zpzspz = mad_std(resid_zpzspz)

    print "Mean, std dev, and Sigma_NMAD for residuals for Photo-z vs Grism-z:", \
    "{:.3f}".format(mean_zpzg), "{:.3f}".format(std_zpzg), "{:.3f}".format(nmad_zpzg)
    print "Mean, std dev, and Sigma_NMAD for residuals for Photo-z vs SPZ:", \
    "{:.3f}".format(mean_zpzspz), "{:.3f}".format(std_zpzspz), "{:.3f}".format(nmad_zpzspz)

    # Compute catastrophic failures
    # i.e., How many galaxies are outside +-3-sigma given the sigma above?
    outlier_idx_zpzg = np.where(abs(resid_zpzg) - mean_zpzg > 3*nmad_zpzg)[0]
    outlier_idx_zpzspz = np.where(abs(resid_zpzspz) - mean_zpzspz > 3*nmad_zpzspz)[0]

    outlier_frac_zpzg = len(outlier_idx_zpzg) / len(resid_zpzg)
    outlier_frac_zpzspz = len(outlier_idx_zpzspz) / len(resid_zpzspz)

    print "Outlier fraction for Photo-z vs Grism-z:", outlier_frac_zpzg
    print "Outlier fraction for Photo-z vs SPZ:", outlier_frac_zpzspz

    # Make plot
    make_z_comparison_plot(resid_zpzg, resid_zpzspz, zp_for_zg, zg_for_zp, zp_for_zspz, zspz_for_zp, \
        mean_zpzg, nmad_zpzg, mean_zpzspz, nmad_zpzspz, d4000_low, d4000_high, \
        outlier_idx_zpzg, outlier_idx_zpzspz, outlier_frac_zpzg, outlier_frac_zpzspz)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)