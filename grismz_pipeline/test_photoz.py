from __future__ import division

import numpy as np
from astropy.stats import mad_std
from scipy.optimize import curve_fit

import sys
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
figs_dir = home + "/Desktop/FIGS/"
massive_figures_dir = figs_dir + 'massive-galaxies-figures/'

def line_func(x, slope, intercept):
    return slope*x + intercept

def make_z_comparison_plot(z1, z2, fig, ax1, ax2, zmin, zmax, z_resid_min, z_resid_max, ylab1, ylab2, title, d4000_thresh_low, d4000_thresh_high):

    # Check for NaNs and remove them
    val_idx1 = np.where(np.isfinite(z1))
    val_idx2 = np.where(np.isfinite(z2))
    val_idx = reduce(np.intersect1d, (val_idx1, val_idx2))

    z1 = z1[val_idx]
    z2 = z2[val_idx]

    # Labels
    ax1.set_ylabel(ylab1, fontsize=14, labelpad=-1)
    ax2.set_ylabel(ylab2, fontsize=14, labelpad=-1)
    ax2.set_xlabel(r'$\mathrm{z_{spec}}$', fontsize=14)

    ax1.set_title(title, fontsize=14)

    # plto the redshifts vs each other and residuals
    ax1.plot(z1, z2, 'o', markersize=4, color='k', zorder=10)
    ax1.plot(np.linspace(zmin, zmax, 20), np.linspace(zmin, zmax, 20), '--', color='gray')

    resid = (z1 - z2)/(1+z1)
    ax2.plot((z1), resid, 'o', markersize=4, color='k', zorder=10)
    ax2.axhline(y=0.0, ls='--', color='gray')

    # plot limits
    ax1.set_xlim(zmin, zmax)
    ax1.set_ylim(zmin, zmax)

    ax2.set_xlim(zmin, zmax)
    ax2.set_ylim(z_resid_min, z_resid_max)

    # dont show xtick labels on upper panel
    ax1.set_xticklabels([])

    # Compute mean, std, and sigma_nmad of the residuals and put on plot
    mean = np.mean(resid)
    std = np.std(resid)
    sigma_nmad = mad_std(resid)

    print "{:.4}".format(mean), "{:.4}".format(std), "{:.4}".format(sigma_nmad)

    # Fitting
    # do the fit with scipy
    popt, pcov = curve_fit(line_func, z1, z2, p0=[1.0, 0.6])

    # plot line fit
    x_plot = np.arange(0.2,1.6,0.01)
    ax1.plot(x_plot, line_func(x_plot, popt[0], popt[1]), '-', color='#41ab5d', linewidth=2.0)
    #ax1.plot(x_plot, line_func(x_plot, popt[0] + sigma_nmad, popt[1]), '-', color='#3690c0', linewidth=2.0)
    #ax1.plot(x_plot, line_func(x_plot, popt[0] - sigma_nmad, popt[1]), '-', color='#3690c0', linewidth=2.0)

    ax2.axhline(y=mean + sigma_nmad, ls='-', color='#3690c0', linewidth=1.5)
    ax2.axhline(y=mean - sigma_nmad, ls='-', color='#3690c0', linewidth=1.5)
    ax2.axhline(y=mean, ls='-', color='#41ab5d', linewidth=1.5)

    # text
    ax1.text(0.05, 0.95, 'N = ' + str(len(z1)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)
    ax1.text(0.05, 0.89, r'$\left< \mathrm{\Delta z / (1+z)} \right>\, =\,$' + "{:.1e}".format(mean), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)
    ax1.text(0.05, 0.83, r'$\sigma^\mathrm{NMAD}_\mathrm{\Delta z / (1+z)} \, =\,$' + "{:.1e}".format(sigma_nmad), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)

    # Larger ticklabels
    ax1.set_yticklabels(ax1.get_yticks().tolist(), size=10)
    ax2.set_xticklabels(ax2.get_xticks().tolist(), size=10)
    ax2.set_yticklabels(ax2.get_yticks().tolist(), size=10)

    return None

if __name__ == '__main__':

    # Read in file copied from terminal
    fh = open(home + '/Desktop/test_photoz_output.txt', 'r')

    # also read in IDs and D4000 from previous grism+photometry run
    ids_gn = np.load(massive_figures_dir + 'large_diff_specz_sample/run_with_1xerrors/' + 'withemlines_id_list_gn.npy')
    ids_gs = np.load(massive_figures_dir + 'large_diff_specz_sample/run_with_1xerrors/' + 'withemlines_id_list_gs.npy')

    d4000_gn = np.load(massive_figures_dir + 'large_diff_specz_sample/run_with_1xerrors/' + 'withemlines_d4000_list_gn.npy')
    d4000_gs = np.load(massive_figures_dir + 'large_diff_specz_sample/run_with_1xerrors/' + 'withemlines_d4000_list_gs.npy')

    # assign lists for redshifts
    specz = []
    photoz_minchi2 = []
    photoz_3d = []
    photoz_wt = []
    ids = []
    fields = []
    
    # loop over all lines and get redshifts
    for line in fh.readlines():

        # check id and get grism+photometry redshift from previous run
        if 'At ID' in line:
            current_id = int(line.split()[2])
            current_field = line.split()[4]

            ids.append(current_id)
            fields.append(current_field)

        # Now get other redshifts
        # this is printed after the ID so it has to be checked after first getting the SPZ
        if 'Ground-based spectroscopic redshift' in line:
            specz.append(float(line.split(':')[1]))
        elif 'Previous photometric redshift from 3DHST' in line:
            photoz_3d.append(float(line.split(':')[1]))
        elif 'Photometric redshift from min chi2' in line:
            photoz_minchi2.append(float(line.split(':')[1]))
        elif 'Photometric redshift (weighted)' in line:
            photoz_wt.append(float(line.split(':')[1]))

    # convert to numpy arrays for plotting
    specz = np.asarray(specz)
    photoz_minchi2 = np.asarray(photoz_minchi2)
    photoz_3d = np.asarray(photoz_3d)
    photoz_wt = np.asarray(photoz_wt)
    ids = np.asarray(ids)
    fields = np.asarray(fields)

    # Now loop again 
    # This loop has to be done separately to get the SPZ
    # becasue in the last run not all galaxies were considered
    # The previous run seems to have only D4000 >= 1.1
    spz = []
    specz_for_spz = []
    photoz_3d_for_spz = []
    photoz_wt_for_spz = []
    d4000 = []

    d4000_thresh_low = 1.4
    d4000_thresh_high = 2.5
    for i in range(len(ids)):
        current_id = ids[i]
        current_field = fields[i]

        # get pz and zrange
        try:
            pz = np.load(massive_figures_dir + 'large_diff_specz_sample/run_with_1xerrors/' + current_field + '_' + str(current_id) + '_pz.npy')
            z_arr = np.load(massive_figures_dir + 'large_diff_specz_sample/run_with_1xerrors/' + current_field + '_' + str(current_id) + '_z_arr.npy')
        except IOError:
            continue

        # Get D4000 from previous run
        # Be careful!! Indices are different compared to the next codeblock which uses where
        if current_field == 'GOODS-N':
            d4000_idx = np.where(ids_gn == current_id)[0]
            current_d4000 = d4000_gn[d4000_idx]
        elif current_field == 'GOODS-S':
            d4000_idx = np.where(ids_gs == current_id)[0]
            current_d4000 = d4000_gs[d4000_idx]

        # Check d4000 threshold
        if (current_d4000 >= d4000_thresh_low) and (current_d4000 < d4000_thresh_high):
            # Find ID and corresponding other redshifts
            spz.append(np.sum(pz*z_arr))
            idx = np.where((ids == current_id) & (fields == current_field))[0][0]
            specz_for_spz.append(specz[idx])
            photoz_3d_for_spz.append(photoz_3d[idx])
            photoz_wt_for_spz.append(photoz_wt[idx])
        else:
            continue

    # again convert to numpy arrays
    specz_for_spz = np.asarray(specz_for_spz)
    spz = np.asarray(spz)
    photoz_3d_for_spz = np.asarray(photoz_3d_for_spz)
    photoz_wt_for_spz = np.asarray(photoz_wt_for_spz)
    print "\n", "D4000 range considered:", d4000_thresh_low, "<= D4000 <", d4000_thresh_high 
    print "Total galaxies:", len(spz)

    # make plot
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(10, 24)
    gs.update(left=0.05, right=0.95, wspace=0.25, hspace=0.0)

    # for photo-z
    ax1 = fig.add_subplot(gs[:7,:11])
    ax2 = fig.add_subplot(gs[7:,:11])
    
    # for spz
    ax3 = fig.add_subplot(gs[:7,13:])
    ax4 = fig.add_subplot(gs[7:,13:])

    make_z_comparison_plot(specz_for_spz, photoz_wt_for_spz, fig, ax1, ax2, 0.5, 1.3, -0.1, 0.1, r'$\mathrm{z_{phot}}$', r'$\mathrm{(z_{spec} - z_{phot}) / (1+z_{spec})}$', 'Photo-z', d4000_thresh_low, d4000_thresh_high)
    make_z_comparison_plot(specz_for_spz, spz, fig, ax3, ax4, 0.5, 1.3, -0.1, 0.1, r'$\mathrm{z_{spz}}$', r'$\mathrm{(z_{spec} - z_{spz}) / (1+z_{spec})}$', 'SPZ', d4000_thresh_low, d4000_thresh_high)

    ax1.text(0.95, 1.1, "{:.1f}".format(d4000_thresh_low) + r"$\, \leq \mathrm{D4000} < \,$" + "{:.1f}".format(d4000_thresh_high), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=16)

    fig.savefig(massive_figures_dir + 'zcomp_myphot_spz_' + str(d4000_thresh_low) + 'd4000' + str(d4000_thresh_high) + '.png', dpi=300, bbox_inches='tight')

    fh.close()
    sys.exit(0)