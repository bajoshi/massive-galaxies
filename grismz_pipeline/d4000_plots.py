from __future__ import division

import numpy as np

import sys
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import matplotlib.colors as colors

import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck15

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
sys.path.append(massive_galaxies_dir + 'codes/')
import mag_hist as mh
import new_refine_grismz_gridsearch_parallel as ngp
import dn4000_catalog as dc
import spz_photoz_grismz_comparison as comp

# SOlution to truncate colormap from stackoverflow:
# https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_all_arrays():

    # ------------------------------- Get catalog for final sample ------------------------------- #
    # Don't use the final catalog for which all 3 redshifts were estimated
    # Use the catlaog from one step before it
    # i.e. all the matched galaxies within our z range.
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample_zrange.txt', dtype=None, names=True)

    # Assign arrays
    redshift_list = []
    d4000_list = []
    d4000_err_list = []

    # Now loop over all galaxies and get their D4000
    for i in range(len(final_sample)):
        current_id = final_sample['pearsid'][i]
        current_field = final_sample['field'][i]
        current_zspec = final_sample['specz'][i]

        grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)

        if return_code == 0:
            print current_field, current_id,
            print "Skipping due to an error with the obs data. See the error message just above this one.",
            print "Moving to the next galaxy."
            continue

        if netsig_chosen < 10:
            print current_field, current_id,
            print "Skipping due to low NetSig:", netsig_chosen
            continue

        # Now get D4000 based on zspec
        lam_em = grism_lam_obs / (1 + current_zspec)
        flam_em = grism_flam_obs * (1 + current_zspec)
        ferr_em = grism_ferr_obs * (1 + current_zspec)

        # These two checks will only be trigerred if the galaxy in question is 
        # at the correct zspec but our wavelngth data array is incomplete.
        # So it should have been in the sample if we had all the data points
        # i.e. in these cases len(lam_em) < 88
        # I've pushed the checking limits a bit inward because I also don't 
        # want the D4000 code extrapolating too much.
        if lam_em[-1] < 4150:
            print current_field, current_id,
            print "Skipping due to incomplete wavelength array."
            continue
        if lam_em[0] > 3850:
            print current_field, current_id,
            print "Skipping due to incomplete wavelength array."
            continue

        d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

        redshift_list.append(current_zspec)
        d4000_list.append(d4000)
        d4000_err_list.append(d4000_err)

    # Convert to numpy arrays and then proceed
    redshift_arr = np.asarray(redshift_list)
    d4000_arr = np.asarray(d4000_list)
    d4000_err_arr = np.asarray(d4000_err_list)

    return redshift_arr, d4000_arr, d4000_err_arr

def make_d4000_vs_redshift_plot():

    #redshift_arr, d4000_arr, d4000_err_arr = get_all_arrays()
    #ids, fields, redshift_arr, zp, zg, zspz, d4000_arr, d4000_err_arr, netsig, imag = comp.get_arrays_to_plot()

    redshift_arr = np.load(massive_figures_dir + 'redshift_arr_for_d4000_plots.npy')
    d4000_arr = np.load(massive_figures_dir + 'd4000_arr_for_d4000_plots.npy')
    d4000_err_arr = np.load(massive_figures_dir + 'd4000_err_arr_for_d4000_plots.npy')
    zspz = np.load(massive_figures_dir + 'zspz_for_d4000_plots.npy')

    # Only consider finite elements
    # I don't seem to need this for the new final sample 
    """
    valid_idx1 = np.where(np.isfinite(redshift_arr))[0]
    valid_idx2 = np.where(np.isfinite(d4000_arr))[0]
    valid_idx3 = np.where(np.isfinite(d4000_err_arr))[0]

    valid_idx = reduce(np.intersect1d, (valid_idx1, valid_idx2, valid_idx3))

    redshift_pears_plot = redshift_arr[valid_idx]
    d4000_pears_plot = d4000_arr[valid_idx]
    d4000_err_pears_plot = d4000_err_arr[valid_idx]
    """

    redshift_pears_plot = redshift_arr
    d4000_pears_plot = d4000_arr
    d4000_err_pears_plot = d4000_err_arr

    d4000_resid = (d4000_pears_plot - 1.0) / d4000_err_pears_plot
    zspz_acc = (zspz - redshift_pears_plot) / (1 + redshift_pears_plot)

    # Add a constant "calibration" error in quadrature to each D4000 error
    #const_calib_err = 0.03
    #d4000_err_pears_plot = np.sqrt(d4000_err_pears_plot**2 + const_calib_err**2)

    # First chuck the really high D4000 errors 
    # Since I found out after making the D4000 error histogram
    # that only 8 galaxies have an error > 0.5.
    valid_err_idx = np.where(d4000_err_pears_plot < 0.5)[0]
    redshift_pears_plot = redshift_pears_plot[valid_err_idx]
    d4000_pears_plot = d4000_pears_plot[valid_err_idx]
    d4000_err_pears_plot = d4000_err_pears_plot[valid_err_idx]
    d4000_resid = d4000_resid[valid_err_idx]
    zspz_acc = zspz_acc[valid_err_idx]

    # ------------------------------- Actual plotting ------------------------------- #
    # d4000 vs redshift 
    fig = plt.figure(figsize=(6, 8))
    
    gs = gridspec.GridSpec(10,1)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.1)

    ax1 = fig.add_subplot(gs[:7, :])
    ax_res = fig.add_subplot(gs[7:, :])

    print "Number of galaxies in D4000 vs redshift plot:", len(d4000_pears_plot)

    ax1.errorbar(redshift_pears_plot, d4000_pears_plot, yerr=d4000_err_pears_plot,\
    fmt='.', color='k', markeredgecolor='k', capsize=0, markersize=2.5, elinewidth=0.1)

    #ax_res.plot(redshift_pears_plot, d4000_resid, 'o', markersize=2, color='k', markeredgecolor='k')

    ax1.axhline(y=1, linewidth=2, linestyle='--', color='fuchsia', zorder=10)
    ax_res.axhline(y=0.0, linewidth=2, linestyle='--', color='fuchsia')
    ax_res.axhline(y=3.0, linewidth=2, linestyle='-.', color='fuchsia')

    # Plot average error bar for points below D4000 = 1.3
    d4000_1p3_idx = np.where(d4000_pears_plot < 1.3)[0]
    avg_1p3_err = np.mean(abs(d4000_err_pears_plot[d4000_1p3_idx]))
    print "Mean error for points below D4000=1.3:", "{:.2f}".format(avg_1p3_err)

    d4000_1p6_idx = np.where(d4000_pears_plot >= 1.6)[0]
    avg_1p6_err = np.mean(abs(d4000_err_pears_plot[d4000_1p6_idx]))
    print "Mean error for points above D4000=1.6:", "{:.2f}".format(avg_1p6_err)

    ax1.axhline(y=1.0 + 3*avg_1p3_err, linewidth=2, linestyle='-.', color='fuchsia', zorder=10)
    ax1.errorbar(1.31, 1.1, yerr=avg_1p3_err, fmt='.', color='k', \
        markeredgecolor='k', capsize=0, markersize=6, elinewidth=0.5, ecolor='r')

    d4000_1p3_to_1p6_idx = np.where((d4000_pears_plot >= 1.3) & (d4000_pears_plot < 1.6))[0]

    #ax_res.scatter(redshift_pears_plot[d4000_1p3_idx], d4000_resid[d4000_1p3_idx], s=40, facecolor='None', edgecolors='orange')
    #ax_res.scatter(redshift_pears_plot[d4000_1p3_to_1p6_idx], d4000_resid[d4000_1p3_to_1p6_idx], s=30, facecolor='None', edgecolors='orange')
    #ax_res.scatter(redshift_pears_plot[d4000_1p6_idx], d4000_resid[d4000_1p6_idx], s=30, facecolor='None', edgecolors='red')

    vmin_level = 1.0
    vmax_level = 1.8

    cmap = plt.get_cmap('jet')
    new_cmap = truncate_colormap(cmap, 0.15, 0.82)

    nipy_spectral_cmap = plt.get_cmap('nipy_spectral')
    trunc_nipy_spec = truncate_colormap(nipy_spectral_cmap, 0.1, 0.9)

    c = ax_res.scatter(redshift_pears_plot, d4000_resid, s=20, facecolor='None', c=d4000_pears_plot, vmin=vmin_level, vmax=vmax_level, cmap=trunc_nipy_spec)
    # add colorbar inside figure
    cbaxes = inset_axes(ax_res, width='3%', height='75%', loc=7, bbox_to_anchor=[-0.05, 0.06, 1, 1], bbox_transform=ax_res.transAxes)
    cb = fig.colorbar(c, cax=cbaxes, ticks=np.arange(1.0, 2.0, 0.2), orientation='vertical')
    #cb.ax.get_children()[0].set_linewidths(10.0)
    cb.ax.set_ylabel(r'$\mathrm{D4000}$', fontsize=14, labelpad=-42)  # This has to be cb.ax NOT cb.ax_res

    d4000_3sigma = np.where(d4000_resid >= 3.0)[0]
    print len(d4000_3sigma), "out of", len(d4000_resid), "galaxies have D4000 measured at 3-sigma or better."
    print "Mean \"sigma\" level for galaxies with D4000<1.3:", np.mean(abs(d4000_resid[d4000_1p3_idx]))
    print "Mean \"sigma\" level for galaxies with D4000>=1.6:", np.mean(abs(d4000_resid[d4000_1p6_idx]))

    five_sig_idx = np.where(abs(d4000_resid >= 5.0))[0]
    print "Mean D4000 value for galaxies which have D4000 measurement at 5sigma or higher:", np.mean(d4000_pears_plot[five_sig_idx])

    # labels and grid
    ax1.set_ylabel(r'$\mathrm{D}4000$', fontsize=15)
    ax1.grid(True, color=mh.rgb_to_hex(240, 240, 240))

    ax_res.set_xlabel(r'$\mathrm{Redshift}$', fontsize=15)
    ax_res.set_ylabel(r'$\mathrm{\frac{D4000 - 1.0}{\sigma_{D4000}}}$', fontsize=15)

    # parallel x axis for age of the Universe
    # This solution came from 
    # http://www.astropy.org/astropy-tutorials/edshift_plot.html
    ax2 = ax1.twiny()

    ages = np.arange(3,9,0.5)*u.Gyr
    ageticks = [z_at_value(Planck15.age, age) for age in ages]
    ax2.set_xticks(ageticks)

    ages_ticklabels = ['{:g}'.format(age) for age in ages.value]
    ax2.set_xticklabels(ages_ticklabels)

    ax2.set_xlim(0.55, 1.35)
    ax1.set_xlim(0.55, 1.35)
    ax1.set_ylim(0.5, 2.05)

    ax_res.set_ylim(-2, 15)
    ax_res.set_yticks(np.arange(-2, 16, 2))
    ax_res.set_xlim(0.55, 1.35)

    ax2.set_xlabel(r'$\mathrm{Time\ since\ Big\ Bang\ (Gyr)}$', fontsize=15)

    # Turen off xaxis tick labels
    ax1.set_xticklabels([])

    # Turn on minor ticks
    ax1.minorticks_on()  # Only ax1 and not ax2. See comment below.
    ax2.minorticks_off()

    """
    Struggled iwth the following error for a couple days (!!)
    Everytime I tried to show the figure or save it, I got the error:
    /Users/bhavinjoshi/anaconda/lib/python2.7/site-packages/matplotlib/ticker.py:2531: 
    RuntimeWarning: invalid value encountered in log10
    x = int(np.round(10 ** (np.log10(majorstep) % 1)))
    ValueError: cannot convert float NaN to integer

    Later, I turned off the plotting of minor ticks that I'd turned on
    in my matplotlibrc by default and it worked. I think, the reason for 
    it to fail with that error is because I was giving it these "weird" 
    locations for the major ticks on ax2 (age ticks) and then when I 
    asked to turn on minor ticks on the ax2 axis it has no clue where 
    to put them.
    """

    # save the figure
    fig.savefig(massive_figures_dir + 'd4000_redshift.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    # ------------------------------------------------------
    # I'd also like to see the histogram of D4000 errors
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get total bins and plot histogram
    iqr = np.std(d4000_err_pears_plot, dtype=np.float64)
    binsize = 2*iqr*np.power(len(d4000_err_pears_plot),-1/3)
    totalbins = int(np.floor((max(d4000_err_pears_plot) - min(d4000_err_pears_plot))/binsize))

    print "Min and Max D4000 error in histogram:", min(d4000_err_pears_plot), max(d4000_err_pears_plot)

    print "Total Bins:", totalbins
    ax.hist(d4000_err_pears_plot, totalbins, histtype='step', range=[0, 0.5])

    # Labels
    ax.set_xlabel(r'$\mathrm{\sigma_{D4000}}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    # Make tick labels larger
    ax.set_xticklabels(ax.get_xticks().tolist(), size=10)
    ax.set_yticklabels(ax.get_yticks().tolist(), size=10)

    # save the figure
    fig.savefig(massive_figures_dir + 'd4000_error_hist.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    # ------------------------------------------------------
    # Also plot the D4000 significance vs D4000
    fig = plt.figure()
    ax = fig.add_subplot(111)

    vmin_level = 0
    vmax_level = 0.046

    c = ax.scatter(d4000_pears_plot, d4000_resid, s=10, c=abs(zspz_acc), vmin=vmin_level, vmax=vmax_level, cmap=new_cmap)
    # add colorbar inside figure
    cbaxes = inset_axes(ax, width='3%', height='55%', loc=2, bbox_to_anchor=[0.02, -0.03, 1, 1], bbox_transform=ax.transAxes)
    cb = fig.colorbar(c, cax=cbaxes, ticks=np.arange(vmin_level, vmax_level + 0.01, 0.01), orientation='vertical')
    cb.ax.set_ylabel(r'$\mathrm{\left| \frac{z_{spz} - z_s}{1 + z_s} \right|}$', fontsize=14, rotation=0, labelpad=25)
    # Default rotation for the y label of a vertical colorbar is 90
    # So I have to force it to be 0 to see it upright

    # Lbaels
    ax.set_xlabel(r'$\mathrm{D4000}$')
    ax.set_ylabel(r'$\mathrm{\frac{D4000 - 1.0}{\sigma_{D4000}}}$')

    # Limits and ticks
    ax.set_ylim(-2, 14)
    ax.set_yticks(np.arange(-2, 15, 2))
    ax.set_xlim(0.6, 2.05)

    # Horizontal and vertical lines
    ax.axhline(y=0.0, ls='--', color='k')
    ax.axvline(x=1.0, ls='--', color='k')

    # save the figure
    fig.savefig(massive_figures_dir + 'd4000_vs_significance.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    # ------------------------------------------------------
    # Also plot the D4000 significance vs redshift accuracy
    fig = plt.figure()
    ax = fig.add_subplot(111)

    c = ax.scatter(d4000_resid, zspz_acc, s=7, color='k')
    # add colorbar inside figure
    #cbaxes = inset_axes(ax, width='3%', height='55%', loc=2, bbox_to_anchor=[0.02, -0.03, 1, 1], bbox_transform=ax.transAxes)
    #cb = fig.colorbar(c, cax=cbaxes, ticks=[vmin_level, vmax_level], orientation='vertical')
    #cb.ax.set_ylabel(r'$\mathrm{\left| \frac{z_{spz} - z_s}{1 + z_s} \right|}$', fontsize=14, rotation=0)
    # Default rotation for the y label of a vertical colorbar is 90
    # So I have to force it to be 0 to see it upright

    # Lbaels
    ax.set_xlabel(r'$\mathrm{\frac{D4000 - 1.0}{\sigma_{D4000}}}$')
    ax.set_ylabel(r'$\mathrm{\left| \frac{z_{spz} - z_s}{1 + z_s} \right|}$')

    # Limits and ticks
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlim(-2, 13)
    ax.set_xticks(np.arange(-2, 15, 2))

    # Horizontal and vertical lines
    ax.axhline(y=0.0, ls='--', color='darkblue')
    #ax.axvline(x=1.0, ls='--', color='darkblue')

    # save the figure
    fig.savefig(massive_figures_dir + 'z_acc_vs_significance.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

def make_d4000_hist():

    # get arrays
    #redshift_arr, d4000_arr, d4000_err_arr = get_all_arrays()

    redshift_arr = np.load(massive_figures_dir + 'redshift_arr_for_d4000_plots.npy')
    d4000_arr = np.load(massive_figures_dir + 'd4000_arr_for_d4000_plots.npy')
    d4000_err_arr = np.load(massive_figures_dir + 'd4000_err_arr_for_d4000_plots.npy')

    print "Number of galaxies with D4000 measurements:", len(d4000_arr)

    # Only consider finite elements
    valid_idx = np.where(np.isfinite(d4000_arr))[0]

    # Again chuck the really high D4000 errors 
    # Since I found out after making the D4000 error histogram
    # that only 8 galaxies have an error > 0.5.
    valid_err_idx = np.where(d4000_err_arr < 0.5)[0]

    comb_idx = reduce(np.intersect1d, (valid_idx, valid_err_idx))

    d4000_pears_plot = d4000_arr[comb_idx]
    d4000_err_plot = d4000_err_arr[comb_idx]

    # ----------------------- PLOT ----------------------- #
    # PEARS dn4000 histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get total bins and plot histogram
    iqr = np.std(d4000_pears_plot, dtype=np.float64)
    binsize = 2*iqr*np.power(len(d4000_pears_plot),-1/3)
    totalbins = int(np.floor((max(d4000_pears_plot) - min(d4000_pears_plot))/binsize))

    print "Total Bins:", totalbins
    print "Number of galaxies with valid D4000 measurements in histogram:", len(d4000_pears_plot)

    ncount, edges, patches = ax.hist(d4000_pears_plot, 50, range=[0.0,2.5], color='lightgray', align='mid', zorder=10)
    ax.grid(True, color=mh.rgb_to_hex(240, 240, 240))

    # shade the selection region
    edges_plot = np.where(edges >= 1.1)[0]
    patches_plot = [patches[edge_ind] for edge_ind in edges_plot[:-1]]
    # I put in the [:-1] because for some reason edges was 1 element longer than patches
    col = np.full(len(patches_plot), 'lightblue', dtype='|S9')
    # make sure the length of the string given in the array initialization is the same as the color name
    for c, p in zip(col, patches_plot):
        plt.setp(p, 'facecolor', c)

    ax.set_xlabel(r'$\mathrm{D}4000$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    ax.set_xlim(0.0, 2.5)
    ax.minorticks_on()

    d40001p1 = np.where(d4000_pears_plot >= 1.1)[0]
    print "Number of galaxies with D4000 >= 1.1:", len(d40001p1)
    print "Fraction of total galaxies with D4000 >= 1.1:", len(d40001p1) / len(d4000_pears_plot)

    #print np.where(d4000_err_plot[d40001p1] < 0.1)
    print "Total galaxies with errors < 0.1:", len(np.where(d4000_err_plot[d40001p1] < 0.1)[0])

    # save figure
    fig.savefig(massive_figures_dir + 'pears_d4000_hist.pdf', dpi=300, bbox_inches='tight')

    return None

def make_redshift_hist():

    # read arrays
    zgrism_n = np.load(massive_figures_dir + 'full_run/zgrism_list_gn.npy')
    zgrism_s = np.load(massive_figures_dir + 'full_run/zgrism_list_gs.npy')

    print len(zgrism_n), len(zgrism_s)

    # concatenate
    redshift_pears_arr = np.concatenate((zgrism_n, zgrism_s))

    # Only consider finite elements
    valid_idx = np.where(np.isfinite(redshift_pears_arr))[0]
    redshift_pears_plot = redshift_pears_arr[valid_idx]

    # ----------------------- PLOT ----------------------- #
    # PEARS dn4000 histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get total bins and plot histogram
    iqr = np.std(redshift_pears_plot, dtype=np.float64)
    binsize = 2*iqr*np.power(len(redshift_pears_plot),-1/3)
    totalbins = np.floor((max(redshift_pears_plot) - min(redshift_pears_plot))/binsize)

    ncount, edges, patches = ax.hist(redshift_pears_plot, totalbins, \
        color='red', histtype='step', align='mid', zorder=10)
    ax.grid(True, color=mh.rgb_to_hex(240, 240, 240))

    ax.set_xlabel(r'$\mathrm{z_{grism}}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    # save figure
    fig.savefig(massive_figures_dir + 'pears_redshift_hist.eps', dpi=300, bbox_inches='tight')

    return None

if __name__ == '__main__':

    """
    ids, fields, redshift_arr, zp, zg, zspz, d4000_arr, d4000_err_arr, netsig, imag = comp.get_arrays_to_plot()

    np.save(massive_figures_dir + 'redshift_arr_for_d4000_plots.npy', redshift_arr)
    np.save(massive_figures_dir + 'd4000_arr_for_d4000_plots.npy', d4000_arr)
    np.save(massive_figures_dir + 'd4000_err_arr_for_d4000_plots.npy', d4000_err_arr)
    np.save(massive_figures_dir + 'zspz_for_d4000_plots.npy', zspz)
    """
    
    make_d4000_vs_redshift_plot()
    #make_d4000_hist()
    #make_redshift_hist()

    sys.exit(0)