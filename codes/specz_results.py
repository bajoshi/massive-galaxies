from __future__ import division

import numpy as np

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

def constrain(spec_cat):
    """
    This function will constrain the galaxy spectra to be plotted for comparison
    based on their D4000, NetSig, and overall error.
    """

    zspec_plot = []
    zgrism_plot = []
    zphot_plot = []

    for i in range(len(spec_cat)):

        # read in data
        current_id = spec_cat['id'][i]
        current_field = spec_cat['field'][i]
        current_redshift = spec_cat['zspec'][i]

        if current_field == 'goodsn':
            current_field = 'GOODS-N'
        elif current_field == 'goodss':
            current_field = 'GOODS-S'

        lam_em, flam_em, ferr_em, specname, pa_chosen, netsig_chosen = gd.fileprep(current_id, current_redshift, current_field)

        # you only want data in the observed frame
        flam_obs = flam_em / (1 + current_redshift)
        ferr_obs = ferr_em / (1 + current_redshift)
        lam_obs = lam_em * (1 + current_redshift)
    
        # D4000 check
        # Netsig check
        #if (netsig_chosen < 10) or (netsig_chosen > 100):
        if netsig_chosen < 100:
            #print "Skipping", current_id, "in", current_field, "due to low NetSig:", netsig_chosen
            continue

        # Overall error check
        if np.sum(abs(ferr_obs)) > 0.2 * np.sum(abs(flam_obs)):
            #print "Skipping", current_id, "in", current_field, "because of overall error."
            continue

        zspec_plot.append(spec_cat['zspec'][i])
        zgrism_plot.append(spec_cat['zgrism'][i])
        zphot_plot.append(spec_cat['zphot'][i])

        print "Selected", current_id, "in", current_field, "for comparison.",
        print "This object has NetSig:", netsig_chosen

    zspec_plot = np.asarray(zspec_plot)
    zgrism_plot = np.asarray(zgrism_plot)
    zphot_plot = np.asarray(zphot_plot)

    return zspec_plot, zgrism_plot, zphot_plot

if __name__ == '__main__':
    
    #spec_res_cat = np.genfromtxt(massive_figures_dir + "new_specz_sample_fits/specz_results/specz_sample_results.txt", \
    #    dtype=None, names=True, skip_header=15)

    #print "Total", len(spec_res_cat), "galaxies that are in PEARS and have a ground-based redshift",
    #print "and the ground-based redshift is also within 0.6 < zspec <= 1.235"
    #zspec_plot, zgrism_plot, zphot_plot = constrain(spec_res_cat)

    use_new = True
    if use_new:
        # Read in arrays
        id_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/from_run_with_linemask_and_defaultlsf_all_netsigGTR30_speczmatches/id_list.npy')
        field_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/from_run_with_linemask_and_defaultlsf_all_netsigGTR30_speczmatches/field_list.npy')
        zgrism_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/from_run_with_linemask_and_defaultlsf_all_netsigGTR30_speczmatches/zgrism_list.npy')
        zspec_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/from_run_with_linemask_and_defaultlsf_all_netsigGTR30_speczmatches/zphot_list.npy')
        zphot_arr = np.load(massive_figures_dir + 'new_specz_sample_fits/from_run_with_linemask_and_defaultlsf_all_netsigGTR30_speczmatches/zspec_list.npy')
        #chi2_arr = np.load(massive_figures_dir + 'chi2_list.npy')
        #netsig_arr = np.load(massive_figures_dir + 'netsig_list.npy')

        # Place some more cuts
        valid_idx = np.where((zgrism_arr >= 0.6) & (zgrism_arr <= 1.235))[0]
        #valid_idx2 = np.where(chi2_arr <= 2.0)[0]
        #valid_idx = np.concatenate((valid_idx1, valid_idx2))

        id_plot = id_arr[valid_idx]
        field_plot = field_arr[valid_idx]
        zgrism_plot = zgrism_arr[valid_idx]
        zspec_plot = zspec_arr[valid_idx]
        zphot_plot = zphot_arr[valid_idx]
        #chi2_plot = chi2_arr[valid_idx]
        #netsig_plot = netsig_arr[valid_idx]

    print len(zgrism_plot), "galaxies in plot."
    #print "Only", len(zspec_plot), "galaxies within the", len(spec_res_cat), "pass the D4000, NetSig, and overall error constraints."

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # define colors
    myblue = mh.rgb_to_hex(0, 100, 180)
    myred = mh.rgb_to_hex(214, 39, 40)  # tableau 20 red

    grism_resid_hist_arr = (zspec_plot - zgrism_plot)/(1+zspec_plot)
    photz_resid_hist_arr = (zspec_plot - zphot_plot)/(1+zspec_plot)

    large_diff_idx = np.where(abs(grism_resid_hist_arr) > 0.05)[0]
    print len(large_diff_idx)
    print id_plot[large_diff_idx]
    print field_plot[large_diff_idx]
    print zgrism_plot[large_diff_idx]
    print zspec_plot[large_diff_idx]
    print zphot_plot[large_diff_idx]

    fullrange = True
    if fullrange:
        # If you don't want to restrict the range
        ax.hist(photz_resid_hist_arr, 50, histtype='step', color=myred, zorder=10)
        ax.hist(grism_resid_hist_arr, 50, histtype='step', color=myblue, zorder=10)
    else:
        ax.hist(photz_resid_hist_arr, 50, range=[-0.15,0.15], histtype='step', color=myred, zorder=10)
        ax.hist(grism_resid_hist_arr, 50, range=[-0.15,0.15], histtype='step', color=myblue, zorder=10)

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
            'new_specz_sample_fits/from_run_with_linemask_and_defaultlsf_all_netsigGTR30_speczmatches/residual_histogram_netsig_30_fullrange.png', \
            dpi=300, bbox_inches='tight')
    else:
        fig.savefig(massive_figures_dir + \
            'new_specz_sample_fits/from_run_with_linemask_and_defaultlsf_all_netsigGTR30_speczmatches/residual_histogram_netsig_30.png', \
            dpi=300, bbox_inches='tight')

    plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    sys.exit(0)