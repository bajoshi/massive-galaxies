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

    z_spec_plot = []
    z_grism_plot = []
    z_phot_plot = []

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

        z_spec_plot.append(spec_cat['zspec'][i])
        z_grism_plot.append(spec_cat['zgrism'][i])
        z_phot_plot.append(spec_cat['zphot'][i])

        print "Selected", current_id, "in", current_field, "for comparison.",
        print "This object has NetSig:", netsig_chosen

    z_spec_plot = np.asarray(z_spec_plot)
    z_grism_plot = np.asarray(z_grism_plot)
    z_phot_plot = np.asarray(z_phot_plot)

    return z_spec_plot, z_grism_plot, z_phot_plot

if __name__ == '__main__':
    
    spec_res_cat = np.genfromtxt(massive_figures_dir + "new_specz_sample_fits/specz_results/specz_sample_results.txt", \
        dtype=None, names=True, skip_header=15)

    print "Total", len(spec_res_cat), "galaxies that are in PEARS and have a ground-based redshift",
    print "and the ground-based redshift is also within 0.6 < zspec <= 1.235"
    z_spec_plot, z_grism_plot, z_phot_plot = constrain(spec_res_cat)

    use_new = False
    if use_new:
        z_grism_plot = np.array([  0.7223, -99.0, 0.7665, 0.9298, 0.8756,   0.6002,\
        0.8461,   0.9377,   0.976,    0.992,    0.601,    0.723,    0.601,    0.602,\
        0.902,    1.032,   0.687,    0.636,    0.76,     0.735,    0.95,     0.976,\
        0.666,    0.601,   0.964,    1.063,    0.854,    0.943,    1.08,     0.6   ])

        z_spec_plot = np.array([0.7603, 0.0846, 0.9805, 0.9698, 0.8516, 0.6382, \
            0.8181, 0.8197, 0.682, 1.012, 0.637, 0.641, 0.745, 0.58, 0.858, \
            1.122, 0.623, 0.618, 0.634, 0.713, 0.976, 0.97, 0.638, 0.659, 0.91, \
            1.015, 0.658, 0.923, 1.38, 0.794])

        z_phot_plot = np.array([0.761, 0.682, 0.851, 0.942, 0.857, 0.634, 0.841, \
            0.836, 0.734, 0.998, 0.605, 0.726, 0.738, 0.615, 0.831, 1.129, 0.668, \
            0.668, 0.67, 0.735, 1.036, 0.955, 0.67, 0.989, 0.954, 1.015, 0.667, \
            0.978, 1.22, 0.783])

        valid_idx = np.where((z_grism_plot >= 0.6) & (z_grism_plot <= 1.235))[0]
        z_grism_plot = z_grism_plot[valid_idx]
        z_spec_plot = z_spec_plot[valid_idx]
        z_phot_plot = z_phot_plot[valid_idx]

    print "Only", len(z_spec_plot), "galaxies within the", len(spec_res_cat), "pass the D4000, NetSig, and overall error constraints."

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # define colors
    myblue = mh.rgb_to_hex(0, 100, 180)
    myred = mh.rgb_to_hex(214, 39, 40)  # tableau 20 red

    grism_resid_hist_arr = (z_spec_plot - z_grism_plot)/(1+z_spec_plot)
    photz_resid_hist_arr = (z_spec_plot - z_phot_plot)/(1+z_spec_plot)

    ax.hist(photz_resid_hist_arr, 15, range=[-0.15,0.22], histtype='step', color=myred, alpha=0.75, zorder=10)
    ax.hist(grism_resid_hist_arr, 15, range=[-0.15,0.22], histtype='step', color=myblue, alpha=0.6, zorder=10)
    
    # If you don't want to restrict the range
    #ax.hist(photz_resid_hist_arr, 15, histtype='step', color=myred, alpha=0.75, zorder=10)
    #ax.hist(grism_resid_hist_arr, 15, histtype='step', color=myblue, alpha=0.6, zorder=10)

    # this plot really needs an alpha channel
    # otherwise you wont see that the photo-z histogram under the grism-z histogram
    # is actually fatter around 0 whereas the grism-z histogram is thinner.

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

    fig.savefig(massive_figures_dir + 'new_specz_sample_fits/residual_histogram_netsig_100.png', dpi=300, bbox_inches='tight')
    # has to be png NOT eps. This plot needs an alpha channel.

    plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    sys.exit(0)