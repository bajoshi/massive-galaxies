from __future__ import division

import numpy as np
from astropy.stats import mad_std

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"

def get_plotting_arrays():
    # Read in SPZ results 
    spz_results_dir = massive_figures_dir + 'spz_run_jan2019/'

    spz_id_list = np.load(spz_results_dir + 'spz_id_list.npy')
    spz_field_list = np.load(spz_results_dir + 'spz_field_list.npy')
    zminchi2 = np.load(spz_results_dir + 'spz_zgrism_list.npy')
    zspec = np.load(spz_results_dir + 'spz_zspec_list.npy')
    spz_d4000 = np.load(spz_results_dir + 'spz_d4000_list.npy')

    # Read in results from earlier photoz run
    zphot = np.load(massive_figures_dir + 'my_photoz_list.npy')
    # Also need ID and field from photoz run to match
    photoz_id_list = np.load(massive_figures_dir + 'my_photoz_id_list.npy')
    photoz_field_list = np.load(massive_figures_dir + 'my_photoz_field_list.npy')

    # Now match SPZ run and photoz run and make arrays for plotting
    zspec_plot = []
    zphot_plot = []
    zspz_plot = []
    d4000_plot = []
    for i in range(len(zphot)):

        # Get photoz id and field and match to spz run
        photoz_id = photoz_id_list[i]
        photoz_field = photoz_field_list[i]

        idx = np.where((spz_id_list == photoz_id) & (spz_field_list == photoz_field))[0]

        # If a match is found then first compute weighted 
        # spz and then append to plotting arrays
        if idx.size:
            # It should only find one exact match
            idx = int(idx)

            pz = np.load(spz_results_dir + str(photoz_field) + '_' + str(photoz_id) + '_pz.npy')
            zarr = np.load(spz_results_dir + str(photoz_field) + '_' + str(photoz_id) + '_z_arr.npy')
            zspz = np.sum(pz * zarr)

            zspec_plot.append(zspec[idx])
            zphot_plot.append(zphot[i])
            zspz_plot.append(zspz)
            d4000_plot.append(spz_d4000[idx])

    # Convert to numpy arrays and return
    zspec_plot = np.asarray(zspec_plot)
    zphot_plot = np.asarray(zphot_plot)
    zspz_plot = np.asarray(zspz_plot)
    d4000_plot = np.asarray(d4000_plot)

    return zspec_plot, zphot_plot, zspz_plot, d4000_plot

def main():

    zspec, zphot, zspz, d4000 = get_plotting_arrays()

    # Apply D4000 cut
    d4000_idx = np.where((d4000 >= 1.4) & (d4000 < 2.5))[0]

    zspec = zspec[d4000_idx]
    zphot = zphot[d4000_idx]
    zspz = zspz[d4000_idx]
    d4000 = d4000[d4000_idx]

    # Make figure
    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(10,24)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.5, hspace=0)

    ax1 = fig.add_subplot(gs[:7, :11])
    ax2 = fig.add_subplot(gs[7:, :11])

    ax3 = fig.add_subplot(gs[:7, 13:])
    ax4 = fig.add_subplot(gs[7:, 13:])

    # Get residuals 
    resid_zphot = (zspec - zphot) / (1 + zspec)
    resid_zspz = (zspec - zspz) / (1 + zspec)

    # Make sure they are finite
    valid_idx1 = np.where(np.isfinite(resid_zphot))[0]
    valid_idx2 = np.where(np.isfinite(resid_zspz))[0]

    valid_idx = reduce(np.intersect1d, (valid_idx1, valid_idx2))

    resid_zphot = resid_zphot[valid_idx]
    resid_zspz = resid_zspz[valid_idx]
    zspec = zspec[valid_idx]
    zphot = zphot[valid_idx]
    zspz = zspz[valid_idx]

    # Remove catastrophic failures
    # Also print out catastrophic failure rate

    # Print info
    mean_zphot = np.mean(resid_zphot)
    std_zphot = np.std(resid_zphot)
    nmad_zphot = mad_std(resid_zphot)

    mean_zspz = np.mean(resid_zspz)
    std_zspz = np.std(resid_zspz)
    nmad_zspz = mad_std(resid_zspz)

    print "Mean, std dev, and Sigma_NMAD for residuals for Photo-z:", \
    "{:.2e}".format(mean_zphot), "{:.2e}".format(std_zphot), "{:.2e}".format(nmad_zphot)
    print "Mean, std dev, and Sigma_NMAD for residuals for SPZs:", \
    "{:.2e}".format(mean_zspz), "{:.2e}".format(std_zspz), "{:.2e}".format(nmad_zspz)

    ax1.plot(zspec, zphot, 'o', markersize=2, color='k', markeredgecolor='k')
    ax2.plot(zspec, resid_zphot, 'o', markersize=2, color='k', markeredgecolor='k')

    ax3.plot(zspec, zspz, 'o', markersize=2, color='k', markeredgecolor='k')
    ax4.plot(zspec, resid_zspz, 'o', markersize=2, color='k', markeredgecolor='k')

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit()