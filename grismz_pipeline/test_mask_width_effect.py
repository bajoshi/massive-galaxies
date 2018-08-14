from __future__ import division

import numpy as np

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
large_diff_dir = massive_figures_dir + "large_diff_specz_sample/"

if __name__ == '__main__':

    # m1 arrays
    m1_zphot = np.load(large_diff_dir + 'zphot_list_m1.npy')
    m1_zspec = np.load(large_diff_dir + 'zspec_list_m1.npy')
    m1_zgrism = np.load(large_diff_dir + 'zgrism_list_m1.npy')

    # m3 array
    m3_zgrism = np.load(large_diff_dir + 'zgrism_list_m3.npy')

    # m5 array
    m5_zgrism = np.load(large_diff_dir + 'zgrism_list_m5.npy')

    # z_grism vs z_phot vs z_spec plot
    gs = gridspec.GridSpec(20,10)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.0, hspace=0.0)

    fig = plt.figure()

    ax1 = fig.add_subplot(gs[:15,:])
    ax2 = fig.add_subplot(gs[15:,:])

    # plot one redshift vs another
    ax1.plot(m1_zspec, m1_zphot, 'o', markersize=3.0, color='r', markeredgecolor='r', zorder=10)
    # zphot and zspec dont change

    ax1.plot(m1_zspec, m1_zgrism, 'o', markersize=3.0, color='blue', markeredgecolor='blue', zorder=10)
    ax1.plot(m1_zspec, m5_zgrism, 'o', markersize=3.0, color='green', markeredgecolor='green', zorder=10)

    ax1.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='k', linewidth=1.0)

    # plot residuals
    ax2.plot(m1_zspec, (m1_zspec - m1_zphot)/(1+m1_zspec), 'o', markersize=3.0, color='r', markeredgecolor='r', zorder=10)

    ax2.plot(m1_zspec, (m1_zspec - m1_zgrism)/(1+m1_zspec), 'o', markersize=3.0, color='blue', markeredgecolor='blue', zorder=10)
    ax2.plot(m1_zspec, (m1_zspec - m5_zgrism)/(1+m1_zspec), 'o', markersize=3.0, color='green', markeredgecolor='green', zorder=10)

    ax2.axhline(y=0, linestyle='--', color='k')

    ax1.set_xlim(0.6,1.24)
    ax1.set_ylim(0.6,1.24)
    ax2.set_xlim(0.6,1.24)

    plt.show()

    sys.exit(0)