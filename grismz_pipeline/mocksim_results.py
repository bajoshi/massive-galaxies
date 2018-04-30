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

if __name__ == '__main__':

    # Read in results arrays
    d4000_in = np.load(massive_figures_dir + 'model_mockspectra_fits/d4000_in_list_1p2to1p3.npy')
    d4000_out = np.load(massive_figures_dir + 'model_mockspectra_fits/d4000_out_list_1p2to1p3.npy')
    d4000_out_err = np.load(massive_figures_dir + 'model_mockspectra_fits/d4000_out_err_list_1p2to1p3.npy')
    mock_model_index = np.load(massive_figures_dir + 'model_mockspectra_fits/mock_model_index_list_1p2to1p3.npy')
    test_redshift = np.load(massive_figures_dir + 'model_mockspectra_fits/test_redshift_list_1p2to1p3.npy')
    mock_zgrism = np.load(massive_figures_dir + 'model_mockspectra_fits/mock_zgrism_list_1p2to1p3.npy')
    mock_zgrism_lowerr = np.load(massive_figures_dir + 'model_mockspectra_fits/mock_zgrism_lowerr_list_1p2to1p3.npy')
    mock_zgrism_higherr = np.load(massive_figures_dir + 'model_mockspectra_fits/mock_zgrism_higherr_list_1p2to1p3.npy')

    # --------- redshift accuracy comparison ---------- # 
    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[:8,:])
    ax2 = fig.add_subplot(gs[8:,:])

    # ------- z_mock vs test_redshift ------- #
    ax1.plot(test_redshift, mock_zgrism, 'o', markersize=5.0, color='k', markeredgecolor='k', zorder=10)
    ax1.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r', linewidth=2.0)

    ax1.set_xlim(0.6, 1.24)
    ax1.set_ylim(0.6, 1.24)

    ax1.set_ylabel(r'$\mathrm{z_{mock}}$', fontsize=12, labelpad=-2)

    ax1.xaxis.set_ticklabels([])

    # ------- residuals ------- #
    ax2.plot(test_redshift, (test_redshift - mock_zgrism)/(1+test_redshift), 'o', markersize=5.0, color='k', markeredgecolor='k', zorder=10)
    ax2.axhline(y=0, linestyle='--', color='r')

    ax2.set_xlim(0.6, 1.24)

    ax2.set_xticklabels([])

    ax2.set_xlabel(r'$\mathrm{Test\ redshift}$', fontsize=12)
    ax2.set_ylabel(r'$\mathrm{Residuals}$', fontsize=12, labelpad=-1)

    fig.savefig(massive_figures_dir + 'model_mockspectra_fits/mock_redshift_comparison_d4000_1p2to1p3.eps', dpi=300, bbox_inches='tight')

    sys.exit(0)