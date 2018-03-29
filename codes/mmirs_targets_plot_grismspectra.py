from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

if __name__ == '__main__':

    """
    Only doing this for the two galaxies that are selected for MMIRS 
    and they also have data from PEARS, FIGS, and 3DHST.
    """

    # Alll in GOODS-N
    # Define ids
    figs_id1 = 300665
    figs_id2 = 300648
    figs_id3 = 400772
    figs_id4 = 400456

    threed_id1 = 34028
    threed_id2 = 33780
    threed_id3 = 33181
    threed_id4 = 35405

    # read spectra
    threed_data_dir_gn16 = home + '/Desktop/3dhst_data/goodsn-16/1D/FITS/'
    threed_data_dir_gn27 = home + '/Desktop/3dhst_data/goodsn-27/1D/FITS/'
    threed_data_dir_gn28 = home + '/Desktop/3dhst_data/goodsn-28/1D/FITS/'

    spec_hdu1 = fits.open(threed_data_dir_gn16 + 'goodsn-16-G141_' + str(threed_id1) + '.1D.fits')
    spec_hdu2 = fits.open(threed_data_dir_gn16 + 'goodsn-16-G141_' + str(threed_id2) + '.1D.fits')
    spec_hdu3 = fits.open(threed_data_dir_gn27 + 'goodsn-27-G141_' + str(threed_id3) + '.1D.fits')
    spec_hdu4 = fits.open(threed_data_dir_gn28 + 'goodsn-28-G141_' + str(threed_id4) + '.1D.fits')

    # plot
    fig = plt.figure(figsize=(6,3.5))
    
    # create grid to plot
    gs = gridspec.GridSpec(2,2)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.01)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    # plot
    ax1.plot(spec_hdu1[1].data['wave'], (spec_hdu1[1].data['flux'] - spec_hdu1[1].data['contam']) / spec_hdu1[1].data['sensitivity'])
    ax1.axhline(y=0, ls='--', color='k', lw=2)

    ax2.plot(spec_hdu2[1].data['wave'], (spec_hdu2[1].data['flux'] - spec_hdu2[1].data['contam']) / spec_hdu2[1].data['sensitivity'])
    ax2.axhline(y=0, ls='--', color='k', lw=2)

    ax3.plot(spec_hdu3[1].data['wave'], (spec_hdu3[1].data['flux'] - spec_hdu3[1].data['contam']) / spec_hdu3[1].data['sensitivity'])
    ax3.axhline(y=0, ls='--', color='k', lw=2)

    ax4.plot(spec_hdu4[1].data['wave'], (spec_hdu4[1].data['flux'] - spec_hdu4[1].data['contam']) / spec_hdu4[1].data['sensitivity'])
    ax4.axhline(y=0, ls='--', color='k', lw=2)

    # limits
    ax1.set_xlim(10900, 16500)
    ax2.set_xlim(10900, 16500)
    ax3.set_xlim(10900, 16500)
    ax4.set_xlim(10900, 16500)

    ax1.set_ylim(-0.05, 0.1)
    ax2.set_ylim(-0.05, 0.2)
    ax3.set_ylim(-0.05, 0.1)
    ax4.set_ylim(-0.05, 0.1)

    # Write x tick labels in microns
    xticks = ax3.get_xticks().tolist()
    xticks = np.asarray(xticks).astype(np.int) / 1e4

    ax3.set_xticklabels(xticks, size=10, rotation=30)
    ax4.set_xticklabels(xticks, size=10, rotation=30)

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    # y tick labels
    ax1.set_yticklabels(ax1.get_yticks().tolist(), size=8, rotation=15)
    ax2.set_yticklabels(ax2.get_yticks().tolist(), size=8, rotation=15)
    ax3.set_yticklabels(ax3.get_yticks().tolist(), size=8, rotation=15)
    ax4.set_yticklabels(ax4.get_yticks().tolist(), size=8, rotation=15)

    ax1.tick_params(axis='y', pad=-18)
    ax2.tick_params(axis='y', pad=-18)
    ax3.tick_params(axis='y', pad=-18)
    ax4.tick_params(axis='y', pad=-18)

    # labels
    ax3.set_xlabel(r'$\mathrm{Wavelength\ [microns]}$', fontsize=12)
    ax3.set_ylabel(r'$\mathrm{Flux\ [1E-17\ CGS]}$', fontsize=12)

    ax3.xaxis.set_label_coords(1.05, -0.14)
    ax3.yaxis.set_label_coords(-0.03, 1.05)

    # text on plot
    ax1.text(0.65, 0.9, r"$\mathrm{FIGS\ ID\ }$" + str(figs_id1), verticalalignment='top', horizontalalignment='left', \
            transform=ax1.transAxes, color='k', size=8)
    ax1.text(0.65, 0.83, r"$\mathrm{M_s\sim11.21\,M_\odot}$", verticalalignment='top', horizontalalignment='left', \
            transform=ax1.transAxes, color='k', size=8)
    ax1.text(0.65, 0.76, r"$\mathrm{z_{phot}\sim2.68}$", verticalalignment='top', horizontalalignment='left', \
            transform=ax1.transAxes, color='k', size=8)

    ax2.text(0.65, 0.44, r"$\mathrm{FIGS\ ID\ }$" + str(figs_id2), verticalalignment='top', horizontalalignment='left', \
            transform=ax2.transAxes, color='k', size=8)
    ax2.text(0.65, 0.37, r"$\mathrm{M_s\sim11.29\,M_\odot}$", verticalalignment='top', horizontalalignment='left', \
            transform=ax2.transAxes, color='k', size=8)
    ax2.text(0.65, 0.3, r"$\mathrm{z_{phot}\sim1.86}$", verticalalignment='top', horizontalalignment='left', \
            transform=ax2.transAxes, color='k', size=8)

    ax3.text(0.65, 0.9, r"$\mathrm{FIGS\ ID\ }$" + str(figs_id3), verticalalignment='top', horizontalalignment='left', \
            transform=ax3.transAxes, color='k', size=8)
    ax3.text(0.65, 0.83, r"$\mathrm{M_s\sim10.77\,M_\odot}$", verticalalignment='top', horizontalalignment='left', \
            transform=ax3.transAxes, color='k', size=8)
    ax3.text(0.65, 0.76, r"$\mathrm{z_{phot}\sim2.22}$", verticalalignment='top', horizontalalignment='left', \
            transform=ax3.transAxes, color='k', size=8)

    ax4.text(0.65, 0.9, r"$\mathrm{FIGS\ ID\ }$" + str(figs_id4), verticalalignment='top', horizontalalignment='left', \
            transform=ax4.transAxes, color='k', size=8)
    ax4.text(0.65, 0.83, r"$\mathrm{M_s\sim10.66\,M_\odot}$", verticalalignment='top', horizontalalignment='left', \
            transform=ax4.transAxes, color='k', size=8)
    ax4.text(0.65, 0.76, r"$\mathrm{z_{phot}\sim2.08}$", verticalalignment='top', horizontalalignment='left', \
            transform=ax4.transAxes, color='k', size=8)

    fig.savefig(home + '/Desktop/mmirs_targets_example_grismspectra.png', dpi=300, bbox_inches='tight')

    spec_hdu1.close()
    spec_hdu2.close()
    spec_hdu3.close()
    spec_hdu4.close()

    sys.exit(0)