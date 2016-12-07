from __future__ import division

import numpy as np

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, AnchoredText

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"

if __name__ == '__main__':
    
    # read catalog adn rename arrays for convenience
    pears_refined_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/pears_refined_4000break_catalog.txt', dtype=None, names=True, skip_header=1)

    z_spec = pears_refined_cat['new_z']
    z_phot = pears_refined_cat['old_z']

    old_chi2 = np.log10(pears_refined_cat['old_chi2'] / 88)
    new_chi2 = np.log10(pears_refined_cat['new_chi2'] / 88)

    # make plots
    # z_spec vs z_phot
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=2.0)

    fig = plt.figure()
    ax1 = fig.add_subplot(gs[:10,:])
    ax2 = fig.add_subplot(gs[10:,:])

    # first subplot
    ax1.plot(z_phot, z_spec, 'o', markersize=1.5, color='k', markeredgecolor='k')
    ax1.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r')

    ax1.set_xlim(0.2, 1.42)
    ax1.set_ylim(0.2, 1.42)

    ax1.set_ylabel(r'$z_\mathrm{phot}$', fontsize=15)

    ax1.xaxis.set_ticklabels([])

    ax1.minorticks_on()
    ax1.tick_params('both', width=1, length=3, which='minor')
    ax1.tick_params('both', width=1, length=4.7, which='major')
    ax1.grid(True)

    # second subplot
    ax2.plot(z_spec, (z_spec - z_phot)/(1+z_spec), 'o', markersize=1.5, color='k', markeredgecolor='k')
    ax2.axhline(y=0, linestyle='--', color='r')

    ax2.set_xlim(0.2, 1.42)
    ax2.set_ylim(-0.5, 0.5)

    ax2.set_xlabel(r'$z_\mathrm{spec}$', fontsize=15)
    ax2.set_ylabel(r'$(z_\mathrm{spec} - z_\mathrm{phot})/(1+z_\mathrm{spec})$', fontsize=15)

    ax2.minorticks_on()
    ax2.tick_params('both', width=1, length=3, which='minor')
    ax2.tick_params('both', width=1, length=4.7, which='major')
    ax2.grid(True)

    fig.savefig(massive_figures_dir + "refined_zspec_vs_zphot.eps", dpi=300, bbox_inches='tight')

    del fig, ax1, ax2

    # histograms of chi2
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get total bins and plot histogram
    old_chi2_indx = np.where(old_chi2 < 20)
    new_chi2_indx = np.where(new_chi2 < 20)

    old_chi2 = old_chi2[old_chi2_indx]
    new_chi2 = new_chi2[new_chi2_indx]

    iqr = np.std(old_chi2, dtype=np.float64)
    binsize = 2*iqr*np.power(len(old_chi2),-1/3)
    totalbins = np.floor((max(old_chi2) - min(old_chi2))/binsize)

    ax.hist(old_chi2, totalbins, facecolor='None', align='mid', linewidth=1, edgecolor='b', histtype='step')

    iqr = np.std(new_chi2, dtype=np.float64)
    binsize = 2*iqr*np.power(len(new_chi2),-1/3)
    totalbins = np.floor((max(new_chi2) - min(new_chi2))/binsize)

    ax.hist(new_chi2, totalbins, facecolor='None', align='mid', linewidth=1, edgecolor='r', histtype='step')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    ax.set_xlabel(r'$\mathrm{log(\chi^2)}$')
    ax.set_ylabel(r'$\mathrm{N}$')

    fig.savefig(massive_figures_dir + "refined_old_new_chi2_hist.eps", dpi=300, bbox_inches='tight')
    
    sys.exit(0)






