from __future__ import division

import numpy as np

import sys
import os

import matplotlib.pyplot as plt

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
figs_dir = home + "/Desktop/FIGS/"

if __name__ == '__main__':

    # read netsig arrays
    netsig_n = np.load(massive_figures_dir + 'full_run/netsig_list_gn.npy')
    netsig_s = np.load(massive_figures_dir + 'full_run/netsig_list_gs.npy')

    # concatenate
    netsig = np.concatenate((netsig_n, netsig_s))
    netsig = np.log10(netsig)  # use the log of the netsig

    # Make hist
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{log(NetSig)}$', fontsize=12)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=12)

    # get total bins and plot histogram
    iqr = np.std(netsig, dtype=np.float64)
    binsize = 2*iqr*np.power(len(netsig),-1/3)
    totalbins = np.floor((max(netsig) - min(netsig))/binsize)
    totalbins = int(totalbins)

    ax.hist(netsig, totalbins, histtype='step', color='k', align='mid')

    ax.text(0.8, 0.9, r'$\mathrm{D4000\geq1.2}$', \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=12)

    fig.savefig(massive_figures_dir + 'netsig_hist.eps', dpi=150, bbox_inches='tight')

    sys.exit(0)