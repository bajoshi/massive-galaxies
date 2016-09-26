from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
print sys.path

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd

if __name__ == '__main__':
    
    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # Read pears + 3dhst catalog
    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt',
                       dtype=None, names=True, skip_header=2)

    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    # Find indices for massive galaxies
    massive_galaxies_indices = np.where(stellarmass >= 10.0)[0]

    """
    # Create grid for making grid plots
    gs = gridspec.GridSpec(9, 18)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

    fig = plt.figure()

    # Loop over all spectra 
    for u in range(len(pears_id[massive_galaxies_indices])):

        redshift = photz[massive_galaxies_indices][u]
        lam_em, flam_em, ferr, specname = gd.fileprep(pears_id[massive_galaxies_indices][u], redshift)

        row = 3 * int(u / 6)
        col = 3 * int(u % 6)
        ax = fig.add_subplot(gs[row:row+3, col:col+3])

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        if (row == 3) and (col == 0):
            ax.set_ylabel(r'$f_{\lambda}\ [\mathrm{erg/s/cm^2/\AA}] $')
        if (row == 6) and (col == 9):
            ax.set_xlabel(r'$\lambda\ [\AA]$')

        ax.plot(lam_em, flam_em, color='k')
        ax.fill_between(lam_em, flam_em + ferr, flam_em - ferr, color='lightgray')

        ax.set_xlim(2500, 6000)

    fig.savefig(massive_figures_dir + 'massive_galaxies_spectra.eps', dpi=150)
    #plt.show()
    """

    for u in range(len(pears_id[massive_galaxies_indices])):

        redshift = photz[massive_galaxies_indices][u]
        lam_em, flam_em, ferr, specname = gd.fileprep(pears_id[massive_galaxies_indices][u], redshift)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel(r'$\lambda\ [\AA]$')
        ax.set_ylabel(r'$f_{\lambda}\ [\mathrm{erg/s/cm^2/\AA}] $')

        ax.plot(lam_em, flam_em, color='k')
        ax.fill_between(lam_em, flam_em + ferr, flam_em - ferr, color='lightgray')

        ax.set_xlim(2500, 6000)

        fig.savefig(savefits_dir + 'massive-galaxy-spectra/' + str(pears_id[massive_galaxies_indices][u]) + '.eps', dpi=150)

    sys.exit(0)

