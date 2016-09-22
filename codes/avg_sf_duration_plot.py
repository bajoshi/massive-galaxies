from __future__ import division

import numpy as np

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

if __name__ == '__main__':

    massive_galaxies_props = np.genfromtxt(massive_galaxies_dir + 'pears_massive_galaxy_specs.txt', dtype=None, names=True)
    stacked_galaxies_props = np.genfromtxt(stacking_analysis_dir + 'stack_props.txt', dtype=None, names=True)

    stellarmass_stacks = 10**stacked_galaxies_props['logmstar']
    stellarmass_massive = 10**massive_galaxies_props['logmstar']

    avg_sf_time_stacks = 10**stacked_galaxies_props['formtime'] - 10**stacked_galaxies_props['mass_wht_age']
    avg_sf_time_massive = 10**massive_galaxies_props['formtime'] - 10**massive_galaxies_props['mass_wht_age']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(stellarmass_massive, avg_sf_time_massive, 'x', markersize=5, color='k', markeredgecolor='k')
    ax.plot(stellarmass_stacks, avg_sf_time_stacks, 'o', markersize=5, color='k', markeredgecolor='k')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(1e9,1e12)
    ax.set_ylim(1e7,1e10)

    plt.show()