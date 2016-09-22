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

    massive_galaxies_props = np.genfromtxt(massive_galaxies_dir + 'pears_massive_galaxy_props.txt', dtype=None, names=True)
    stacked_galaxies_props = np.genfromtxt(stacking_analysis_dir + 'stack_props.txt', dtype=None, names=True)

    stellarmass_stacks = 10**stacked_galaxies_props['logmstar']
    stellarmass_massive = 10**massive_galaxies_props['logmstar']

    avg_sf_time_stacks = 10**stacked_galaxies_props['formtime'] - 10**stacked_galaxies_props['mass_wht_age']
    avg_sf_time_massive = 10**massive_galaxies_props['formtime'] - 10**massive_galaxies_props['mass_wht_age']

    avg_sf_time_stacked_err = np.log(10) * (10**stacked_galaxies_props['formtime'] * stacked_galaxies_props['formtime_err'] - 10**stacked_galaxies_props['mass_wht_age'] * stacked_galaxies_props['mass_wht_age_err'])
    avg_sf_time_massive_err = np.log(10) * (10**massive_galaxies_props['formtime'] * massive_galaxies_props['formtime_err'] - 10**massive_galaxies_props['mass_wht_age'] * massive_galaxies_props['mass_wht_age_err'])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{log(M_*/M_\odot)}$')
    ax.set_ylabel(r'$\left< t_{SF} \right>$')

    for k in range(len(stellarmass_massive)):
        if massive_galaxies_props['libname'][k] == 'bc03':
            ax.errorbar(stellarmass_massive[k], avg_sf_time_massive[k], yerr=avg_sf_time_massive_err[k], fmt='x', markersize=5, color='r', markeredgecolor='r')
            ax.plot(stellarmass_massive[k], avg_sf_time_massive[k], 'x', markersize=5, color='r', markeredgecolor='r')
        elif massive_galaxies_props['libname'][k] == 'fsps':
            ax.errorbar(stellarmass_massive[k], avg_sf_time_massive[k], yerr=avg_sf_time_massive_err[k], fmt='x', markersize=5, color='g', markeredgecolor='g')
            ax.plot(stellarmass_massive[k], avg_sf_time_massive[k], 'x', markersize=5, color='g', markeredgecolor='g')

    for k in range(len(stellarmass_stacks)):
        if stacked_galaxies_props['libname'][k] == 'bc03':
            ax.errorbar(stellarmass_stacks[k], avg_sf_time_stacks[k], yerr=avg_sf_time_stacked_err[k], fmt='o', markersize=5, color='r', markeredgecolor='r')
            ax.plot(stellarmass_stacks[k], avg_sf_time_stacks[k], 'o', markersize=5, color='r', markeredgecolor='r')
        elif stacked_galaxies_props['libname'][k] == 'fsps':
            ax.errorbar(stellarmass_stacks[k], avg_sf_time_stacks[k], yerr=avg_sf_time_stacked_err[k], fmt='o', markersize=5, color='g', markeredgecolor='g')
            ax.plot(stellarmass_stacks[k], avg_sf_time_stacks[k], 'o', markersize=5, color='g', markeredgecolor='g')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(1e7,1e12)
    ax.set_ylim(1e7,1e10)

    fig.savefig(massive_figures_dir + 'avg_sf_mstar.eps', dpi=300)

