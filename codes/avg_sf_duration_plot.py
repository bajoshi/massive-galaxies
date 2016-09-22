from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

def get_threed_mass_wht_ages(ages, logtau):

    timestep = 1e5
    mass_wht_ages = np.zeros(len(ages))

    for j in range(len(ages)):
        formtime = 10**ages[j]
        timearr = np.arange(timestep, formtime, timestep) # in years
        tau = 10**logtau[j] # in years
        n_arr = np.log10(formtime - timearr) * np.exp(-timearr/tau) * timestep
        d_arr = np.exp(-timearr/tau) * timestep
        
        n = np.sum(n_arr)
        d = np.sum(d_arr)
        mass_wht_ages[j] = n / d
    
    return mass_wht_ages

def threedhst_plots():

    threed_n_cat = fits.open('/Users/bhavinjoshi/Desktop/FIGS/new_codes/goodsn_3dhst.v4.1.cats/Fast/goodsn_3dhst.v4.1.fout.FITS')
    threed_s_cat = fits.open('/Users/bhavinjoshi/Desktop/FIGS/new_codes/goodss_3dhst.v4.1.cats/Fast/goodss_3dhst.v4.1.fout.FITS')

    # tau vs mstar
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{log(M_*/M_\odot)}$')
    ax.set_ylabel(r'$\mathrm{log(\tau)}$')

    ax.plot(threed_n_cat[1].data['lmass'], threed_n_cat[1].data['ltau'], 'o', markersize=1, color='k')
    ax.plot(threed_s_cat[1].data['lmass'], threed_s_cat[1].data['ltau'], 'o', markersize=1, color='k')

    ax.set_xlim(6,12)
    ax.set_ylim(6,10)

    fig.savefig(massive_figures_dir + 'threed_tau_mstar.eps', dpi=300)

    # mass weighted ages vs mstar
    mass_wht_ages_n = get_threed_mass_wht_ages(threed_n_cat[1].data['lage'], threed_n_cat[1].data['ltau'])
    mass_wht_ages_s = get_threed_mass_wht_ages(threed_s_cat[1].data['lage'], threed_s_cat[1].data['ltau'])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{log(M_*/M_\odot)}$')
    ax.set_ylabel(r'$\mathrm{\left< t_M \right>}$')

    ax.plot(threed_n_cat[1].data['lmass'], mass_wht_ages_n, 'o', markersize=1, color='k')
    ax.plot(threed_s_cat[1].data['lmass'], mass_wht_ages_s, 'o', markersize=1, color='k')

    ax.set_xlim(6,12)
    #ax.set_ylim(7,11)

    fig.savefig(massive_figures_dir + 'threed_mass_wht_ages_mstar.eps', dpi=300)

    return None

if __name__ == '__main__':

    threedhst_plots()
    sys.exit(0)

    massive_galaxies_props = np.genfromtxt(massive_galaxies_dir + 'pears_massive_galaxy_props.txt', dtype=None, names=True)
    stacked_galaxies_props = np.genfromtxt(stacking_analysis_dir + 'stack_props.txt', dtype=None, names=True)

    stellarmass_stacks = 10**stacked_galaxies_props['logmstar']
    stellarmass_massive = 10**massive_galaxies_props['logmstar']

    avg_sf_time_stacks = 10**stacked_galaxies_props['formtime'] - 10**stacked_galaxies_props['mass_wht_age']
    avg_sf_time_massive = 10**massive_galaxies_props['formtime'] - 10**massive_galaxies_props['mass_wht_age']
    avg_sf_time_stacked_err = np.log(10) * (10**stacked_galaxies_props['formtime'] * stacked_galaxies_props['formtime_err'] - 10**stacked_galaxies_props['mass_wht_age'] * stacked_galaxies_props['mass_wht_age_err'])
    avg_sf_time_massive_err = np.log(10) * (10**massive_galaxies_props['formtime'] * massive_galaxies_props['formtime_err'] - 10**massive_galaxies_props['mass_wht_age'] * massive_galaxies_props['mass_wht_age_err'])

    tau_stacks = np.log10(stacked_galaxies_props['tau'] * 1e9)
    tau_massive = np.log10(massive_galaxies_props['tau'] * 1e9)

    bc03_indices_stacks = np.where(stacked_galaxies_props['libname'] == 'bc03')[0]
    fsps_indices_stacks = np.where(stacked_galaxies_props['libname'] == 'fsps')[0]

    bc03_indices_massive = np.where(massive_galaxies_props['libname'] == 'bc03')[0]
    fsps_indices_massive = np.where(massive_galaxies_props['libname'] == 'fsps')[0]

    ## avg sf time vs mstar
    #fig = plt.figure()
    #ax = fig.add_subplot(111)

    #ax.set_xlabel(r'$\mathrm{log(M_*/M_\odot)}$')
    #ax.set_ylabel(r'$\left< t_{SF} \right>$')

    #for k in range(len(stellarmass_massive)):
    #    if massive_galaxies_props['libname'][k] == 'bc03':
    #        ax.errorbar(stellarmass_massive[k], avg_sf_time_massive[k], yerr=avg_sf_time_massive_err[k], fmt='x', markersize=5, color='r', markeredgecolor='r')
    #        ax.plot(stellarmass_massive[k], avg_sf_time_massive[k], 'x', markersize=5, color='r', markeredgecolor='r')
    #    elif massive_galaxies_props['libname'][k] == 'fsps':
    #        ax.errorbar(stellarmass_massive[k], avg_sf_time_massive[k], yerr=avg_sf_time_massive_err[k], fmt='x', markersize=5, color='g', markeredgecolor='g')
    #        ax.plot(stellarmass_massive[k], avg_sf_time_massive[k], 'x', markersize=5, color='g', markeredgecolor='g')

    #for k in range(len(stellarmass_stacks)):
    #    if stacked_galaxies_props['libname'][k] == 'bc03':
    #        ax.errorbar(stellarmass_stacks[k], avg_sf_time_stacks[k], yerr=avg_sf_time_stacked_err[k], fmt='o', markersize=5, color='r', markeredgecolor='r')
    #        ax.plot(stellarmass_stacks[k], avg_sf_time_stacks[k], 'o', markersize=5, color='r', markeredgecolor='r')
    #    elif stacked_galaxies_props['libname'][k] == 'fsps':
    #        ax.errorbar(stellarmass_stacks[k], avg_sf_time_stacks[k], yerr=avg_sf_time_stacked_err[k], fmt='o', markersize=5, color='g', markeredgecolor='g')
    #        ax.plot(stellarmass_stacks[k], avg_sf_time_stacks[k], 'o', markersize=5, color='g', markeredgecolor='g')

    #ax.set_xscale('log')
    #ax.set_yscale('log')

    #ax.set_xlim(1e7,1e12)
    #ax.set_ylim(1e7,1e10)

    #fig.savefig(massive_figures_dir + 'avg_sf_mstar.eps', dpi=300)

    # tau vs mstar
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{log(M_*/M_\odot)}$')
    ax.set_ylabel(r'$\mathrm{log(\tau)}$')

    ax.plot(stacked_galaxies_props['logmstar'][bc03_indices_stacks], tau_stacks[bc03_indices_stacks], 'o', color='k')
    ax.plot(massive_galaxies_props['logmstar'][bc03_indices_massive], tau_massive[bc03_indices_massive], 'x', color='k')

    ax.set_xlim(7,12)
    ax.set_ylim(6,11)

    fig.savefig(massive_figures_dir + 'tau_mstar.eps', dpi=300)

    # mass weighted ages vs mstar
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{log(M_*/M_\odot)}$')
    ax.set_ylabel(r'$\mathrm{\left< t_M \right>}$')

    ax.plot(stacked_galaxies_props['logmstar'][bc03_indices_stacks], stacked_galaxies_props['mass_wht_age'][bc03_indices_stacks], 'o', color='k')
    ax.plot(massive_galaxies_props['logmstar'][bc03_indices_massive], massive_galaxies_props['mass_wht_age'][bc03_indices_massive], 'x', color='k')

    ax.set_xlim(7,12)
    ax.set_ylim(7,10)

    fig.savefig(massive_figures_dir + 'mass_wht_age_mstar.eps', dpi=300)

