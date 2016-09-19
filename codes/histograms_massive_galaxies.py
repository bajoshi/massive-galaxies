from __future__ import division

import numpy as np
from astropy.io import fits

import collections
import sys
import os
import datetime
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd
import fast_chi2_jackknife as fcj

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

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
    massive_galaxies_indices = np.where(stellarmass >= 11.0)[0]

    num_jackknife_samps = 1e4
    # Loop over all spectra 
    for u in range(len(pears_id[massive_galaxies_indices])):

        print "Currently working with PEARS object id: ", pears_id[massive_galaxies_indices][u]

        redshift = photz[massive_galaxies_indices][u]
        lam_em, flam_em, ferr, specname = gd.fileprep(pears_id[massive_galaxies_indices][u], redshift)

        resampling_lam_grid = lam_em 

        # Create figures to be used for plots for all 3 SPS libraries
        #### BC03 and FSPS ####
        #fig_ages = plt.figure()
        #fig_tau = plt.figure()
        #fig_mass_wht_ages = plt.figure()
        #fig_quench = plt.figure()
    
        #### MILES ####
        #fig_metals_miles = plt.figure()

        # Read files with params from jackknife runs
        #### BC03 ####
        ages_bc03 = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_ages_bc03.txt', usecols=range(int(num_jackknife_samps)))
        logtau_bc03 = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_logtau_bc03.txt', usecols=range(int(num_jackknife_samps)))
        tauv_bc03 = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_tauv_bc03.txt', usecols=range(int(num_jackknife_samps)))
        mass_wht_ages_bc03 = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_mass_weighted_ages_bc03.txt', usecols=range(int(num_jackknife_samps)))

        av_bc03 = (2.5 / np.log(10)) * tauv_bc03 * 10
    
        quenching_times_bc03 = 10**ages_bc03 - 10**mass_wht_ages_bc03
        quenching_times_bc03 = np.log10(quenching_times_bc03)
    
        #### MILES ####
        ages_miles = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_ages_miles.txt', usecols=range(int(num_jackknife_samps)))
        metals_miles = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_metals_miles.txt', usecols=range(int(num_jackknife_samps)))
    
        #### FSPS ####
        ages_fsps = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_ages_fsps.txt', usecols=range(int(num_jackknife_samps)))
        logtau_fsps = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_logtau_fsps.txt', usecols=range(int(num_jackknife_samps)))
        mass_wht_ages_fsps = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_mass_weighted_ages_fsps.txt', usecols=range(int(num_jackknife_samps)))
    
        quenching_times_fsps = 10**ages_fsps - 10**mass_wht_ages_fsps
        quenching_times_fsps = np.log10(quenching_times_fsps)

        print '{:1.2e}'.format(10**np.median(mass_wht_ages_bc03)), '{:1.2e}'.format(10**np.median(quenching_times_bc03)), '{:1.2e}'.format(10**np.median(mass_wht_ages_bc03) / 10**np.median(quenching_times_bc03))
        print '{:1.2e}'.format(10**np.median(mass_wht_ages_fsps)), '{:1.2e}'.format(10**np.median(quenching_times_fsps)), '{:1.2e}'.format(10**np.median(mass_wht_ages_fsps) / 10**np.median(quenching_times_fsps))

        """
        # --------------------------- Histogram plots ------------------------------
        ############ BC03 and FSPS ############
        #### Age grid plots ####
        #ax_gs_ages = fig_ages.add_subplot(111)
        #ax_gs_ages.set_title('Formation Times')
        #ax_gs_ages.hist(ages_bc03, len(np.unique(ages_bc03)), histtype='step', align='mid', color='r', linewidth=2)
        #ax_gs_ages.hist(ages_miles, len(np.unique(ages_miles)), histtype='step', align='mid', color='b', linewidth=2)
        #ax_gs_ages.hist(ages_fsps, len(np.unique(ages_fsps)), histtype='step', align='mid', color='g', linewidth=2)

        ##ax_gs_ages.set_xlim(8, 10)
        #ax_gs_ages.set_yscale('log')

        #fig_ages.savefig(massive_figures_dir + 'agedist_' + str(pears_id[massive_galaxies_indices][u]) + '.eps', dpi=300)

        #### Mass weighted age grid plots ####
        ax_gs_mass_wht_ages = fig_mass_wht_ages.add_subplot(111)
        ax_gs_mass_wht_ages.set_title('Mass weighted ages')
        ax_gs_mass_wht_ages.hist(mass_wht_ages_bc03, 6, histtype='step', align='mid', color='r', linewidth=2)
        ax_gs_mass_wht_ages.hist(mass_wht_ages_fsps, 6, histtype='step', align='mid', color='g', linewidth=2)

        ax_gs_mass_wht_ages.set_yscale('log')
        ax_gs_mass_wht_ages.set_xlim(9, 10)
        ax_gs_mass_wht_ages.set_ylim(1, 1e4)

        fig_mass_wht_ages.savefig(massive_figures_dir + 'mass_wht_agedist_' + str(pears_id[massive_galaxies_indices][u]) + '.eps', dpi=300)

        #### Quenching timescale grid plots ####
        ax_gs_quench = fig_quench.add_subplot(111)
        ax_gs_quench.set_title('Quenching Times')
        ax_gs_quench.hist(quenching_times_bc03, 4, histtype='step', align='mid', color='r', linewidth=2)
        ax_gs_quench.hist(quenching_times_fsps, 4, histtype='step', align='mid', color='g', linewidth=2)

        ax_gs_quench.set_yscale('log')
        ax_gs_quench.set_xlim(7, 10)
        ax_gs_quench.set_ylim(1, 1e4)

        fig_quench.savefig(massive_figures_dir + 'quenchdist_grid_bc03_' + str(pears_id[massive_galaxies_indices][u]) + '.eps', dpi=300)

        #### Tau grid plots ####
        #ax_gs_tau = fig_tau.add_subplot(111)
        #ax_gs_tau.set_title('SFH timescale, tau')
        #ax_gs_tau.hist(logtau_bc03, len(np.unique(logtau_bc03)), histtype='step', align='mid', color='r', linewidth=2)
        #ax_gs_tau.hist(logtau_fsps, len(np.unique(logtau_fsps)), histtype='step', align='mid', color='g', linewidth=2)

        ##ax_gs_tau.set_xlim(-2, 2)
        #ax_gs_tau.set_yscale('log')

        #fig_tau.savefig(massive_figures_dir + 'logtaudist_' + str(pears_id[massive_galaxies_indices][u]) + '.eps', dpi=300)

        #del ax_gs_ages, ax_gs_tau, ax_gs_mass_wht_ages, ax_gs_quench

        ############ MILES ############
        #### Metallicity grid plots ####
        #ax_gs_metals = fig_metals_miles.add_subplot(111)
        #ax_gs_metals.set_title('MILES, metallicity')
        #ax_gs_metals.hist(metals_miles, len(np.unique(metals_miles)), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ##ax_gs_metals.set_xlim(0, 0.05)
        #ax_gs_metals.set_yscale('log')

        #fig_metals_miles.savefig(massive_figures_dir + 'metalsdist_grid_miles_'+ str(pears_id[massive_galaxies_indices][u]) + '.eps', dpi=300)

        #del ax_gs_metals

        plt.show()
        """

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)