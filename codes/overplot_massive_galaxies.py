from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits

import sys
import os
import datetime
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, AnchoredText

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/fits_comp_spectra/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
new_codes_dir = home + "/Desktop/FIGS/new_codes/"
lsf_dir = new_codes_dir

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
import grid_coadd as gd
import fast_chi2_jackknife as fcj
import cosmology_calculator as cc

def get_mass_weighted_ages(library, ages, logtau, pearsid):

    # its just a 1d array.
    # These two lines are included so that it behaves with arrays of just one element
    #ages = np.array([ages])
    #logtau = np.array([logtau])

    # the previous array called ages (also called by the same name in an argument for this function)
    # is actually formation time i.e. age of the oldest star
    f_mass_wht_ages = open(savefits_dir + 'jackknife_withlsf_' + str(pearsid) + '_mass_weighted_ages_' + library + '.txt', 'wa')
    timestep = 1e5
    mass_wht_ages = np.zeros(len(ages))

    for j in range(len(ages)):

        formtime = 10**ages[j]
        timearr = np.arange(timestep, formtime, timestep)  # in years
        tau = 10**logtau[j] * 10**9  # in years
        n_arr = np.log10(formtime - timearr) * np.exp(-timearr/tau) * timestep
        d_arr = np.exp(-timearr/tau) * timestep
        
        n = np.sum(n_arr)
        d = np.sum(d_arr)
        mass_wht_ages[j] = n / d
        f_mass_wht_ages.write(str(mass_wht_ages[j]) + ' ')

    f_mass_wht_ages.write('\n')

    print np.max(mass_wht_ages), np.min(mass_wht_ages)
    f_mass_wht_ages.close()
    
    return None

def get_formation_redshifts(library, ages, redshift, pearsid):

    mpc, H_0, omega_m0, omega_r0, omega_lam0, year = cc.get_cosmology_params()

    ao = 1 / (1+redshift)  # scale factor at the observation time i.e. at observed redshift
    age_of_universe = cc.time_after_big_bang(H_0, omega_m0, omega_r0, omega_lam0, ao)[0] * mpc / year  # this age is in years

    f_form_redshifts = open(savefits_dir + 'jackknife_withlsf_' + str(pearsid) + '_formation_redshifts_' + library + '.txt', 'wa')
    formation_redshifts = np.zeros(len(ages))

    redshift_age_lookup_table = np.load(massive_galaxies_dir + 'lookuptable_redshift_ages.npy')
    lookup_z = redshift_age_lookup_table[0]
    lookup_ages = redshift_age_lookup_table[1]

    for j in range(len(ages)):

        formtime = 10**ages[j]
        age_at_formation = age_of_universe - formtime  # in years
        lookup_age_index = np.argmin(abs(lookup_ages - age_at_formation))
        
        f_form_redshifts.write(str(lookup_z[lookup_age_index]) + ' ')
        formation_redshifts[j] = lookup_z[lookup_age_index]

    f_form_redshifts.write('\n')

    print np.median(formation_redshifts)
    f_form_redshifts.close()

    return None

if __name__ == '__main__':
  
    """
    Command line parameter(s):
    run_mode: string
        The run_mode parameter decides if you just want to get mass weighted ages or if you want to run the 
        code to overplot best fit spectra.
        If you would like to just get mass weighted ages then on the command line give it 'get mass weighted ages'. 
        Give it exactly that!

        The running mode also decides if you want to find the best fitting spectrum in the model library by
        looping over all spectra in the model library and finding which one has the closest parameters 
        to the best fit ones (each best fit parameter is the median over all jackknifed runs) or if you
        want to find the best fitting spectrum by just taking the median of all the best fit extension 
        numbers which were saved for each jackknife run.
        Doing some testing it seems like these two numbers are very close.
        The first described way is what I call the 'normal' way and that is the string that the 
        program expects on the command line if you wish to do the matching that way.
        The second way is faster (I suspect more faster if you have a larger number of jackkife runs) and 
        if you want to do the matching that way then you can
        literally give it any other string than 'normal' (except for 'get mass weighted ages').
    """
    run_mode = sys.argv[1]

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # Read pears + 3dhst catalog
    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt', dtype=None, names=True, skip_header=2)

    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    # Find indices for massive galaxies
    massive_galaxies_indices = np.where(stellarmass >= 10.5)[0]

    # Create pdf file to plot figures in
    pdfname = massive_figures_dir + 'overplot_all_sps_massive_galaxies.pdf'
    pdf = PdfPages(pdfname)

    # Create grid for making grid plots
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.2)

    # define number of jackknife samples
    num_jackknife_samps = 1e3

    # To get massweighted ages
    if run_mode == 'get mass weighted ages':

        # Loop over all spectra 
        pears_unique_ids, pears_unique_ids_indices = np.unique(pears_id[massive_galaxies_indices], return_index=True)
        for current_pears_index, count in zip(pears_unique_ids, pears_unique_ids_indices):
            redshift = photz[massive_galaxies_indices][count]
            print "\n", "Currently working with PEARS object id: ", current_pears_index, "with log(M/M_sol) =", stellarmass[massive_galaxies_indices][count], "at redshift", redshift

            # Read files with ages and logtau from jackknife runs
            try:
                #### BC03 ####
                ages_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_ages_bc03.txt', usecols=range(int(num_jackknife_samps)))
                logtau_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_logtau_bc03.txt', usecols=range(int(num_jackknife_samps)))

                #### FSPS ####
                ages_fsps = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_ages_fsps.txt', usecols=range(int(num_jackknife_samps)))
                logtau_fsps = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_logtau_fsps.txt', usecols=range(int(num_jackknife_samps)))

            except IOError as e:
                print e
                print "LSF was not taken into account for this galaxy. Moving on to next galaxy for now."
                continue    

            get_mass_weighted_ages('bc03', ages_bc03, logtau_bc03, current_pears_index)
            get_mass_weighted_ages('fsps', ages_fsps, logtau_fsps, current_pears_index)
            print "Computed and saved mass weighted ages for", current_pears_index, "in", savefits_dir  

        print "Done."
        sys.exit(0)
    
    # To get formation redshifts
    if run_mode == 'get formation redshifts':

        # Loop over all spectra 
        pears_unique_ids, pears_unique_ids_indices = np.unique(pears_id[massive_galaxies_indices], return_index=True)
        for current_pears_index, count in zip(pears_unique_ids, pears_unique_ids_indices):
            redshift = photz[massive_galaxies_indices][count]
            print "\n", "Currently working with PEARS object id: ", current_pears_index, "with log(M/M_sol) =", stellarmass[massive_galaxies_indices][count], "at redshift", redshift

            # Read files with ages from jackknife runs
            try:
                #### BC03 ####
                ages_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_ages_bc03.txt', usecols=range(int(num_jackknife_samps)))

                #### FSPS ####
                ages_fsps = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_ages_fsps.txt', usecols=range(int(num_jackknife_samps)))

            except IOError as e:
                print e
                print "LSF was not taken into account for this galaxy. Moving on to next galaxy for now."
                continue   

            get_formation_redshifts('bc03', ages_bc03, redshift, current_pears_index)
            get_formation_redshifts('fsps', ages_fsps, redshift, current_pears_index)
            print "Computed and saved formation redshifts for", current_pears_index, "in", savefits_dir  

        print "Done."
        sys.exit(0)

    best_age_bc03 = []
    best_tau_bc03 = []
    best_mass_wht_age_bc03 = []
    best_form_redshift_bc03 = []
    best_alpha_bc03 = []

    best_age_fsps = []
    best_tau_fsps = []
    best_mass_wht_age_fsps = []
    best_form_redshift_fsps = []
    best_alpha_fsps = []

    stellarmass_arr = []

    # Loop over all spectra 
    pears_unique_ids, pears_unique_ids_indices = np.unique(pears_id[massive_galaxies_indices], return_index=True)
    for current_pears_index, count in zip(pears_unique_ids, pears_unique_ids_indices):
        redshift = photz[massive_galaxies_indices][count]
        print "\n", "Currently working with PEARS object id: ", current_pears_index, "with log(M/M_sol) =", stellarmass[massive_galaxies_indices][count], "at redshift", redshift

        lam_em, flam_em, ferr, specname = gd.fileprep(current_pears_index, redshift)
        
        resampling_lam_grid = lam_em 

        # Read files with params from jackknife runs
        try:
            #### BC03 ####
            ages_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_ages_bc03.txt', usecols=range(int(num_jackknife_samps)))
            logtau_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_logtau_bc03.txt', usecols=range(int(num_jackknife_samps)))
            tauv_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_tauv_bc03.txt', usecols=range(int(num_jackknife_samps)))
            exten_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_exten_bc03.txt', usecols=range(int(num_jackknife_samps)))
            chi2_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_chi2_bc03.txt', usecols=range(int(num_jackknife_samps)))
            alpha_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_alpha_bc03.txt', usecols=range(int(num_jackknife_samps)))
    
            #### MILES ####
            #ages_miles = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_ages_miles.txt', usecols=range(int(num_jackknife_samps)))
            #metals_miles = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_metals_miles.txt', usecols=range(int(num_jackknife_samps)))
            #exten_miles = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_exten_miles.txt', usecols=range(int(num_jackknife_samps)))
            #chi2_miles = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_chi2_miles.txt', usecols=range(int(num_jackknife_samps)))
            #alpha_miles = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_alpha_miles.txt', usecols=range(int(num_jackknife_samps)))

            #### FSPS ####
            ages_fsps = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_ages_fsps.txt', usecols=range(int(num_jackknife_samps)))
            logtau_fsps = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_logtau_fsps.txt', usecols=range(int(num_jackknife_samps)))
            exten_fsps = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_exten_fsps.txt', usecols=range(int(num_jackknife_samps)))
            chi2_fsps = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_chi2_fsps.txt', usecols=range(int(num_jackknife_samps)))
            alpha_fsps = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_alpha_fsps.txt', usecols=range(int(num_jackknife_samps)))

        except IOError as e:
            print e
            print "LSF was not taken into account for this galaxy. Moving on to next galaxy for now."
            continue            

        if (run_mode != 'get mass weighted ages') or (run_mode != 'get formation redshifts'):
            mass_wht_ages_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_mass_weighted_ages_bc03.txt', usecols=range(int(num_jackknife_samps)))
            mass_wht_ages_fsps = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_mass_weighted_ages_fsps.txt', usecols=range(int(num_jackknife_samps)))

            formation_redshifts_bc03 = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_formation_redshifts_bc03.txt', usecols=range(int(num_jackknife_samps)))
            formation_redshifts_fsps = np.loadtxt(savefits_dir + 'jackknife_withlsf_' + str(current_pears_index) + '_formation_redshifts_fsps.txt', usecols=range(int(num_jackknife_samps)))

        # Open fits files with comparison spectra
        bc03_spec = fits.open(savefits_dir + 'all_comp_spectra_bc03_solar_withlsf_' + str(current_pears_index) + '.fits', memmap=False)
        #miles_spec = fits.open(savefits_dir + 'all_comp_spectra_miles_withlsf_' + str(current_pears_index) + '.fits', memmap=False)
        fsps_spec = fits.open(savefits_dir + 'all_comp_spectra_fsps_withlsf_' + str(current_pears_index) + '.fits', memmap=False)  

        # Find number of extensions in each
        bc03_extens = fcj.get_total_extensions(bc03_spec)
        #miles_extens = fcj.get_total_extensions(miles_spec)
        fsps_extens = fcj.get_total_extensions(fsps_spec)

        # Make multi-dimensional arrays of parameters for all SPS libraries 
        # This is done to make access to the fits headers easier because
        # there is no way to directly search the headers for all extensions in a fits file simultaneously.
        # Once the numpy arrays have the header values in them the arrays can easily be searched by using np.where

        if run_mode == 'normal':

            bc03_params = np.empty([bc03_extens, 3])  # the 3 is because the BC03 param space is 3D, in this case (metallicity fixed to solar)
            for i in range(bc03_extens):
                age = float(bc03_spec[i+1].header['LOG_AGE'])
                tau = float(bc03_spec[i+1].header['TAU_GYR'])
                tauv = float(bc03_spec[i+1].header['TAUV'])
    
                bc03_params[i] = np.array([age, tau, tauv])
    
            miles_params = np.empty([miles_extens, 2])  # the 2 is because the MILES param space is 2D, in this case
            for i in range(miles_extens):
                age = float(miles_spec[i+1].header['LOG_AGE'])
                z = float(miles_spec[i+1].header['METAL'])
    
                miles_params[i] = np.array([age, z])
    
            fsps_params = np.empty([fsps_extens, 2])  # the 2 is because the FSPS param space is 2D, in this case (metallicity fixed to solar)
            for i in range(fsps_extens):
                age = float(fsps_spec[i+1].header['LOG_AGE'])
                tau = float(fsps_spec[i+1].header['TAU_GYR'])
    
                fsps_params[i] = np.array([age, tau])

        # Make figure and axes and plot stacked spectrum with errors from bootstrap
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[:10,:])
        ax2 = fig.add_subplot(gs[10:,:])

        ax1.set_ylabel('$f_{\lambda}\ [\mathrm{erg/s/cm^2/\AA}] $')
        ax2.set_xlabel('$\lambda\ [\AA]$')

        ax1.plot(resampling_lam_grid, flam_em, color='k')
        ax1.fill_between(resampling_lam_grid, flam_em + ferr, flam_em - ferr, color='lightgray')
        ax1.set_xlim(2500, 6000)
        ax1.xaxis.set_ticklabels([])
        ax1.minorticks_on()
        ax1.tick_params('both', width=1, length=3, which='minor')
        ax1.tick_params('both', width=1, length=4.7, which='major')

        # Find the best fit parameters and plot them after matching
        #### BC03 ####
        best_age = np.median(ages_bc03)
        best_tau = 10**np.median(logtau_bc03)
        best_tauv = np.median(tauv_bc03)
        best_mass_wht_age = np.median(mass_wht_ages_bc03)
        best_form_redshift = np.median(formation_redshifts_bc03)
        best_exten = int(np.median(exten_bc03))
        best_chi2 = np.min(chi2_bc03)
        best_alpha = np.median(alpha_bc03)

        best_age_err = np.std(ages_bc03) * 10**best_age / (1e9 * 0.434)
        # the number 0.434 is (1 / ln(10)) 
        # the research notebook has this short calculation at the end 
        best_tau_err = np.std(logtau_bc03) * best_tau / 0.434
        best_tauv_err = np.std(tauv_bc03)
        best_mass_wht_age_err = np.std(mass_wht_ages_bc03) * 10**best_mass_wht_age / (1e9 * 0.434)

        best_age_bc03.append(best_age)
        best_tau_bc03.append(best_tau)
        best_mass_wht_age_bc03.append(best_mass_wht_age)
        best_form_redshift_bc03.append(best_form_redshift)
        best_alpha_bc03.append(best_alpha)
        stellarmass_arr.append(stellarmass[massive_galaxies_indices][count])

        #print 'bc03', best_age, best_age_err, best_tau, best_tau_err, best_mass_wht_age, best_mass_wht_age_err, stellarmass[massive_galaxies_indices][count]

        if run_mode == 'normal':
            for j in range(bc03_extens):
                if np.allclose(bc03_params[j], np.array([best_age, best_tau, best_tauv]).reshape(3)):
                    best_exten = j + 1       

        currentspec = bc03_spec[best_exten].data

        alpha = np.sum(flam_em * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)
        print best_chi2, best_alpha, alpha

        # Plot best fit parameters as anchored text boxes
        redshiftbox = TextArea(r'$z$ = ' + str(redshift), textprops=dict(color='k', size=9))
        anc_redshiftbox = AnchoredOffsetbox(loc=2, child=redshiftbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.03, 0.65),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_redshiftbox)                

        form_redshiftbox = TextArea(r'$\mathrm{z_{formation}}$ = ' + "{:.2f}".format(float(best_form_redshift)), textprops=dict(color='r', size=9))
        anc_form_redshiftbox = AnchoredOffsetbox(loc=2, child=form_redshiftbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.03, 0.8),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_form_redshiftbox) 

        stellarmassbox = TextArea(r'$\mathrm{M}_* = 10^{' + str(stellarmass[massive_galaxies_indices][count]) + r'}$' + r' $\mathrm{M}_\odot$', textprops=dict(color='k', size=9))
        anc_stellarmassbox = AnchoredOffsetbox(loc=2, child=stellarmassbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.03, 0.6),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_stellarmassbox)   

        pearsidbox = TextArea(str(current_pears_index), textprops=dict(color='k', size=9))
        anc_pearsidbox = AnchoredOffsetbox(loc=2, child=pearsidbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.03, 0.55),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_pearsidbox)   

        labelbox = TextArea("BC03", textprops=dict(color='r', size=8))
        anc_labelbox = AnchoredOffsetbox(loc=2, child=labelbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.03, 0.95),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_labelbox)

        agebox = TextArea(r"$\left<t\right>_M$ = " + "{:.2f}".format(float(10**best_mass_wht_age/1e9)) + r" $\pm$ " + "{:.2f}".format(float(best_mass_wht_age_err)) + " Gyr",
         textprops=dict(color='r', size=8))
        anc_agebox = AnchoredOffsetbox(loc=2, child=agebox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.03, 0.9),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_agebox)

        #metalbox = TextArea("Z = " + "{:.2f}".format(0.02), textprops=dict(color='r', size=8))  # because metallicity is fixed to solar
        #anc_metalbox = AnchoredOffsetbox(loc=2, child=metalbox, pad=0.0, frameon=False,\
        #                                     bbox_to_anchor=(0.03, 0.8),\
        #                                     bbox_transform=ax1.transAxes, borderpad=0.0)
        #ax1.add_artist(anc_metalbox)

        taubox = TextArea(r"$\tau$ = " + "{:.2f}".format(float(best_tau)) + r" $\pm$ " + "{:.2f}".format(float(best_tau_err)) + " Gyr",
         textprops=dict(color='r', size=8))
        anc_taubox = AnchoredOffsetbox(loc=2, child=taubox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.03, 0.85),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_taubox)

        chi2box = TextArea(r'$\mathrm{\chi^2_{red}}$ = ' + "{:.2f}".format(float(best_chi2/len(currentspec))), textprops=dict(color='r', size=9))
        anc_chi2box = AnchoredOffsetbox(loc=2, child=chi2box, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.03, 0.75),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_chi2box) 

        alphabox = TextArea(r'$\alpha$ = ' + str(best_alpha), textprops=dict(color='r', size=9))
        anc_alphabox = AnchoredOffsetbox(loc=2, child=alphabox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.03, 0.7),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_alphabox) 

        #tauvbox = TextArea(r"$A_v$ = " + "{:.2f}".format(float(best_tauv)) + r" $\pm$ " + "{:.2f}".format(float(best_tauv_err)),
        # textprops=dict(color='r', size=8))
        #anc_tauvbox = AnchoredOffsetbox(loc=2, child=tauvbox, pad=0.0, frameon=False,\
        #                                     bbox_to_anchor=(0.03, 0.75),\
        #                                     bbox_transform=ax1.transAxes, borderpad=0.0)
        #ax1.add_artist(anc_tauvbox)

        # Plot the best fit spectrum
        ax1.plot(resampling_lam_grid, alpha * currentspec, color='r')
        ax1.set_xlim(2500, 6000)
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_tick_params(labelsize=9)

        ax1.minorticks_on()
        ax1.tick_params('both', width=1, length=3, which='minor')
        ax1.tick_params('both', width=1, length=4.7, which='major')
                
        # Plot the residual
        ax2.plot(resampling_lam_grid, flam_em - alpha * currentspec, '-', color='r', drawstyle='steps-mid')

        ax2.set_ylim(-1e-17, 1e-17)
        ax2.set_xlim(2500, 6000)
        ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax2.xaxis.set_tick_params(labelsize=10)

        ax2.axhline(y=0.0, color='k', linestyle='--')
        ax2.grid(True)
                
        ax2.minorticks_on()
        ax2.tick_params('both', width=1, length=3, which='minor')
        ax2.tick_params('both', width=1, length=4.7, which='major')

        #### MILES ####
        """
        best_age = np.median(ages_miles)
        best_metal = np.median(metals_miles)
        best_exten = int(np.median(exten_miles))

        best_age_err = np.std(ages_miles) * 10**best_age / (1e9 * 0.434)
        best_metal_err = np.std(metals_miles)

        #print 'miles', best_age, best_metal, stellarmass[massive_galaxies_indices][count]

        if run_mode == 'normal':
            for j in range(miles_extens):
                if np.allclose(miles_params[j], np.array([best_age, best_metal]).reshape(2)):
                    best_exten = j + 1

        currentspec = miles_spec[best_exten].data
        mask_indices = np.isnan(miles_spec[best_exten].data)
        currentspec = ma.masked_array(currentspec, mask = mask_indices)

        alpha = np.sum(flam_em * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)

        # Plot best fit parameters as anchored text boxes
        labelbox = TextArea("MILES", textprops=dict(color='b', size=8))
        anc_labelbox = AnchoredOffsetbox(loc=2, child=labelbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.28, 0.95),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_labelbox)

        agebox = TextArea(r"$t$ = " + "{:.2f}".format(float(10**best_age/1e9)) + r" $\pm$ " + "{:.2f}".format(float(best_age_err)) + " Gyr",
         textprops=dict(color='b', size=8))
        anc_agebox = AnchoredOffsetbox(loc=2, child=agebox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.28, 0.9),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_agebox)

        metalbox = TextArea("Z = " + "{:.2f}".format(float(best_metal)) + r" $\pm$ " + "{:.2f}".format(float(best_metal_err)),
         textprops=dict(color='b', size=8))
        anc_metalbox = AnchoredOffsetbox(loc=2, child=metalbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.28, 0.85),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_metalbox)

        # Plot the best fit spectrum
        ax1.plot(resampling_lam_grid, alpha * currentspec, color='b')
        ax1.set_xlim(2500, 6000)
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_tick_params(labelsize=9)

        ax1.minorticks_on()
        ax1.tick_params('both', width=1, length=3, which='minor')
        ax1.tick_params('both', width=1, length=4.7, which='major')
                
        # Plot the residual
        ax2.plot(resampling_lam_grid, flam_em - alpha * currentspec, '-', color='b', drawstyle='steps-mid')
                
        ax2.set_ylim(-1e-17, 1e-17)
        ax2.set_xlim(2500, 6000)
        ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax2.xaxis.set_tick_params(labelsize=10)

        ax2.axhline(y=0.0, color='k', linestyle='--')
        ax2.grid(True)
                
        ax2.minorticks_on()
        ax2.tick_params('both', width=1, length=3, which='minor')
        ax2.tick_params('both', width=1, length=4.7, which='major')
        """

        #### FSPS ####
        best_age = np.median(ages_fsps)
        best_tau = 10**np.median(logtau_fsps)
        best_mass_wht_age = np.median(mass_wht_ages_fsps)
        best_form_redshift = np.median(formation_redshifts_fsps)
        best_exten = int(np.median(exten_fsps))
        best_chi2 = np.min(chi2_fsps)
        best_alpha = np.median(alpha_fsps)

        best_age_err = np.std(ages_fsps) * 10**best_age / (1e9 * 0.434)
        best_tau_err = np.std(logtau_fsps) * best_tau / 0.434
        best_mass_wht_age_err = np.std(mass_wht_ages_fsps) * 10**best_mass_wht_age / (1e9 * 0.434)

        best_age_fsps.append(best_age)
        best_tau_fsps.append(best_tau)
        best_mass_wht_age_fsps.append(best_mass_wht_age)
        best_form_redshift_fsps.append(best_form_redshift)
        best_alpha_fsps.append(best_alpha)

        #print 'fsps', best_age, best_age_err, best_tau, best_tau_err, best_mass_wht_age, best_mass_wht_age_err, stellarmass[massive_galaxies_indices][count]

        if run_mode == 'normal':
            for j in range(fsps_extens):
                if np.allclose(fsps_params[j], np.array([best_age, best_tau]).reshape(2)):
                    best_exten = j + 1
        
        currentspec = fsps_spec[best_exten].data

        alpha = np.sum(flam_em * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)
        print best_chi2, best_alpha, alpha

        # Plot best fit parameters as anchored text boxes
        labelbox = TextArea("FSPS", textprops=dict(color='g', size=8))
        anc_labelbox = AnchoredOffsetbox(loc=2, child=labelbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.28, 0.95),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_labelbox)

        agebox = TextArea(r"$\left<t\right>_M$ = " + "{:.2f}".format(float(10**best_mass_wht_age/1e9)) + r" $\pm$ " + "{:.2f}".format(float(best_mass_wht_age_err)) + " Gyr",
         textprops=dict(color='g', size=8))
        anc_agebox = AnchoredOffsetbox(loc=2, child=agebox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.28, 0.9),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_agebox)

        taubox = TextArea(r"$\tau$ = " + "{:.2f}".format(float(best_tau)) + r" $\pm$ " + "{:.2f}".format(float(best_tau_err)) + " Gyr",
         textprops=dict(color='g', size=8))
        anc_taubox = AnchoredOffsetbox(loc=2, child=taubox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.28, 0.85),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_taubox)

        form_redshiftbox = TextArea(r'$\mathrm{z_{formation}}$ = ' + "{:.2f}".format(float(best_form_redshift)), textprops=dict(color='g', size=9))
        anc_form_redshiftbox = AnchoredOffsetbox(loc=2, child=form_redshiftbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.28, 0.8),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_form_redshiftbox) 

        chi2box = TextArea(r'$\mathrm{\chi^2_{red}}$ = ' + "{:.2f}".format(float(best_chi2/len(currentspec))), textprops=dict(color='g', size=9))
        anc_chi2box = AnchoredOffsetbox(loc=2, child=chi2box, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.28, 0.75),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_chi2box) 

        alphabox = TextArea(r'$\alpha$ = ' + str(best_alpha), textprops=dict(color='g', size=9))
        anc_alphabox = AnchoredOffsetbox(loc=2, child=alphabox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.28, 0.7),\
                                             bbox_transform=ax1.transAxes, borderpad=0.0)
        ax1.add_artist(anc_alphabox) 

        # Plot the best fit spectrum
        ax1.plot(resampling_lam_grid, alpha * currentspec, color='g')
        ax1.set_xlim(2500, 6000)
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_tick_params(labelsize=9)

        ax1.minorticks_on()
        ax1.tick_params('both', width=1, length=3, which='minor')
        ax1.tick_params('both', width=1, length=4.7, which='major')
                
        # Plot the residual
        ax2.plot(resampling_lam_grid, flam_em - alpha * currentspec, '-', color='g', drawstyle='steps-mid')
                
        ax2.set_ylim(-1e-17, 1e-17)
        ax2.set_xlim(2500, 6000)
        ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax2.xaxis.set_tick_params(labelsize=10)

        ax2.get_yaxis().set_ticklabels(['-1', '-0.5', '0.0', '0.5', ''], fontsize=8, rotation=45)

        ax2.axhline(y=0.0, color='k', linestyle='--')
        ax2.grid(True)

        ax2.minorticks_on()
        ax2.tick_params('both', width=1, length=3, which='minor')
        ax2.tick_params('both', width=1, length=4.7, which='major')

        # Get the residual tick label to show that the axis ticks are multiplied by 10^-18 
        resid_labelbox = TextArea(r"$\times 1 \times 10^{-17}$", textprops=dict(color='k', size=8))
        anc_resid_labelbox = AnchoredOffsetbox(loc=2, child=resid_labelbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.01, 0.98),\
                                             bbox_transform=ax2.transAxes, borderpad=0.0)
        ax2.add_artist(anc_resid_labelbox)

        pdf.savefig(bbox_inches='tight')
        plt.close()
        del fig, ax1, ax2
    
    best_age_bc03 = np.asarray(best_age_bc03)
    best_tau_bc03 = np.asarray(best_tau_bc03)
    best_mass_wht_age_bc03 = np.asarray(best_mass_wht_age_bc03)
    best_form_redshift_bc03 = np.asarray(best_form_redshift_bc03)
    best_alpha_bc03 = np.asarray(best_alpha_bc03)

    best_age_fsps = np.asarray(best_age_fsps)
    best_tau_fsps = np.asarray(best_tau_fsps)
    best_mass_wht_age_fsps = np.asarray(best_mass_wht_age_fsps)
    best_form_redshift_fsps = np.asarray(best_form_redshift_fsps)
    best_alpha_fsps = np.asarray(best_alpha_fsps)

    stellarmass_arr = np.asarray(stellarmass_arr)

    save_array = np.vstack((best_age_bc03, best_tau_bc03, best_mass_wht_age_bc03, best_form_redshift_bc03, best_alpha_bc03, best_age_fsps, best_tau_fsps, best_mass_wht_age_fsps, best_form_redshift_fsps, best_alpha_fsps, stellarmass_arr))

    np.save(massive_galaxies_dir + "best_fit_params", save_array)

    pdf.close()

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)
