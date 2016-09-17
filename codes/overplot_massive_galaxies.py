from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from scipy import stats

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
savefits_dir = home + "/Desktop/FIGS/new_codes/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd
import fast_chi2_jackknife as fcj

def get_mass_weighted_ages(library, ages, logtau, pearsid):

    # its just a 1d array.
    # These two lines are included so that it behaves with arrays with just one element
    ages = np.array([ages])
    logtau = np.array([logtau])

    # the previous array called ages (also called by the same name in an argument for this function)
    # is actually formation time i.e. age of the oldest star
    f_mass_wht_ages = open(savefits_dir + 'jackknife' + str(pearsid) + '_mass_weighted_ages_' + library + '.txt', 'wa')
    timestep = 1e5
    mass_wht_ages = np.zeros(len(ages))

    for j in range(len(ages)):
        formtime = 10**ages[j]
        timearr = np.arange(timestep, formtime, timestep) # in years
        tau = 10**logtau[j] * 10**9 # in years
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

    # Create pdf file to plot figures in
    pdfname = massive_figures_dir + 'overplot_all_sps_massive_galaxies.pdf'
    pdf = PdfPages(pdfname)

    # Create grid for making grid plots
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.2)

    num_jackknife_samps = 1
    # Loop over all spectra 
    for u in range(len(pears_id[massive_galaxies_indices])):

        print "Currently working with PEARS object id: ", pears_id[massive_galaxies_indices][u]

        redshift = photz[massive_galaxies_indices][u]
        lam_em, flam_em, ferr, specname = gd.fileprep(pears_id[massive_galaxies_indices][u], redshift)

        resampling_lam_grid = lam_em 

        # Read files with params from jackknife runs
        #### BC03 ####
        ages_bc03 = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_ages_bc03.txt', usecols=range(int(num_jackknife_samps)))
        logtau_bc03 = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_logtau_bc03.txt', usecols=range(int(num_jackknife_samps)))
        tauv_bc03 = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_tauv_bc03.txt', usecols=range(int(num_jackknife_samps)))
        mass_wht_ages_bc03 = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_mass_weighted_ages_bc03.txt', usecols=range(int(num_jackknife_samps)))
    
        #### MILES ####
        ages_miles = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_ages_miles.txt', usecols=range(int(num_jackknife_samps)))
        metals_miles = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_metals_miles.txt', usecols=range(int(num_jackknife_samps)))
    
        #### FSPS ####
        ages_fsps = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_ages_fsps.txt', usecols=range(int(num_jackknife_samps)))
        logtau_fsps = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_logtau_fsps.txt', usecols=range(int(num_jackknife_samps)))
        mass_wht_ages_fsps = np.loadtxt(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_mass_weighted_ages_fsps.txt', usecols=range(int(num_jackknife_samps)))

        # If you need to make the mass-weighted ages files, just after running the chi2 fitting code, run the next two lines.
        # Make sure to comment out the rest of the code below it.
        #get_mass_weighted_ages('bc03', ages_bc03, logtau_bc03, pears_id[massive_galaxies_indices][u])
        #get_mass_weighted_ages('fsps', ages_fsps, logtau_fsps, pears_id[massive_galaxies_indices][u])

        # Open fits files with comparison spectra
        bc03_spec = fits.open(savefits_dir + 'all_comp_spectra_bc03_solar_' + str(pears_id[massive_galaxies_indices][u]) + '.fits', memmap=False)
        miles_spec = fits.open(savefits_dir + 'all_comp_spectra_miles_' + str(pears_id[massive_galaxies_indices][u]) + '.fits', memmap=False)
        fsps_spec = fits.open(savefits_dir + 'all_comp_spectra_fsps_' + str(pears_id[massive_galaxies_indices][u]) + '.fits', memmap=False)    

        # Find number of extensions in each
        bc03_extens = fcj.get_total_extensions(bc03_spec)
        miles_extens = fcj.get_total_extensions(miles_spec)
        fsps_extens = fcj.get_total_extensions(fsps_spec)
    
        # Make multi-dimensional arrays of parameters for all SPS libraries 
        # This is done to make access to the fits headers easier because
        # there is no way to directly search the headers for all extensions in a fits file simultaneously.
        # Once the numpy arrays have the header values in them the arrays can easily be searched by using np.where
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
        best_age = stats.mode(ages_bc03)[0]
        best_tau = 10**stats.mode(logtau_bc03)[0]
        best_tauv = stats.mode(tauv_bc03)[0]
        best_mass_wht_age = stats.mode(mass_wht_ages_bc03)[0]

        best_age_err = np.std(ages_bc03) * 10**best_age / (1e9 * 0.434)
        # the number 0.434 is (1 / ln(10)) 
        # the research notebook has this short calculation at the end 
        best_tau_err = np.std(logtau_bc03) * best_tau / 0.434
        best_tauv_err = np.std(tauv_bc03)
        best_mass_wht_age_err = np.std(mass_wht_ages_bc03) * 10**best_mass_wht_age / (1e9 * 0.434)

        for j in range(bc03_extens):
            if np.allclose(bc03_params[j], np.array([best_age, best_tau, best_tauv]).reshape(3)):
                currentspec = bc03_spec[j+1].data

                alpha = np.sum(flam_em * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)

                # Plot best fit parameters as anchored text boxes
                redshiftbox = TextArea(r'$z$ = ' + str(redshift), textprops=dict(color='k', size=9))
                anc_redsfihtbox = AnchoredOffsetbox(loc=2, child=redshiftbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.75),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_redsfihtbox)                

                stellarmassbox = TextArea(r'$\mathrm{M}_* = 10^{' + str(stellarmass[massive_galaxies_indices][u]) + r'}$' + r' $\mathrm{M}_\odot$', textprops=dict(color='k', size=9))
                anc_stellarmassbox = AnchoredOffsetbox(loc=2, child=stellarmassbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.7),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_stellarmassbox)   

                pearsidbox = TextArea(str(pears_id[massive_galaxies_indices][u]), textprops=dict(color='k', size=9))
                anc_pearsidbox = AnchoredOffsetbox(loc=2, child=pearsidbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.65),\
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

                metalbox = TextArea("Z = " + "{:.2f}".format(0.02), textprops=dict(color='r', size=8))  # because metallicity is fixed to solar
                anc_metalbox = AnchoredOffsetbox(loc=2, child=metalbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.8),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_metalbox)

                taubox = TextArea(r"$\tau$ = " + "{:.2f}".format(float(best_tau)) + r" $\pm$ " + "{:.2f}".format(float(best_tau_err)) + " Gyr",
                 textprops=dict(color='r', size=8))
                anc_taubox = AnchoredOffsetbox(loc=2, child=taubox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.85),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_taubox)

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

        del best_age, best_tau, best_tauv

        #### MILES ####
        best_age = stats.mode(ages_miles)[0]
        best_metal = stats.mode(metals_miles)[0]

        best_age_err = np.std(ages_miles) * 10**best_age / (1e9 * 0.434)
        best_metal_err = np.std(metals_miles)

        for j in range(miles_extens):
            if np.allclose(miles_params[j], np.array([best_age, best_metal]).reshape(2)):
                currentspec = miles_spec[j+1].data
                mask_indices = np.isnan(miles_spec[j+1].data)
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

        del best_age, best_metal

        #### FSPS ####
        best_age = stats.mode(ages_fsps)[0]
        best_tau = 10**stats.mode(logtau_fsps)[0]
        best_mass_wht_age = stats.mode(mass_wht_ages_fsps)[0]

        best_age_err = np.std(ages_fsps) * 10**best_age / (1e9 * 0.434)
        best_tau_err = np.std(logtau_fsps) * best_tau / 0.434
        best_mass_wht_age_err = np.std(mass_wht_ages_fsps) * 10**best_mass_wht_age / (1e9 * 0.434)

        for j in range(fsps_extens):
            if np.allclose(fsps_params[j], np.array([best_age, best_tau]).reshape(2)):
                currentspec = fsps_spec[j+1].data

                alpha = np.sum(flam_em * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)

                # Plot best fit parameters as anchored text boxes
                labelbox = TextArea("FSPS", textprops=dict(color='g', size=8))
                anc_labelbox = AnchoredOffsetbox(loc=2, child=labelbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.53, 0.95),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_labelbox)

                agebox = TextArea(r"$\left<t\right>_M$ = " + "{:.2f}".format(float(10**best_mass_wht_age/1e9)) + r" $\pm$ " + "{:.2f}".format(float(best_mass_wht_age_err)) + " Gyr",
                 textprops=dict(color='g', size=8))
                anc_agebox = AnchoredOffsetbox(loc=2, child=agebox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.53, 0.9),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_agebox)

                metalbox = TextArea("Z = " + "{:.2f}".format(0.02), textprops=dict(color='r', size=8))  # because metallicity is fixed to solar
                anc_metalbox = AnchoredOffsetbox(loc=2, child=metalbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.8),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)

                taubox = TextArea(r"$\tau$ = " + "{:.2f}".format(float(best_tau)) + r" $\pm$ " + "{:.2f}".format(float(best_tau_err)) + " Gyr",
                 textprops=dict(color='g', size=8))
                anc_taubox = AnchoredOffsetbox(loc=2, child=taubox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.53, 0.85),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_taubox)

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

        del best_age, best_tau

        pdf.savefig(bbox_inches='tight')

    pdf.close()

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)
