from __future__ import division

import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy.cosmology import Planck15 as cosmo

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, AnchoredText

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/fits_comp_spectra/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
new_codes_dir = home + "/Desktop/FIGS/new_codes/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd
import create_fsps_miles_libraries as ct
import fast_chi2_jackknife as fcj
import fast_chi2_jackknife_massive_galaxies as fcjm

def get_interplsf(pearsid, redshift, fieldforid, pa_forlsf, interp_factor=1):

    pa_str = pa_forlsf.replace('PA', 'pa')

    if fieldforid == 'GOODS-N':
        lsf = np.loadtxt(home + '/Desktop/FIGS/new_codes/pears_lsfs/north_lsfs/n' + str(pearsid) + '_' + pa_str + '_lsf.txt')
    elif fieldforid == 'GOODS-S':
        lsf = np.loadtxt(home + '/Desktop/FIGS/new_codes/pears_lsfs/south_lsfs/n' + str(pearsid) + '_' + pa_str + '_lsf.txt')

    # Interpolate the LSF to the rest frame delta lambda grid of the galaxy
    dlam_obs = 24
    dlam_em = dlam_obs / (1+redshift)
    interppoints = np.linspace(0, len(lsf), int(len(lsf)*dlam_em*interp_factor))
    interplsf = np.interp(interppoints, xp=np.arange(len(lsf)), fp=lsf)

    return interplsf

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # data for the galaxy being tested
    current_id = 36105  # 81011
    current_field = 'GOODS-N'  # 'GOODS-S'
    current_redshift = 0.936  # 0.938

    lam_em, flam_em, ferr, specname, pa_forlsf = gd.fileprep(current_id, current_redshift, current_field, apply_smoothing=True, width=2, kernel_type='gauss')

    resampling_lam_grid = lam_em 

    ## extend lam_grid to be able to move the lam_grid later 
    #avg_dlam = get_avg_dlam(lam_em)

    #lam_low_to_insert = np.arange(1500, lam_em[0], avg_dlam)
    #lam_high_to_append = np.arange(lam_em[-1] + avg_dlam, 7500, avg_dlam)

    #resampling_lam_grid = np.insert(lam_em, obj=0, values=lam_low_to_insert)
    #resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    #create_bc03_lib_ssp_csp(current_id, current_redshift, current_field, resampling_lam_grid, pa_forlsf, include_csp=True)

    # read in comparison model spectra with and without lsf
    #bc03_spec_withlsf = fits.open(savefits_dir + 'all_comp_spectra_bc03_solar_withlsf_' + str(current_id) + '.fits', memmap=False)
    #bc03_spec_nolsf = fits.open(savefits_dir + 'all_comp_spectra_bc03_solar_' + str(current_id) + '.fits', memmap=False)

    # Get random samples by bootstrapping
    #num_samp_to_draw = int(1)
    #if num_samp_to_draw == 1:
    #    resampled_spec = flam_em
    #else:
    #    print "Running over", num_samp_to_draw, "random bootstrapped samples."
    #    resampled_spec = ma.empty((len(flam_em), num_samp_to_draw))
    #    for i in range(len(flam_em)):
    #        if flam_em[i] is not ma.masked:
    #            resampled_spec[i] = np.random.normal(flam_em[i], ferr[i], num_samp_to_draw)
    #        else:
    #            resampled_spec[i] = ma.masked
    #    resampled_spec = resampled_spec.T

    # set up figure to plot comparisons in
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ferr = ferr + 0.01 * flam_em

    ax.plot(lam_em, flam_em, '-', color='k')
    ax.fill_between(lam_em, flam_em + ferr, flam_em - ferr, color='lightgray')

    # run the comparison code 
    """
    bc03_specs = [bc03_spec_withlsf]#[bc03_spec_nolsf, bc03_spec_withlsf]

    comp_count = 1
    for bc03_spec in bc03_specs:

        bc03_extens = fcj.get_total_extensions(bc03_spec)

        comp_spec_bc03 = np.zeros([bc03_extens, len(resampling_lam_grid)], dtype=np.float64)
        for i in range(bc03_extens):
            comp_spec_bc03[i] = bc03_spec[i+1].data

        flam = resampled_spec
        currentspec = comp_spec_bc03

        chi2 = np.zeros(bc03_extens, dtype=np.float64)
        alpha = np.sum(flam * currentspec / (ferr**2), axis=1) / np.sum(currentspec**2 / ferr**2, axis=1)
        chi2 = np.sum(((flam - (alpha * currentspec.T).T) / ferr)**2, axis=1)

        # This is to get only physical ages
        sortargs = np.argsort(chi2)
        for k in range(len(chi2)):
            best_age = float(bc03_spec[sortargs[k] + 1].header['LOG_AGE'])
            age_at_z = cosmo.age(current_redshift).value * 1e9 # in yr
            if (best_age < np.log10(age_at_z)) & (best_age > 9 + np.log10(0.1)):
                fitage = best_age
                fitmetal = float(bc03_spec[sortargs[k] + 1].header['METAL'])
                fitlogtau = float(bc03_spec[sortargs[k] + 1].header['TAU_GYR'])
                fittauv = float(bc03_spec[sortargs[k] + 1].header['TAUV'])
                best_exten = sortargs[k] + 1
                bestalpha = alpha[sortargs[k]]
                fitchi2 = chi2[sortargs[k]]
                best_fit_model = currentspec[sortargs[k]]
                break

        # get mass weighted age
        timestep = 1e5
        formtime = 10**fitage
        timearr = np.arange(timestep, formtime, timestep)  # in years
        tau = 10**fitlogtau * 10**9  # in years
        n_arr = np.log10(formtime - timearr) * np.exp(-timearr/tau) * timestep
        d_arr = np.exp(-timearr/tau) * timestep
        
        n = np.sum(n_arr)
        d = np.sum(d_arr)
        mass_wht_age = n / d

        # put in labels
        id_labelbox = TextArea(current_field + "  " + str(current_id), textprops=dict(color='k', size=12))
        anc_id_labelbox = AnchoredOffsetbox(loc=2, child=id_labelbox, pad=0.0, frameon=False,\
                                             bbox_to_anchor=(0.1, 0.9),\
                                             bbox_transform=ax.transAxes, borderpad=0.0)
        ax.add_artist(anc_id_labelbox)

        if comp_count == 0:
            col = 'b'

            without_labelbox = TextArea("Without LSF", textprops=dict(color='b', size=12))
            anc_without_labelbox = AnchoredOffsetbox(loc=2, child=without_labelbox, pad=0.0, frameon=False,\
                                                 bbox_to_anchor=(0.05, 0.8),\
                                                 bbox_transform=ax.transAxes, borderpad=0.0)
            ax.add_artist(anc_without_labelbox)

            chi2_without_labelbox = TextArea(r"$\chi^2_{red} = $" + str("{:.4}".format(fitchi2/len(flam))), textprops=dict(color='b', size=12))
            anc_chi2_without_labelbox = AnchoredOffsetbox(loc=2, child=chi2_without_labelbox, pad=0.0, frameon=False,\
                                                 bbox_to_anchor=(0.05, 0.75),\
                                                 bbox_transform=ax.transAxes, borderpad=0.0)
            ax.add_artist(anc_chi2_without_labelbox)

            age_without_labelbox = TextArea(r"$\left<t\right>_M$ = " + "{:.2f}".format(float(10**mass_wht_age/1e9)) + " Gyr", textprops=dict(color='b', size=12))
            anc_age_without_labelbox = AnchoredOffsetbox(loc=2, child=age_without_labelbox, pad=0.0, frameon=False,\
                                                 bbox_to_anchor=(0.05, 0.7),\
                                                 bbox_transform=ax.transAxes, borderpad=0.0)
            ax.add_artist(anc_age_without_labelbox)

        elif comp_count == 1:
            col = 'r'

            with_labelbox = TextArea("With LSF", textprops=dict(color='r', size=12))
            anc_with_labelbox = AnchoredOffsetbox(loc=2, child=with_labelbox, pad=0.0, frameon=False,\
                                                 bbox_to_anchor=(0.3, 0.8),\
                                                 bbox_transform=ax.transAxes, borderpad=0.0)
            ax.add_artist(anc_with_labelbox)

            chi2_with_labelbox = TextArea(r"$\chi^2_{red} = $" + str("{:.4}".format(fitchi2/len(flam))), textprops=dict(color='r', size=12))
            anc_chi2_with_labelbox = AnchoredOffsetbox(loc=2, child=chi2_with_labelbox, pad=0.0, frameon=False,\
                                                 bbox_to_anchor=(0.3, 0.75),\
                                                 bbox_transform=ax.transAxes, borderpad=0.0)
            ax.add_artist(anc_chi2_with_labelbox)

            age_with_labelbox = TextArea(r"$\left<t\right>_M$ = " + "{:.2f}".format(float(10**mass_wht_age/1e9)) + " Gyr", textprops=dict(color='r', size=12))
            anc_age_with_labelbox = AnchoredOffsetbox(loc=2, child=age_with_labelbox, pad=0.0, frameon=False,\
                                                 bbox_to_anchor=(0.3, 0.7),\
                                                 bbox_transform=ax.transAxes, borderpad=0.0)
            ax.add_artist(anc_age_with_labelbox)

        ax.plot(resampling_lam_grid, best_fit_model * bestalpha, '-', color=col)

        ax.set_xlabel(r'$\lambda\,[\AA]$', fontsize=15)
        ax.set_ylabel(r'$\mathrm{f_\lambda\,[erg\,s^{-1}\,cm^{-2}\,\AA}]$', fontsize=15)

        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')
        ax.grid(True)

        print fitage, fitlogtau, mass_wht_age, fitmetal, fittauv, fitchi2

        comp_count += 1
    """
    # get best spectrum in original bc03 fits files.
    fitlogtau = 0.0398
    fittauv = 0.19
    fitage = 9.47712
    tau_string = str(int(fitlogtau*10000))
    tauv_string = str(int(fittauv*100))
    metalfolder = 'm62' + '/'

    cspout = home + '/Documents/GALAXEV_BC03/bc03/src/cspout_new/'
    filename = cspout + metalfolder + 'bc2003_hr_m62_tauV' + tauv_string + '_csp_tau' + tau_string + '_salp.fits'
    fitsfile = fits.open(filename)
    agearr = fitsfile[2].data
    ageindx = np.argmin(abs(agearr - 10**fitage))

    interp_factor_arr = np.arange(0.5, 4.5, 0.5)
    # loop over some interpolated lsf options and plot all of them with their chi-square values 
    for interp_factor in interp_factor_arr:
        bestspec = fitsfile[ageindx+3].data

        interplsf = get_interplsf(current_id, current_redshift, current_field, pa_forlsf, interp_factor)
        bestspec = convolve_fft(bestspec, interplsf)
        bestspec = ct.resample_single(fitsfile[1].data, bestspec, lam_em)

        bestalpha = np.sum(flam_em * bestspec / (ferr**2)) / np.sum(bestspec**2 / ferr**2)

        ax.plot(lam_em, bestspec * bestalpha, '-', label=str(interp_factor))
    #ax.set_xlim(3000, 5000)
    ax.legend(loc=0)

    #fig.savefig(massive_figures_dir + 'lsf_test.eps', dpi=300, bbox_inches='tight')
    plt.show()

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)
