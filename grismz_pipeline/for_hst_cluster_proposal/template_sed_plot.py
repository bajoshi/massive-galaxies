from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from numpy import nansum
#import pysynphot
from functools import reduce

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(massive_galaxies_dir + 'cluster_codes/')
from cluster_do_fitting import get_avg_dlam

def get_klambda(wav):
    # Calzetti law
    # wavlength has to be in microns

    rv = 4.05

    if wav > 0.63:
        klam = 2.659 * (-1.857 + 1.040/wav) + rv

    elif wav < 0.63:
        klam = 2.659 * (-2.156 + 1.509/wav - 0.198/wav**2 + 0.011/wav**3) + rv

    elif wav == 0.63:
        klam = 3.49
        # Since the curves dont exactly meet at 0.63 micron, I've taken the average of the two.
        # They're close though. One gives me klam=3.5 and the other klam=3.48

    return klam

def get_ebv(av):
    return av / 4.05

def get_model_photometry(model_lam_grid, model_flux_arr, filt):

    # first interpolate the transmission curve to the model lam grid
    filt_interp = griddata(points=filt['wav'], values=filt['trans'], xi=model_lam_grid, method='linear')

    # multiply model spectrum to filter curve
    num = nansum(model_flux_arr * filt_interp)
    den = nansum(filt_interp)

    model_flam = num / den

    return model_flam

def get_model_grism_spec(model_lam_grid, model_flux_arr, grism_resamp_grid, grism_curve):

    # Resample the model spectrum to a finer grid (?)

    # Resample the template on to the grism resampling grid
    # Define array to save modified models
    model_grism_spec = np.zeros(len(grism_resamp_grid), dtype=np.float64)

    ### Zeroth element
    lam_step = grism_resamp_grid[1] - grism_resamp_grid[0]
    idx = np.where((model_lam_grid >= grism_resamp_grid[0] - lam_step) & (model_lam_grid < grism_resamp_grid[0] + lam_step))[0]

    model_grism_spec[0] = np.mean(model_flux_arr[idx])

    ### all elements in between
    for i in range(1, len(grism_resamp_grid) - 1):
        idx = np.where((model_lam_grid >= grism_resamp_grid[i-1]) & (model_lam_grid < grism_resamp_grid[i+1]))[0]
        model_grism_spec[i] = np.mean(model_flux_arr[idx])

    ### Last element
    lam_step = grism_resamp_grid[-1] - grism_resamp_grid[-2]
    idx = np.where((model_lam_grid >= grism_resamp_grid[-1] - lam_step) & (model_lam_grid < grism_resamp_grid[-1] + lam_step))[0]
    model_grism_spec[-1] = np.mean(model_flux_arr[idx])

    ### ------- 
    """
    # Fold in grism transmission curve by making sure 
    # that the total flux through the grism curve
    # is the same as the total flux in the resampled spectrum.
    model_grism_flam = get_model_photometry(template, grism_curve)
    model_grism_spec_sum = nansum(model_grism_spec)

    corr_factor = model_grism_flam / model_grism_spec_sum
    model_grism_spec *= corr_factor
    """

    return model_grism_spec

def chop_grism_spec(grism_curve, resampling_lam_grid, grism_spec):

    # Now chop the grism curve to where the transmission is above 10%
    grism_idx = np.where(grism_curve['trans'] >= 0.1)[0]
    low_grism_lim = grism_curve['wav'][grism_idx][0]
    high_grism_lim = grism_curve['wav'][grism_idx][-1]

    low_idx = np.argmin(abs(resampling_lam_grid - low_grism_lim))
    high_idx = np.argmin(abs(resampling_lam_grid - high_grism_lim))

    resampling_lam_grid = resampling_lam_grid[low_idx: high_idx+1]
    grism_spec = grism_spec[low_idx: high_idx+1]

    return resampling_lam_grid, grism_spec

def get_dust_atten_model(model_wav_arr, model_flux_arr, av):
    """
    This function will apply the Calzetti dust extinction law 
    to the given model using the supplied value of Av.

    It assumes that the model it is being given is dust-free.
    It assumes that the model wavelengths it is given are in Angstroms.

    It returns the dust-attenuated flux array at the same wavelengths as before.
    """

    ebv = get_ebv(av)
        
    # Now loop over the dust-free SED and generate a new dust-attenuated SED
    dust_atten_model_flux = np.zeros(len(model_wav_arr), dtype=np.float64)
    for i in range(len(model_wav_arr)):

        current_wav = model_wav_arr[i] / 1e4  # because this has to be in microns

        # The calzetti law is only valid up to 2.2 micron so beyond 
        # 2.2 micron this function just replaces the old values
        if current_wav <= 2.2:
            klam = get_klambda(current_wav)
            alam = klam * ebv

            dust_atten_model_flux[i] = model_flux_arr[i] * 10**(-1 * 0.4 * alam)
        else:
            dust_atten_model_flux[i] = model_flux_arr[i]

    return dust_atten_model_flux

def tau(lam_obs, redshift):
    # lam_obs in angstroms

    aj = np.array((3.6e-3, 1.7e-3, 1.2e-3, 9.3e-4, 8.0e-4, 7.0e-4, 6.2e-4, 5.7e-4, 5.2e-4, 4.8e-4,\
                   4.5e-4, 4.2e-4, 3.9e-4, 3.7e-4, 3.5e-4, 3.3e-4, 3.16e-4))
    upper_level = np.arange(2, 19, 1)

    z = redshift
    tau_line = 0
    tau_c = 0

    for count in range(len(upper_level)):

        i_line = upper_level[count]
        lam_line = 911.7535 / (1 - (1/i_line**2))

        if lam_obs < ((1 + z) * lam_line):
            tau_j = aj[count] * (lam_obs / lam_line)**3.46
        else:
            tau_j = 0

        tau_line += tau_j

    x_c = lam_obs / 911.7535
    x_em = (1 + z)

    if lam_obs < 911.7535 * (1 + z):
        tau_c = (x_c**3 * (x_em**0.46 - x_c**0.46)/4) + \
        (9.4 * x_c**1.5 * (x_em**0.18 - x_c**0.18)) - \
        (0.7 * x_c**3 * (x_c**-1.32 - x_em**-1.32)) - \
        (0.023 * (x_em**1.68 - x_c**1.68))
    else:
        tau_c = 0

    return (tau_line + tau_c)

def plot_template_sed(model, model_name, phot_lam, model_photometry, resampling_lam_grid, model_grism_spec, \
    all_filters, filter_names, grism_name):

    # --------- Plot --------- # 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Labels
    ax.set_ylabel(r'$\mathrm{f_\lambda\ [Arbitrary\ scale]}$', fontsize=15)
    ax.set_xlabel(r'$\mathrm{Wavelength\, [\AA]}$', fontsize=15)

    # Plot all filters
    # need twinx first
    ax1 = ax.twinx()
    for filt in all_filters:
        ax1.plot(filt['wav'], filt['trans'], zorder=4)

    # Twin axis related stuff
    #ax1.set_ylabel(r'$\mathrm{Transmission}$', fontsize=15)
    ax1.set_ylim(0, 2.0)
    ax1.set_yticklabels([])

    # plot template (along with the dust attenuated one if required)
    if grism_name == 'G102':
        # Plot smoothed spectrum if using high-res BC03
        from astropy.convolution import convolve, Gaussian1DKernel
        kernel = Gaussian1DKernel(stddev=5)
        ax.plot(model['wav'], convolve(model['flam_norm'], kernel), color='k', zorder=2)
    else:
        ax.plot(model['wav'], model['flam_norm'], color='k', zorder=2)

    # If you're working with Mrk231 or the DSFG then also show the dust-attenuated SEDs
    if model_name == 'AGN':
        dust_atten_model_flux_av1 = get_dust_atten_model(model['wav'], model['flam_norm'], 1.0)
        dust_atten_model_flux_av5 = get_dust_atten_model(model['wav'], model['flam_norm'], 5.0)

        ax.plot(model['wav'], dust_atten_model_flux_av1, color='blue', zorder=4)
        ax.plot(model['wav'], dust_atten_model_flux_av5, color='maroon', zorder=4)

        wav_arr_for_shading_idx = np.where(model['wav'] <= 2.2e4)[0]
        ax.fill_between(model['wav'][wav_arr_for_shading_idx], \
            model['flam_norm'][wav_arr_for_shading_idx], dust_atten_model_flux_av5[wav_arr_for_shading_idx], \
            color='gray', alpha=0.25)

    if model_name == 'DSFG':
        dust_atten_model_flux = get_dust_atten_model(model['wav'], model['flam_norm'], 2.0)
        ax.plot(model['wav'], dust_atten_model_flux, color='maroon', zorder=4)

        wav_arr_for_shading_idx = np.where(model['wav'] <= 2.2e4)[0]
        ax.fill_between(model['wav'][wav_arr_for_shading_idx], \
            model['flam_norm'][wav_arr_for_shading_idx], dust_atten_model_flux[wav_arr_for_shading_idx], \
            color='gray', alpha=0.25)

    # plot template photometry
    ax.plot(phot_lam, model_photometry, 'o', color='r', markersize=6, zorder=6)

    # Plot simulated grism spectrum
    # Take out NaN values so it connects all the dots
    nonan_idx = np.where(~np.isnan(model_grism_spec))
    resampling_lam_grid_nonan = resampling_lam_grid[nonan_idx]
    model_grism_spec_nonan = model_grism_spec[nonan_idx]
    ax.plot(resampling_lam_grid_nonan, model_grism_spec_nonan, 'o-', markersize=2, color='teal', lw=3.5, zorder=5)

    # Also show the grism only spectrum as an inset plot
    ax_in = inset_axes(ax, width="35%", height="22%", loc=1)
    ax_in.plot(resampling_lam_grid_nonan, model_grism_spec_nonan, 'o-', markersize=2, color='teal', lw=1.5, zorder=5)
    ax_in.minorticks_on()
    ax_in.set_yticklabels([])

    # Show labels for emission lnies
    if grism_name == 'G102':
        if 'Ell' in model_name:
            ax_in.axvline(x=8860, ls='--', color='r', ymin=0.25, ymax=0.6)
            ax_in.text(0.2, 0.23, r'H${\alpha}$', verticalalignment='top', horizontalalignment='left', \
            transform=ax_in.transAxes, color='k', size=10)
            ax_in.text(0.45, 0.95, 'G102 spectrum', verticalalignment='top', horizontalalignment='left', \
            transform=ax_in.transAxes, color='k', size=10)
        elif 'SF' in model_name:
            ax_in.axvline(x=11181, ls='--', color='r', ymin=0.2, ymax=0.8)
            ax_in.text(0.46, 0.53, r'${\rm [OII]}\lambda 3727$', verticalalignment='top', horizontalalignment='left', \
            transform=ax_in.transAxes, color='k', size=10)
            ax_in.text(0.05, 0.95, 'G102 spectrum', verticalalignment='top', horizontalalignment='left', \
            transform=ax_in.transAxes, color='k', size=10)

    # Fix all zorder
    if grism_name == 'G141':
        ax.set_zorder(ax1.get_zorder()+1) # put ax in front of ax1 
        ax.patch.set_visible(False) # hide the 'canvas'

    # Other formatting stuff
    if model_name == 'AGN':
        ax.set_ylim(-0.05, 10.05)
    elif model_name == 'SFG':
        ax.set_ylim(-0.05, 0.5)
    elif model_name == 'DSFG':
        ax.set_ylim(-0.05, 0.82)
    elif 'Ell' in model_name:
        ax.set_ylim(-0.05, 0.42)

    ax.set_xscale('log')

    if (grism_name == 'G102') and ('SF' in model_name):
        ax.set_yscale('log')
        #ax.set_ylim(0.01, 100)

    # ---------- tick labels for the logarithmic axis ---------- #
    ax.minorticks_on()
    # Must be done after setting the scale to log

    if grism_name == 'G141':
        ax.set_xlim(0.3e4, 6e4)
        ax.set_xticks([4000, 1e4, 2e4, 5e4])
        ax.text(0.485, 0.97, 'G141' + '\n' + 'spectrum', verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=13)
    else:
        ax.set_xlim(0.3e4, 10e4)
        ax.set_xticks([4000, 1e4, 2e4, 5e4, 8e4])

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))

    # Horizontal line at 0
    #ax.axhline(y=0.0, ls='--', color='k')

    # Text for info
    # Template name
    ax.text(0.02, 0.98, model_name, verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=17)    

    # Other info 
    if model_name == 'AGN':
        ax.axhline(y=9.25, xmin=0.13, xmax=0.18, color='blue')
        ax.text(0.19, 0.94, r'$A_V = 1.0$', verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=15)
    
        ax.axhline(y=8.75, xmin=0.13, xmax=0.18, color='maroon')
        ax.text(0.19, 0.88, r'$A_V = 5.0$', verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=15)

    if model_name == 'DSFG':

        ax.axhline(y=0.73, xmin=0.03, xmax=0.1, color='k')
        ax.text(0.107, 0.925, r'$\mathrm{no\ dust}$', verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=15)

        ax.axhline(y=0.68, xmin=0.03, xmax=0.1, color='maroon')
        ax.text(0.107, 0.85, r'$A_V = 2.0$', verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=15)

    # Filer names
    ax.text(0.16, 0.1, filter_names[0], verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='#1f77b4', size=12, zorder=10)
    ax.text(0.31, 0.1, filter_names[1], verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='#ff7f0e', size=12, zorder=10)
    ax.text(0.56, 0.1, filter_names[2], verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='#2ca02c', size=12, zorder=10)
    ax.text(0.62, 0.15, filter_names[3], verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='#d62728', size=12, zorder=10)
    ax.text(0.84, 0.1, filter_names[4], verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='#9467bd', size=12, zorder=10)

    #plt.show()
    savename = 'sed_plot_' + model_name.replace(' ', '_') + '.pdf'
    fig.savefig(savename, dpi=300, bbox_inches='tight')

    return None

def do_all_mods_plot(template, template_name, all_filters, filter_names, phot_lam, grism_curve, grism_name):

    if 'DSFG' in template_name:
        template_flux = get_dust_atten_model(template['wav'], template['flam_norm'], 2.0)
    else:
        template_flux = template['flam_norm']

    template_flam_list = []
    for i in range(len(all_filters)):
        template_flam_list.append(get_model_photometry(template['wav'], template_flux, all_filters[i]))

    # Create grism resampling grid
    if grism_name == 'G141':
        # Get an example observed G141 spectrum from 3D-HST
        example_g141_hdu = fits.open('goodsn-16-G141_33780.1D.fits')
        grism_lam_obs = example_g141_hdu[1].data['wave']

        # extend lam_grid to be able to move the lam_grid later 
        avg_dlam = get_avg_dlam(grism_lam_obs)

        grism_low_lim = 10000
        grism_high_lim = 17800

    elif grism_name == 'G102':
        # Get an example observed G102 spectrum from FIGS
        example_g102_hdu = fits.open(figs_dir + 'spc_files/GS1_G102_2.combSPC.fits')
        grism_lam_obs = example_g102_hdu[1].data['LAMBDA']

        # extend lam_grid to be able to move the lam_grid later 
        avg_dlam = get_avg_dlam(grism_lam_obs)

        grism_low_lim = 7600
        grism_high_lim = 13000

    lam_low_to_insert = np.arange(grism_low_lim, grism_lam_obs[0], avg_dlam, dtype=np.float64)
    lam_high_to_append = np.arange(grism_lam_obs[-1] + avg_dlam, grism_high_lim, avg_dlam, dtype=np.float64)

    resampling_lam_grid = np.insert(grism_lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    template_grism_spec = get_model_grism_spec(template['wav'], template_flux, resampling_lam_grid, grism_curve)
    resampling_lam_grid_short, template_grism_spec_short = chop_grism_spec(grism_curve, resampling_lam_grid, template_grism_spec)

    # ---------------------------------- Plot ---------------------------------- #
    plot_template_sed(template, template_name, phot_lam, template_flam_list, \
        resampling_lam_grid_short, template_grism_spec_short, all_filters, filter_names, grism_name)

    return None

def get_template(age, tau, tauv, metallicity, \
    log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, \
    model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap, plotcheck=False):

    print("\n", "Finding closest model to --")
    print("Age [Gyr]:", 10**age / 1e9)
    print("Tau [Gyr]:", tau)
    print("Tau_v:", tauv)
    print("Metallicity [abs. frac.]:", metallicity)

    # First find closest values and then indices corresponding to them
    # It has to be done this way because 
    closest_age_idx = np.argmin(abs(log_age_arr - age))
    closest_tau_idx = np.argmin(abs(tau_gyr_arr - tau))

    # Now get indices
    age_idx = np.where(log_age_arr == log_age_arr[closest_age_idx])[0]
    tau_idx = np.where(tau_gyr_arr == tau_gyr_arr[closest_tau_idx])[0]
    tauv_idx = np.where(tauv_arr == tauv)[0]
    metal_idx = np.where(metal_arr == metallicity)[0]

    model_idx = int(reduce(np.intersect1d, (age_idx, tau_idx, tauv_idx, metal_idx)))

    model_flam = model_comp_spec_withlines_mmap[model_idx]

    chosen_age = 10**log_age_arr[model_idx] / 1e9
    chosen_tau = tau_gyr_arr[model_idx]
    chosen_av = 1.086 * tauv_arr[model_idx]
    chosen_metallicity = metal_arr[model_idx]

    print("Chosen model index:", model_idx)
    print("Chosen model parameters -- ")
    print("Age [Gyr]:", chosen_age)
    print("Tau [Gyr]:", chosen_tau)
    print("A_v:", chosen_av)
    print("Metallicity [abs. frac.]:", chosen_metallicity)

    # Plot to check that you have the correct template
    if plotcheck:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(model_lam_grid_withlines_mmap, model_flam)

        # Limtis and scales
        ax.set_xscale('log')
        ax.set_xlim(500, 100000)

        # Text on plot
        ax.text(0.03, 0.95, r'$\mathrm{Age\,[Gyr]\,=\,}$' + str("{:.2f}".format(chosen_age)), \
            verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=12)
        ax.text(0.03, 0.9, r'$\tau\,\mathrm{[Gyr]}\,=\,$' + str("{:.2f}".format(chosen_tau)), \
            verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=12)
        ax.text(0.03, 0.85, r'$A_V\,=\,$' + str("{:.1f}".format(chosen_av)), \
            verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=12)
        ax.text(0.03, 0.8, r'${\rm Z}\,=\,$' + str("{:.2f}".format(chosen_metallicity)), \
            verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=12)

        plt.show()

    return model_flam

def plot_for_pclusters_Cy28_2020():

    # ---------------------------------- Read in the filters from Seth ---------------------------------- #
    f435w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/F435W_ACS.res', \
        dtype=None, names=['wav', 'trans'])
    f606w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/F606W_ACS.res', \
        dtype=None, names=['wav', 'trans'])
    f814w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/F814W_ACS.txt', \
        dtype=None, names=['wav', 'trans'])
    g141  = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/g141l_tot.txt', \
        dtype=None, names=['wav', 'trans'])
    f140w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f140w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])

    # Spitzer/IRAC channels
    irac1 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac1.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac2 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac2.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac3 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac3.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac4 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac4.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)

    # IRAC wavelengths are in microns # convert to angstroms
    irac1['wav'] *= 1e4
    irac2['wav'] *= 1e4
    irac3['wav'] *= 1e4
    irac4['wav'] *= 1e4

    # Herschel PACS filters
    pacs_blue = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/Herschel_pacs.blue.dat', dtype=None, names=['wav', 'trans'])
    pacs_green = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/Herschel_pacs.green.dat', dtype=None, names=['wav', 'trans'])
    pacs_red = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/Herschel_pacs.red.dat', dtype=None, names=['wav', 'trans'])

    #all_filters = [f435w, f606w, f814w, f140w, irac1, irac2]  # irac3, irac4, pacs_blue, pacs_green, pacs_red]
    #filter_names = ['F435W', 'F606W', 'F814W', 'F140W', 'IRAC_CH1', 'IRAC_CH2']  # 'IRAC_CH3', 'IRAC_CH4', 'PACS_BLUE', 'PACS_GREEN', 'PACS_RED']

    all_filters = [f606w, f814w, f140w, irac1, irac2]
    filter_names = ['F606W', 'F814W', 'F140W', 'IRAC-CH1', 'IRAC-CH2']

    # Scale the filter curves to look like the ones in the handbook:
    # For ACS : http://www.stsci.edu/hst/acs/documents/handbooks/current/c10_ImagingReference01.html
    # For WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/appendixA01.html
    # For Spitzer: 
    # For Herschel: 
    # The numbers here were determined by eye. I simply looked at the curves
    # in the handbook and made sure the peak transmission here matches that
    # in the handbook.
    # Get scaling factors  # the numerator here is my eyeballed peak transmission
    # I eyeballed it for the HST filters. The Spitzer docmentation has the peak trans listed.
    f435w_scalefac = 0.37 / max(f435w['trans'])
    f606w_scalefac = 0.465 / max(f606w['trans'])
    f814w_scalefac = 0.44 / max(f814w['trans'])
    f140w_scalefac = 0.56 / max(f140w['trans'])
    irac1_scalefac = 0.748 / max(irac1['trans'])
    irac2_scalefac = 0.859 / max(irac2['trans'])
    irac3_scalefac = 0.653 / max(irac3['trans'])
    irac4_scalefac = 0.637 / max(irac4['trans'])

    pacs_blue_scalefac = 0.51 / max(pacs_blue['trans'])
    pacs_green_scalefac = 0.57 / max(pacs_green['trans'])
    pacs_red_scalefac = 0.465 / max(pacs_red['trans'])

    f435w['trans'] *= f435w_scalefac
    f606w['trans'] *= f606w_scalefac
    f814w['trans'] *= f814w_scalefac
    f140w['trans'] *= f140w_scalefac
    irac1['trans'] *= irac1_scalefac
    irac2['trans'] *= irac2_scalefac
    irac3['trans'] *= irac3_scalefac
    irac4['trans'] *= irac4_scalefac

    pacs_blue['trans'] *= pacs_blue_scalefac
    pacs_green['trans'] *= pacs_green_scalefac
    pacs_red['trans'] *= pacs_red_scalefac

    # PLot to check
    # Do not delete. This is used to compare with the plots in the handbooks.
    """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for filt in all_filters:
        ax2.plot(filt['wav'], filt['trans'])
    ax2.set_xscale('log')
    plt.show()
    sys.exit(0)
    """

    # Pivot wavelengths
    # From here --
    # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/
    # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
    # Spitzer IRAC channels: http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/6/#_Toc410728283
    # Herschel PACS: http://herschel.esac.esa.int/Docs/PACS/html/ch03s02.html
    #phot_lam = np.array([4328.2, 5921.1, 8057.0, 13923.0, 35500.0, 44930.0]) # 57310.0, 78720.0, 700000.0, 1000000.0, 1600000.0])  # angstroms
    phot_lam = np.array([5921.1, 8057.0, 13923.0, 35500.0, 44930.0])

    # ---------------------------------- Now get the templates ---------------------------------- #
    # Older templates from Brenda
    # Use the newer ones from Maria Polletta
    """
    ell2gyr = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/Ell2Gyr_template.tbl', \
        dtype=None, names=['wav', 'flam_norm'], skip_header=2)
    mrk231 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/Mrk231_template.tbl', \
        dtype=None, names=['wav', 'flam_norm'], skip_header=2)    
    zless = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/zless_template.tbl', \
        dtype=None, names=['wav', 'flam_norm'], skip_header=2)

    all_templates = [ell2gyr, mrk231, zless]
    all_template_names = ['Ell 2 Gyr', 'Mrk 231', 'z less']
    """

    # NEwer templates 
    # AGN, SF galaxy, and sub-mm galaxy
    ell2gyr = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/Ell2Gyr_template.tbl', \
        dtype=None, names=['wav', 'flam_norm'], skip_header=2)
    agn_template = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/QSO1_template.sed', \
        dtype=None, names=['wav', 'flam_norm'], skip_header=3)
    sfg_template = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/M82_starburst_template.sed', \
        dtype=None, names=['wav', 'flam_norm'], skip_header=3)
    smg_template = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/aless_SMG_template.sed', \
        dtype=None, names=['wav', 'flam_norm'], skip_header=3)

    #all_templates = [ell2gyr, agn_template, sfg_template, smg_template]
    #all_template_names = ['Ell 2 Gyr', 'AGN', 'SFG', 'DSFG']

    all_templates = [ell2gyr, smg_template]
    all_template_names = ['Ell 2 Gyr', 'DSFG']

    # Define redshift for templates
    z = 2.6

    # ---------------------------------- Convolve templates with filters ---------------------------------- #
    for count in range(len(all_templates)):
        template = all_templates[count]
        template_name = all_template_names[count]

        # Redshift templates
        template['wav'] *= (1+z)

        dl_tbl = np.genfromtxt(massive_galaxies_dir + 'cluster_codes/dl_lookup_table.txt', dtype=None, names=True)
        z_idx = np.argmin(abs(dl_tbl['z'] - z))
        dl = dl_tbl['dl_cm'][z_idx]  # in cm

        # Convert to realistic fluxes
        # assume a stellar mass 
        stellar_mass = 10**9  # solar masses
        L_sun = 3.84e33  # in erg per sec
        #template['flam_norm'] = (template['flam_norm'] * L_sun * stellar_mass) / (4 * np.pi * dl**2 * (1 + z))
        template['flam_norm'] /= (1+z)

        do_all_mods_plot(template, template_name, all_filters, filter_names, phot_lam, g141, 'G141')

    return None

def plot_for_g165():

    # ---------------------------------- Read in the filters ---------------------------------- #
    f435w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/F435W_ACS.res', \
        dtype=None, names=['wav', 'trans'])
    f606w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/F606W_ACS.res', \
        dtype=None, names=['wav', 'trans'])
    f814w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/F814W_ACS.txt', \
        dtype=None, names=['wav', 'trans'])
    f105w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/wfc3_ir_filters/F105W_IR_throughput.csv', \
        dtype=None, names=['wav', 'trans'], usecols=(1,2), delimiter=',', skip_header=1)
    f110w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/wfc3_ir_filters/F110W_IR_throughput.csv', \
        dtype=None, names=['wav', 'trans'], usecols=(1,2), delimiter=',', skip_header=1)
    f160w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f160w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    lbt_luci_k = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/LBT_LUCIFER.K_ED059.dat', \
        dtype=None, names=['wav', 'trans'])
    mmirs_k = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/mmirs_k.dat', \
        dtype=None, names=['wav', 'trans'], skip_header=1)
    # I've been told (see Chris Willmer's email) that this is proper throughput 
    # for the MMIRS K-band and shouldn't need scaling.

    # Read G102 throughput curve and save to ascii file
    # This has to come from pysynphot # It should not need any scaling # Check with handbook
    # This code block only needs to run once
    """
    g102_pysyn  = pysynphot.ObsBandpass('wfc3,ir,g102')
    g102_wav = g102_pysyn.binset
    g102_trans = g102_pysyn(g102_wav)
    g102_dat = np.array(zip(g102_wav, g102_trans), dtype=[('g102_wav', np.float64), ('g102_trans', np.float64)])
    np.savetxt(massive_galaxies_dir + 'grismz_pipeline/g102_filt_curve.txt', g102_dat, \
        fmt=['%.6f', '%.6f'], header='wav trans')
    """
    # Now read it back from the ascii file to get it into a format consistent with all other curves
    g102 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/g102_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])

    # Spitzer/IRAC channels
    irac1 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac1.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac2 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac2.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac3 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac3.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac4 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac4.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)

    # IRAC wavelengths are in microns # convert to angstroms
    irac1['wav'] *= 1e4
    irac2['wav'] *= 1e4
    irac3['wav'] *= 1e4
    irac4['wav'] *= 1e4

    # mmirs k also in microns
    mmirs_k['wav'] *= 1e4

    all_filters = [f435w, f606w, f814w, f105w, f110w, f160w, mmirs_k, irac1, irac2, irac3, irac4]
    filter_names = ['F435W', 'F606W', 'F814W', 'F105W', 'F110W', 'F160W', 'K', 'IRAC_CH1', 'IRAC_CH2', 'IRAC_CH3', 'IRAC_CH4']

    # Multiply by scale factor to make the curves look like those in the handbook
    # See the comments in the plot_for_pclusters() function above
    f435w_scalefac = 0.37  / max(f435w['trans'])
    f606w_scalefac = 0.465 / max(f606w['trans'])
    f814w_scalefac = 0.44  / max(f814w['trans'])
    f105w_scalefac = 0.515 / max(f105w['trans'])
    f110w_scalefac = 0.56 / max(f110w['trans'])
    f160w_scalefac = 0.555 / max(f160w['trans'])
    lucik_scalefac = 0.91 / max(lbt_luci_k['trans'])
    irac1_scalefac = 0.748 / max(irac1['trans'])
    irac2_scalefac = 0.859 / max(irac2['trans'])
    irac3_scalefac = 0.653 / max(irac3['trans'])
    irac4_scalefac = 0.637 / max(irac4['trans'])

    f435w['trans'] *= f435w_scalefac
    f606w['trans'] *= f606w_scalefac
    f814w['trans'] *= f814w_scalefac
    f105w['trans'] *= f105w_scalefac
    f110w['trans'] *= f110w_scalefac
    f160w['trans'] *= f160w_scalefac
    lbt_luci_k['trans'] *= lucik_scalefac
    irac1['trans'] *= irac1_scalefac
    irac2['trans'] *= irac2_scalefac
    irac3['trans'] *= irac3_scalefac
    irac4['trans'] *= irac4_scalefac

    # Pivot wavelengths
    # From here --
    # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/
    # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
    # Spitzer IRAC channels: http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/6/#_Toc410728283
    phot_lam = np.array([4328.2, 5921.1, 8057.0, 10552.0, 11534.0, 15369.0, 21860.85, \
    35500.0, 44930.0, 57310.0, 78720.0])  # angstroms
    # Using LBT LUCI K pivot wavelength for the MMIRS K filter
    # Need to compute the pivot wav for the MMIRS filter properly.

    # ---------------------------------- Now get the BC03 templates ---------------------------------- #
    """
    1. Elliptical galaxy with modest H-alpha emission at z=0.35
    I'm choosing a solar metallicity template with a 6 Gyr old population with no dust and a tau=1 Gyr.
    2. Starforming galaxy at z=2
    I'm choosing a solar metallicity template with a 50 Myr old population with no dust and a tau=0.2 Gyr.
    """
    
    # -------------------
    # For the purposes of this code the following block to save the templates only needs to be run once.
    # -------------------
    """
    # Generate lists including parameter values for as many templates as needed
    template_metallicities = [0.02, 0.02]
    # Since my CSPs only have solar metallicity and I'm including a tau
    # the chosen SED will have solar metallicity by default.
    template_ages_gyr = [6.0, 0.05]  # in Gyr
    template_tau_gyr = [1.0, 0.2]  # in Gyr
    template_tauv = [0.0, 0.0]  # make sure this is a multiple of 0.2 because that is the grid spacing
    redshifts = [0.35, 2.0]
    template_names = ['E_6gyr_z0p35', 'SF_50Myr_z2']

    # Read in all models and parameters
    model_lam_grid_withlines_mmap = np.load(figs_dir + 'model_lam_grid_withlines.npy', mmap_mode='r')
    model_comp_spec_withlines_mmap = np.load(figs_dir + 'model_comp_spec_withlines.npy', mmap_mode='r')

    log_age_arr = np.load(figs_dir + 'log_age_arr.npy', mmap_mode='r')
    metal_arr = np.load(figs_dir + 'metal_arr.npy', mmap_mode='r')
    tau_gyr_arr = np.load(figs_dir + 'tau_gyr_arr.npy', mmap_mode='r')
    tauv_arr = np.load(figs_dir + 'tauv_arr.npy', mmap_mode='r')

    for i in range(len(template_ages_gyr)):
        current_age = np.log10(template_ages_gyr[i] * 1e9)  # because the saved age parameter is the log(age[yr])
        current_tau = template_tau_gyr[i] # because the saved tau is in Gyr
        current_tauv = template_tauv[i]
        current_metallicity = template_metallicities[i]
        z = redshifts[i]

        template_flam = get_template(current_age, current_tau, current_tauv, current_metallicity, \
            log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, \
            model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap, plotcheck=True)

        # Redshift template and save ascii files to conform with the remaining code structure
        template_wav = model_lam_grid_withlines_mmap * (1+z)
        template_flux = template_flam / (1+z)

        # Normalize just liek the previous ones
        idx5500 = np.argmin(abs(model_lam_grid_withlines_mmap - 5500.0)) 
        template_flux_at5500 = template_flux[idx5500]
        template_flux /= template_flux_at5500

        # Now save
        template_dat = np.array(zip(template_wav, template_flux), dtype=[('template_wav', np.float64), ('template_flux', np.float64)])
        np.savetxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/' + template_names[i] + '.txt', \
            template_dat, fmt=['%.6e', '%.6e'], header='wav flam_norm')
    """

    # ---- Now read the templates back in from the ascii files
    ell6gyr = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/E_6gyr_z0p35.txt', \
        dtype=None, names=True) 
    sf50myr = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/SF_50Myr_z2.txt', \
        dtype=None, names=True)

    all_templates = [ell6gyr, sf50myr]
    all_template_names = ['Ell 6Gyr z=0.35', 'SF 50Myr z=2']

    # ---------------------------------- Convolve templates with filters ---------------------------------- #
    for count in range(len(all_templates)):
        template = all_templates[count]
        template_name = all_template_names[count]
        do_all_mods_plot(template, template_name, all_filters, filter_names, phot_lam, g102, 'G102')

    return None

def plot_for_g165_Cy28_2020():

    # Optical and near-IR filters
    # More to be added # Ask Brenda
    f110w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/wfc3_ir_filters/F110W_IR_throughput.csv', \
        dtype=None, names=['wav', 'trans'], usecols=(1,2), delimiter=',', skip_header=1)
    f160w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f160w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    lbt_luci_k = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/LBT_LUCIFER.K_ED059.dat', \
        dtype=None, names=['wav', 'trans'])

    # Spitzer/IRAC channels
    irac1 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac1.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac2 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac2.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac3 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac3.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac4 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac4.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)

    # IRAC wavelengths are in microns # convert to angstroms
    irac1['wav'] *= 1e4
    irac2['wav'] *= 1e4
    irac3['wav'] *= 1e4
    irac4['wav'] *= 1e4

    # Multiply by scale factor to make the curves look like those in the handbook
    # See the comments in the plot_for_pclusters() function above
    f110w_scalefac = 0.56 / max(f110w['trans'])
    f160w_scalefac = 0.555 / max(f160w['trans'])
    lucik_scalefac = 0.91 / max(lbt_luci_k['trans'])
    irac1_scalefac = 0.748 / max(irac1['trans'])
    irac2_scalefac = 0.859 / max(irac2['trans'])
    irac3_scalefac = 0.653 / max(irac3['trans'])
    irac4_scalefac = 0.637 / max(irac4['trans'])

    f110w['trans'] *= f110w_scalefac
    f160w['trans'] *= f160w_scalefac
    lbt_luci_k['trans'] *= lucik_scalefac
    irac1['trans'] *= irac1_scalefac
    irac2['trans'] *= irac2_scalefac
    irac3['trans'] *= irac3_scalefac
    irac4['trans'] *= irac4_scalefac

    all_filters = [f110w, f160w, lbt_luci_k, irac1, irac2, irac3, irac4]
    filter_names = ['F110W', 'F160W', 'K', 'IRAC_CH1', 'IRAC_CH2', 'IRAC_CH3', 'IRAC_CH4']

    # Pivot wavelengths
    # From here --
    # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/
    # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
    # Spitzer IRAC channels: http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/6/#_Toc410728283
    phot_lam = np.array([11534.0, 15369.0, 21860.85, \
    35500.0, 44930.0, 57310.0, 78720.0])  # angstroms

    # ---- Now read the templates back in from the ascii files
    template = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/SF_50Myr_z2.txt', \
        dtype=None, names=True)

    # ---- Now plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Labels
    ax.set_ylabel(r'$\mathrm{f_\lambda\ [Arbitrary\ scale]}$', fontsize=15)
    ax.set_xlabel(r'$\mathrm{Wavelength\, [\AA]}$', fontsize=15)

    # Get dust attenuated model and plot
    dust_atten_model_flux = get_dust_atten_model(template['wav'], template['flam_norm'], 2.0)

    # ---- Now get photometry for dust-attenuated model
    template_flam_list = []
    for i in range(len(all_filters)):
        template_flam_list.append(get_model_photometry(template['wav'], dust_atten_model_flux, all_filters[i]))

    ax.plot(template['wav'], template['flam_norm'], color='blue', zorder=2)
    ax.plot(template['wav'], dust_atten_model_flux, color='firebrick', zorder=4)

    wav_arr_for_shading_idx = np.where(template['wav'] <= 2.2e4)[0]
    ax.fill_between(template['wav'][wav_arr_for_shading_idx], \
        template['flam_norm'][wav_arr_for_shading_idx], dust_atten_model_flux[wav_arr_for_shading_idx], \
        color='gray', alpha=0.25)

    # plot template photometry
    ax.plot(phot_lam, template_flam_list, 'o', color='k', markersize=6, zorder=6)

    # Plot all filters
    # need twinx first
    ax1 = ax.twinx()
    for filt in all_filters:
        ax1.plot(filt['wav'], filt['trans'], zorder=4)

    # Twin axis related stuff
    ax1.set_ylabel(r'$\mathrm{Transmission}$', fontsize=15)
    ax1.set_ylim(0, 1.2)

    # Set scales and limits
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 600)

    # ---------- tick labels for the logarithmic axis ---------- #
    ax.minorticks_on()

    ax.set_xlim(0.3e4, 10e4)
    ax.set_xticks([4000, 1e4, 2e4, 5e4, 8e4])

    fig.savefig("sed_plot_SF_z2_Cy28.pdf", dpi=300, bbox_inches='tight')

    return None

def main():

    plot_for_pclusters_Cy28_2020()
    #plot_for_g165_Cy28_2020()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)