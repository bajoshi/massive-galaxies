from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from numpy import nansum

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

sys.path.append(massive_galaxies_dir + 'codes/')
import refine_redshifts_dn4000 as old_ref

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

def get_model_photometry(template, filt):

    # Get template flux and wav grid
    model_lam_grid = template['wav']
    model_flux_arr = template['flam_norm']

    # first interpolate the transmission curve to the model lam grid
    filt_interp = griddata(points=filt['wav'], values=filt['trans'], xi=model_lam_grid, method='linear')

    # multiply model spectrum to filter curve
    num = nansum(model_flux_arr * filt_interp)
    den = nansum(filt_interp)

    model_flam = num / den

    return model_flam

def get_model_grism_spec(template, grism_resamp_grid, grism_curve):

    # Get template flux and wav grid
    model_lam_grid = template['wav']
    model_flux_arr = template['flam_norm']

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

def chop_grism_spec(g141, resampling_lam_grid, grism_spec):

    # Now chop the grism curve to where the transmission is above 10%
    grism_idx = np.where(g141['trans'] >= 0.1)[0]
    low_grism_lim = g141['wav'][grism_idx][0]
    high_grism_lim = g141['wav'][grism_idx][-1]

    low_idx = np.argmin(abs(resampling_lam_grid - low_grism_lim))
    high_idx = np.argmin(abs(resampling_lam_grid - high_grism_lim))

    resampling_lam_grid = resampling_lam_grid[low_idx: high_idx+1]
    grism_spec = grism_spec[low_idx: high_idx+1]

    return resampling_lam_grid, grism_spec

def get_dust_atten_model(model, av):

    ebv = get_ebv(av)
        
    # Now loop over the dust-free SED and generate a new dust-attenuated SED
    dust_atten_model_flux = np.zeros(len(model), dtype=np.float64)
    for i in range(len(model)):

        current_wav = model['wav'][i] / 1e4  # because this has to be in microns

        klam = get_klambda(current_wav)
        alam = klam * ebv

        dust_atten_model_flux[i] = model['flam_norm'][i] * 10**(-1 * 0.4 * alam)

    return dust_atten_model_flux

def plot_template_sed(model, model_name, phot_lam, model_photometry, resampling_lam_grid, model_grism_spec, all_filters, filter_names):

    # If you're working with Mrk231 then also show the dust-attenuated SEDs
    if model_name == 'Mrk 231':
        dust_atten_model_flux_av2 = get_dust_atten_model(model, 2.0)
        dust_atten_model_flux_av5 = get_dust_atten_model(model, 5.0)

    # --------- Plot --------- # 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Labels
    ax.set_ylabel(r'$\mathrm{f_\lambda\ [Arbitrary\ scale]}$', fontsize=15)
    ax.set_xlabel(r'$\mathrm{Wavelength\, [\AA]}$', fontsize=15)

    # plot template (along with the dust attenuated one if required)
    ax.plot(model['wav'], model['flam_norm'], color='k')
    if model_name == 'Mrk 231':
        ax.plot(model['wav'], dust_atten_model_flux_av2, color='orange')
        ax.plot(model['wav'], dust_atten_model_flux_av5, color='maroon')

    # plot template photometry
    ax.plot(phot_lam, model_photometry, 'o', color='r', markersize=6, zorder=6)

    # Plot simulated grism spectrum
    ax.plot(resampling_lam_grid, model_grism_spec, 'o-', markersize=2, color='seagreen', lw=4, zorder=5)

    # Plot all filters
    # need twinx first
    ax1 = ax.twinx()
    for filt in all_filters:
        ax1.plot(filt['wav'], filt['trans'], zorder=2)

    # Twin axis related stuff
    ax1.set_ylabel(r'$\mathrm{Transmission}$', fontsize=15)
    ax1.set_ylim(0, 1.01)

    # Other formatting stuff
    ax.set_xlim(0.3e4, 500e4)
    ax.set_ylim(-0.05, 1.41)

    ax.set_xscale('log')

    ax.minorticks_on()

    # ---------- tick labels for the logarithmic axis ---------- #
    # Must be done after setting the scale to log
    ax.set_xticks([4000, 10000, 20000, 50000, 80000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Horizontal line at 0
    #ax.axhline(y=0.0, ls='--', color='k')

    # Text for info
    # Template name
    ax.text(0.75, 0.95, model_name, verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=18)

    # Other info 
    
    # Filer names
    """
    ax.text(0.024, 0.15, filter_names[0], verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=12, zorder=10)
    ax.text(0.14, 0.05, filter_names[1], verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=12, zorder=10)
    ax.text(0.25, 0.15, filter_names[2], verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=12, zorder=10)
    ax.text(0.38, 0.05, filter_names[3], verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=12, zorder=10)
    """

    #savename = 'sed_plot_' + model_name.replace(' ', '_') + '.pdf'
    #fig.savefig(savename, dpi=300, bbox_inches='tight')

    plt.show()

    return None

def do_all_mods_plot(template, template_name, all_filters, filter_names, phot_lam, g141):

    template_flam_list = []
    for i in range(len(all_filters)):
        template_flam_list.append(get_model_photometry(template, all_filters[i]))

    # Create grism resampling grid
    # Get an example observed G141 spectrum from 3D-HST
    example_g141_hdu = fits.open('goodsn-16-G141_33780.1D.fits')
    grism_lam_obs = example_g141_hdu[1].data['wave']

    # extend lam_grid to be able to move the lam_grid later 
    avg_dlam = old_ref.get_avg_dlam(grism_lam_obs)

    lam_low_to_insert = np.arange(10000, grism_lam_obs[0], avg_dlam, dtype=np.float64)
    lam_high_to_append = np.arange(grism_lam_obs[-1] + avg_dlam, 17800, avg_dlam, dtype=np.float64)

    resampling_lam_grid = np.insert(grism_lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    template_grism_spec = get_model_grism_spec(template, resampling_lam_grid, g141)

    resampling_lam_grid_short, template_grism_spec_short = chop_grism_spec(g141, resampling_lam_grid, template_grism_spec)

    # ---------------------------------- Plot ---------------------------------- #
    plot_template_sed(template, template_name, phot_lam, template_flam_list, \
        resampling_lam_grid_short, template_grism_spec_short, all_filters, filter_names)

    return None

def main():

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

    all_filters = [f435w, f606w, f814w, f140w, irac1, irac2, irac3, irac4]
    filter_names = ['F435W', 'F606W', 'F814W', 'F140W', 'IRAC_CH1', 'IRAC_CH2', 'IRAC_CH3', 'IRAC_CH4']

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

    f435w['trans'] *= f435w_scalefac
    f606w['trans'] *= f606w_scalefac
    f814w['trans'] *= f814w_scalefac
    f140w['trans'] *= f140w_scalefac
    irac1['trans'] *= irac1_scalefac
    irac2['trans'] *= irac2_scalefac
    irac3['trans'] *= irac3_scalefac
    irac4['trans'] *= irac4_scalefac

    # PLot to check
    # Do not delete. This is used to compare with the plots in the handbooks.
    """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for filt in all_filters:
        ax2.plot(filt['wav'], filt['trans'])
    plt.show()
    sys.exit(0)
    """

    # Pivot wavelengths
    # From here --
    # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/
    # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
    # Spitzer IRAC channels: http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/6/#_Toc410728283
    phot_lam = np.array([4328.2, 5921.1, 8057.0, 13923.0, 35500.0, 44930.0, 57310.0, 78720.0])  # angstroms

    # ---------------------------------- Now get the templates ---------------------------------- #
    ell2gyr = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/Ell2Gyr_template.tbl', \
        dtype=None, names=['wav', 'flam_norm'], skip_header=2)
    mrk231 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/Mrk231_template.tbl', \
        dtype=None, names=['wav', 'flam_norm'], skip_header=2)    
    zless = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/zless_template.tbl', \
        dtype=None, names=['wav', 'flam_norm'], skip_header=2)

    all_templates = [ell2gyr, mrk231, zless]
    all_template_names = ['Ell 2 Gyr', 'Mrk 231', 'z less']

    # ---------------------------------- Convolve templates with filters ---------------------------------- #
    for count in range(len(all_templates)):
        template = all_templates[count]
        template_name = all_template_names[count]
        do_all_mods_plot(template, template_name, all_filters, filter_names, phot_lam, g141)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)