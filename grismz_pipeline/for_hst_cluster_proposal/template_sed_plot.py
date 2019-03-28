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

def plot_template_sed(model, phot_lam, model_photometry, resampling_lam_grid, model_grism_spec, all_filters):

    # ---------
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot template
    ax.plot(model['wav'], model['flam_norm'], color='lightgray')

    # plot template photometry
    ax.plot(phot_lam, model_photometry, 'o', color='r', markersize=6)

    # Plot simulated grism spectrum
    ax.plot(resampling_lam_grid, model_grism_spec, color='seagreen', lw=2)

    # Plot all filters
    # need twinx first
    ax1 = ax.twinx()
    for filt in all_filters:
        ax1.plot(filt['wav'], filt['trans'])

    ax1.set_ylim(0, 1.2)

    ax.set_xlim(3000, 85000)
    ax.set_xscale('log')

    plt.show()

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

    # ---------------------------------- Convolve templates with filters ---------------------------------- #
    ell2gyr_flam_list = []
    for i in range(len(all_filters)):
        ell2gyr_flam_list.append(get_model_photometry(ell2gyr, all_filters[i]))

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

    ell2gyr_grism_spec = get_model_grism_spec(ell2gyr, resampling_lam_grid, g141)

    # Now chop the grism curve to where the transmission is above 10%
    grism_idx = np.where(g141['trans'] >= 0.1)[0]
    low_grism_lim = g141['wav'][grism_idx][0]
    high_grism_lim = g141['wav'][grism_idx][-1]

    low_idx = np.argmin(abs(resampling_lam_grid - low_grism_lim))
    high_idx = np.argmin(abs(resampling_lam_grid - high_grism_lim))

    resampling_lam_grid = resampling_lam_grid[low_idx: high_idx+1]
    ell2gyr_grism_spec = ell2gyr_grism_spec[low_idx: high_idx+1]

    # ---------------------------------- Plot ---------------------------------- #
    plot_template_sed(ell2gyr, phot_lam, ell2gyr_flam_list, resampling_lam_grid, ell2gyr_grism_spec, all_filters)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)