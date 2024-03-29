from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.integrate import simps
from scipy.signal import fftconvolve

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"
threedhst_datadir = home + "/Desktop/3dhst_data/"
massive_figures_dir = figs_dir + 'massive-galaxies-figures/'
savedir_spz = massive_figures_dir + 'spz_run_jan2019/'
savedir_photoz = massive_figures_dir + 'photoz_run_jan2019/'
savedir_grismz = massive_figures_dir + 'grismz_run_jan2019/'

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import refine_redshifts_dn4000 as old_ref
import fullfitting_grism_broadband_emlines as ff
import photoz
import new_refine_grismz_gridsearch_parallel as ngp
import model_mods as mm
import dn4000_catalog as dc
import mocksim_results as mr
import spz_photoz_grismz_comparison as comp

speed_of_light = 299792458e10  # angstroms per second

def makefig():
    # ---------- Create figure ---------- #
    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[:8,:])
    ax2 = fig.add_subplot(gs[8:,:])

    # ---------- labels ---------- #
    ax1.set_ylabel(r'$\mathrm{f_\lambda\ [erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]}$', fontsize=14)
    ax2.set_xlabel(r'$\mathrm{Wavelength\, [\AA]}$', fontsize=15)
    ax2.set_ylabel(r'$\mathrm{\frac{f^{obs}_\lambda\ - \alpha f^{mod}_\lambda}{\sigma_{f_{\lambda}^{\rm obs}}}}$', fontsize=17)

    return fig, ax1, ax2

def make_pz_labels(ax):

    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('p(z)', fontsize=12)

    ax.xaxis.set_label_coords(0.8,-0.05)

    return ax

def plot_photoz_fit(phot_lam_obs, phot_flam_obs, phot_ferr_obs, model_lam_grid, \
    best_fit_model_fullres, all_filt_flam_bestmodel, bestalpha, \
    obj_id, obj_field, specz, zp, low_z_lim, upper_z_lim, chi2, age, tau, av, netsig, d4000, savedir):

    # Make figure and place on grid
    fig, ax1, ax2 = makefig()

    # ---------- plot data, model, and residual ---------- #
    # plot full res model but you'll have to redshift it
    ax1.plot(model_lam_grid * (1+zp), bestalpha*best_fit_model_fullres / (1+zp), color='dimgrey', alpha=0.2)
    # plot model photometry
    ax1.scatter(phot_lam_obs, bestalpha*all_filt_flam_bestmodel, s=20, color='lightseagreen', zorder=20)

    # ----- plot data
    ax1.errorbar(phot_lam_obs, phot_flam_obs, yerr=phot_ferr_obs, fmt='.', color='crimson', markeredgecolor='crimson', \
        capsize=2, markersize=10.0, elinewidth=2.0)

    # ----- Residuals
    # For the photometry
    resid_fit_phot = (phot_flam_obs - bestalpha*all_filt_flam_bestmodel) / phot_ferr_obs
    ax2.scatter(phot_lam_obs, resid_fit_phot, s=4, color='k')
    ax2.axhline(y=0, ls='--', color='k')

    # ---------- limits ---------- #
    max_y_obs = np.max(phot_flam_obs)
    min_y_obs = np.min(phot_flam_obs)

    max_ylim = 1.25 * max_y_obs
    min_ylim = 0.2 * min_y_obs

    ax1.set_ylim(min_ylim, max_ylim)

    ax1.set_xlim(3000, 85000)
    ax2.set_xlim(3000, 85000)

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # ---------- tick labels for the logarithmic axis ---------- #
    ax2.set_xticks([4000, 10000, 20000, 50000, 80000])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # ---------- minor ticks ---------- #
    ax1.minorticks_on()
    ax2.minorticks_on()

    # ---------- text for info ---------- #
    ax1.text(0.71, 0.61, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    low_zerr = zp - low_z_lim
    high_zerr = upper_z_lim - zp

    ax1.text(0.71, 0.55, \
    r'$\mathrm{z_{p;best} = }$' + "{:.4}".format(zp) + \
    r'$\substack{+$' + "{:.3}".format(high_zerr) + r'$\\ -$' + "{:.3}".format(low_zerr) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.71, 0.48, r'$\mathrm{z_{spec} = }$' + "{:.4}".format(specz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    #ax1.text(0.71, 0.22, r'$\mathrm{z_{p;wt} = }$' + "{:.4}".format(zp), \
    #verticalalignment='top', horizontalalignment='left', \
    #transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.71, 0.43, r'$\mathrm{\chi^2 = }$' + "{:.3}".format(chi2), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.71, 0.36, r'$\mathrm{NetSig = }$' + mr.convert_to_sci_not(netsig), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=8)
    ax1.text(0.71, 0.3, r'$\mathrm{D4000 = }$' + "{:.3}".format(d4000), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=8)


    ax1.text(0.37, 0.18,'log(Age[yr]) = ' + "{:.4}".format(age), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.37, 0.12, r'$\tau$' + '[Gyr] = ' + "{:.3}".format(tau), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    if av < 0:
        av = -99.0

    ax1.text(0.37, 0.06, r'$\mathrm{A_V}$' + ' = ' + "{:.3}".format(av), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    # ---------- Plot p(z) curve in an inset figure ---------- #
    # Solution for inset came from SO:
    # https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.62, 0.72, 0.3, 0.2]
    ax3 = fig.add_axes([left, bottom, width, height])

    # Read in p(z) curve. It should be in the same folder where all these figures are being saved.
    pz = np.load(savedir + obj_field + '_' + str(obj_id) + '_photoz_pz.npy')
    zarr = np.load(savedir + obj_field + '_' + str(obj_id) + '_photoz_z_arr.npy')

    ax3 = make_pz_labels(ax3)

    ax3.plot(zarr, pz)
    ax3.axvline(x=specz, ls='--', color='darkred')
    ax3.minorticks_on()

    # ---------- Save figure ---------- #
    fig.savefig(savedir + obj_field + '_' + str(obj_id) + '_photoz_fit.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

def plot_spz_fit(grism_lam_obs, grism_flam_obs, grism_ferr_obs, phot_lam_obs, phot_flam_obs, phot_ferr_obs, \
    model_lam_grid, best_fit_model_fullres, best_fit_model_in_objlamgrid, all_filt_flam_bestmodel, bestalpha, \
    obj_id, obj_field, specz, low_zp_lim, upper_zp_lim, zp, low_zspz_lim, upper_zspz_lim, zspz, \
    low_zg_lim, upper_zg_lim, zg, chi2, age, tau, av, netsig, d4000, savedir):

    # Make figure and place on grid
    fig, ax1, ax2 = makefig()

    # ---------- plot data, model, and residual ---------- #
    # plot full res model but you'll have to redshift it
    ax1.plot(model_lam_grid * (1+zspz), bestalpha*best_fit_model_fullres / (1+zspz), color='dimgrey', alpha=0.2)

    # ----- plot data
    ax1.plot(grism_lam_obs, grism_flam_obs, 'o-', color='k', markersize=2, lw=2, zorder=10)
    ax1.fill_between(grism_lam_obs, grism_flam_obs + grism_ferr_obs, grism_flam_obs - grism_ferr_obs, color='gray', zorder=10)

    ax1.errorbar(phot_lam_obs, phot_flam_obs, yerr=phot_ferr_obs, fmt='.', color='crimson', markeredgecolor='crimson', \
        capsize=2, markersize=10.0, elinewidth=2.0)

    # ----- plot best fit model
    ax1.plot(grism_lam_obs, bestalpha*best_fit_model_in_objlamgrid, ls='-', lw=1.2, color='lightseagreen', zorder=20)
    ax1.scatter(phot_lam_obs, bestalpha*all_filt_flam_bestmodel, s=20, color='lightseagreen', zorder=20)

    # ----- Residuals
    # For the grism points
    resid_fit_grism = (grism_flam_obs - bestalpha*best_fit_model_in_objlamgrid) / grism_ferr_obs

    # Now plot
    ax2.scatter(grism_lam_obs, resid_fit_grism, s=4, color='k')
    ax2.axhline(y=0, ls='--', color='k')

    # For the photometry
    resid_fit_phot = (phot_flam_obs - bestalpha*all_filt_flam_bestmodel) / phot_ferr_obs
    ax2.scatter(phot_lam_obs, resid_fit_phot, s=4, color='k')

    # ---------- limits ---------- #
    max_y_obs = np.max(np.concatenate((grism_flam_obs, phot_flam_obs)))
    min_y_obs = np.min(np.concatenate((grism_flam_obs, phot_flam_obs)))

    max_ylim = 1.25 * max_y_obs
    min_ylim = 0.2 * min_y_obs

    ax1.set_ylim(min_ylim, max_ylim)
    ax2.set_ylim(-14, 14)

    ax1.set_xlim(3000, 85000)
    ax2.set_xlim(3000, 85000)

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # ---------- tick labels for the logarithmic axis ---------- #
    ax2.set_xticks([4000, 10000, 20000, 50000, 80000])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # ---------- minor ticks ---------- #
    ax1.minorticks_on()
    ax2.minorticks_on()

    # ---------- text for info ---------- #
    ax1.text(0.71, 0.61, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    low_zspz_err = zspz - low_zspz_lim
    high_zspz_err = upper_zspz_lim - zspz

    low_zp_err = zp - low_zp_lim
    high_zp_err = upper_zp_lim - zp

    low_zg_err = zg - low_zg_lim
    high_zg_err = upper_zg_lim - zg

    ax1.text(0.71, 0.55, \
    r'$\mathrm{z_{spz;best} = }$' + "{:.3}".format(zspz) + \
    r'$\substack{+$' + "{:.2f}".format(high_zspz_err) + r'$\\ -$' + "{:.2f}".format(low_zspz_err) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.71, 0.48, r'$\mathrm{z_{spec} = }$' + "{:.3f}".format(specz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.71, 0.42, \
    r'$\mathrm{z_{p;best} = }$' + "{:.3}".format(zp) + \
    r'$\substack{+$' + "{:.2f}".format(high_zp_err) + r'$\\ -$' + "{:.2f}".format(low_zp_err) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.71, 0.36, r'$\mathrm{\chi^2 = }$' + "{:.3}".format(chi2), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.71, 0.3, r'$\mathrm{NetSig = }$' + mr.convert_to_sci_not(netsig), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.71, 0.24, r'$\mathrm{D4000 = }$' + "{:.3}".format(d4000), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)


    ax1.text(0.17, 0.18,'log(Age[yr]) = ' + "{:.4}".format(age), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.17, 0.12, r'$\tau$' + '[Gyr] = ' + "{:.3}".format(tau), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    if av < 0:
        av = -99.0

    ax1.text(0.17, 0.06, r'$\mathrm{A_V}$' + ' = ' + "{:.3}".format(av), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    # ---------- Plot p(z) curve in an inset figure ---------- #
    # Solution for inset came from SO:
    # https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.62, 0.72, 0.3, 0.2]
    ax3 = fig.add_axes([left, bottom, width, height])

    # Read in p(z) curve. It should be in the same folder where all these figures are being saved.
    pz = np.load(savedir + obj_field + '_' + str(obj_id) + '_spz_pz.npy')
    zarr = np.load(savedir + obj_field + '_' + str(obj_id) + '_spz_z_arr.npy')

    ax3 = make_pz_labels(ax3)

    ax3.plot(zarr, pz)
    ax3.axvline(x=specz, ls='--', color='darkred')
    ax3.minorticks_on()

    # ---------- Save figure ---------- #
    fig.savefig(savedir + obj_field + '_' + str(obj_id) + '_spz_fit.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

def plot_grismz_fit(grism_lam_obs, grism_flam_obs, grism_ferr_obs, \
    model_lam_grid, best_fit_model_fullres, best_fit_model_in_objlamgrid, bestalpha, \
    obj_id, obj_field, specz, low_zp_lim, upper_zp_lim, zp, low_zg_lim, upper_zg_lim, zg, \
    chi2, age, tau, av, netsig, d4000, savedir):

    # Make figure and place on grid
    fig, ax1, ax2 = makefig()

    # ---------- plot data, model, and residual ---------- #
    # plot full res model but you'll have to redshift it
    ax1.plot(model_lam_grid * (1+zg), bestalpha*best_fit_model_fullres / (1+zg), color='dimgrey', alpha=0.2)

    # ----- plot data
    ax1.plot(grism_lam_obs, grism_flam_obs, 'o-', color='k', markersize=2, lw=2, zorder=10)
    ax1.fill_between(grism_lam_obs, grism_flam_obs + grism_ferr_obs, grism_flam_obs - grism_ferr_obs, color='gray', zorder=10)

    # ----- plot best fit model
    ax1.plot(grism_lam_obs, bestalpha*best_fit_model_in_objlamgrid, ls='-', lw=1.2, color='lightseagreen', zorder=20)

    # ----- Residuals
    # For the grism points
    resid_fit_grism = (grism_flam_obs - bestalpha*best_fit_model_in_objlamgrid) / grism_ferr_obs

    # Now plot
    ax2.scatter(grism_lam_obs, resid_fit_grism, s=4, color='k')
    ax2.axhline(y=0, ls='--', color='k')

    # ---------- limits ---------- #
    max_y_obs = np.max(grism_flam_obs)
    min_y_obs = np.min(grism_flam_obs)

    max_ylim = 1.25 * max_y_obs
    min_ylim = 0.2 * min_y_obs

    ax1.set_ylim(min_ylim, max_ylim)

    ax1.set_xlim(5800, 9700)
    ax2.set_xlim(5800, 9700)

    # ---------- minor ticks ---------- #
    ax1.minorticks_on()
    ax2.minorticks_on()

    # ---------- text for info ---------- #
    ax1.text(0.71, 0.61, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    low_zg_err = zg - low_zg_lim
    high_zg_err = upper_zg_lim - zg

    low_zp_err = zp - low_zp_lim
    high_zp_err = upper_zp_lim - zp

    ax1.text(0.71, 0.55, \
    r'$\mathrm{z_{grism;best} = }$' + "{:.4}".format(zg) + \
    r'$\substack{+$' + "{:.3}".format(high_zg_err) + r'$\\ -$' + "{:.3}".format(low_zg_err) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.71, 0.48, r'$\mathrm{z_{spec} = }$' + "{:.4}".format(specz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.71, 0.43, \
    r'$\mathrm{z_{p;best} = }$' + "{:.4}".format(zp) + \
    r'$\substack{+$' + "{:.3}".format(high_zp_err) + r'$\\ -$' + "{:.3}".format(low_zp_err) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.71, 0.36, r'$\mathrm{\chi^2 = }$' + "{:.3}".format(chi2), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    ax1.text(0.71, 0.3, r'$\mathrm{NetSig = }$' + mr.convert_to_sci_not(netsig), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.71, 0.24, r'$\mathrm{D4000 = }$' + "{:.3}".format(d4000), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)


    ax1.text(0.37, 0.18,'log(Age[yr]) = ' + "{:.4}".format(age), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)
    ax1.text(0.37, 0.12, r'$\tau$' + '[Gyr] = ' + "{:.3}".format(tau), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    if av < 0:
        av = -99.0

    ax1.text(0.37, 0.06, r'$\mathrm{A_V}$' + ' = ' + "{:.3}".format(av), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=13)

    # ---------- Plot p(z) curve in an inset figure ---------- #
    # Solution for inset came from SO:
    # https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.62, 0.72, 0.3, 0.2]
    ax3 = fig.add_axes([left, bottom, width, height])

    # Read in p(z) curve. It should be in the same folder where all these figures are being saved.
    pz = np.load(savedir + obj_field + '_' + str(obj_id) + '_zg_pz.npy')
    zarr = np.load(savedir + obj_field + '_' + str(obj_id) + '_zg_z_arr.npy')

    ax3 = make_pz_labels(ax3)

    ax3.plot(zarr, pz)
    ax3.axvline(x=specz, ls='--', color='darkred')
    ax3.minorticks_on()

    # ---------- Save figure ---------- #
    fig.savefig(savedir + obj_field + '_' + str(obj_id) + '_grismz_fit.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

def get_best_fit_model_spz(resampling_lam_grid, resampling_lam_grid_length, model_lam_grid, model_comp_spec, \
    grism_lam_obs, redshift, model_idx, phot_fin_idx, all_model_flam, lsf, total_models):

    # ------------ Get best fit model at grism resolution ------------ #
    # First do the convolution with the LSF
    model_comp_spec_lsfconv = np.zeros(model_comp_spec.shape)
    for i in range(total_models):
        model_comp_spec_lsfconv[i] = fftconvolve(model_comp_spec[i], lsf, mode = 'same')

    # chop model to get the part within objects lam obs grid
    model_lam_grid_indx_low = np.argmin(abs(resampling_lam_grid - grism_lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(abs(resampling_lam_grid - grism_lam_obs[-1]))

    # force types before passing to cython code
    #lam_obs = lam_obs.astype(np.float64)
    #model_lam_grid = model_lam_grid.astype(np.float64)
    #model_comp_spec = model_comp_spec.astype(np.float64)
    #resampling_lam_grid = resampling_lam_grid.astype(np.float64)
    #total_models = int(total_models)
    #lsf = lsf.astype(np.float64)

    # Will have to redo the model modifications at the new found redshift
    model_comp_spec_modified = \
    mm.redshift_and_resample_fast(model_comp_spec_lsfconv, redshift, total_models, \
        model_lam_grid, resampling_lam_grid, resampling_lam_grid_length)
    print "Model mods done (only for plotting purposes) at the new SPZ:", redshift

    best_fit_model_in_objlamgrid = model_comp_spec_modified[model_idx, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # ------------ Get photomtery for model ------------- #
    all_filt_flam_bestmodel = get_photometry_best_fit_model(redshift, model_idx, phot_fin_idx, all_model_flam, total_models)

    # ------------ Get best fit model at full resolution ------------ #
    best_fit_model_fullres = model_comp_spec[model_idx]

    return best_fit_model_in_objlamgrid, all_filt_flam_bestmodel, best_fit_model_fullres

def get_best_fit_model_grismz(resampling_lam_grid, resampling_lam_grid_length, model_lam_grid, model_comp_spec, \
    grism_lam_obs, redshift, model_idx, lsf, total_models):

    # ------------ Get best fit model at grism resolution ------------ #
    # First do the convolution with the LSF
    model_comp_spec_lsfconv = np.zeros(model_comp_spec.shape)
    for i in range(total_models):
        model_comp_spec_lsfconv[i] = fftconvolve(model_comp_spec[i], lsf, mode = 'same')

    # chop model to get the part within objects lam obs grid
    model_lam_grid_indx_low = np.argmin(abs(resampling_lam_grid - grism_lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(abs(resampling_lam_grid - grism_lam_obs[-1]))

    # force types before passing to cython code
    #lam_obs = lam_obs.astype(np.float64)
    #model_lam_grid = model_lam_grid.astype(np.float64)
    #model_comp_spec = model_comp_spec.astype(np.float64)
    #resampling_lam_grid = resampling_lam_grid.astype(np.float64)
    #total_models = int(total_models)
    #lsf = lsf.astype(np.float64)

    # Will have to redo the model modifications at the new found redshift
    model_comp_spec_modified = \
    mm.redshift_and_resample_fast(model_comp_spec_lsfconv, redshift, total_models, \
        model_lam_grid, resampling_lam_grid, resampling_lam_grid_length)
    print "Model mods done (only for plotting purposes) at the new grism-z:", redshift

    best_fit_model_in_objlamgrid = model_comp_spec_modified[model_idx, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # ------------ Get best fit model at full resolution ------------ #
    best_fit_model_fullres = model_comp_spec[model_idx]

    return best_fit_model_in_objlamgrid, best_fit_model_fullres

def get_photometry_best_fit_model(redshift, model_idx, phot_fin_idx, all_model_flam, total_models):
    # All you need from here is the photometry for the best fit model
    # ------------ Get photomtery for model ------------- #
    # The model mags were computed on a finer redshift grid
    # So make sure to get the z_idx correct
    z_model_arr = np.arange(0.0, 6.0, 0.005)

    z_idx = np.where(z_model_arr == redshift)[0]

    # and because for some reason it does not find matches 
    # in the model redshift array (sometimes), I need this check here.
    if not z_idx.size:
        z_idx = np.argmin(abs(z_model_arr - redshift))

    all_filt_flam_model = all_model_flam[:, z_idx, :]
    all_filt_flam_model = all_filt_flam_model[phot_fin_idx, :]
    all_filt_flam_model = all_filt_flam_model.reshape(len(phot_fin_idx), total_models)

    all_filt_flam_model_t = all_filt_flam_model.T
    all_filt_flam_bestmodel = all_filt_flam_model_t[model_idx]

    return all_filt_flam_bestmodel

def get_zpeak_and_zerr(obj_id, obj_field, z_minchi2, redshift_type):

    if redshift_type == 'photo-z':
        results_dir = savedir_photoz
        redshift_str = 'photoz'
    elif redshift_type == 'grism-z':
        results_dir = savedir_grismz
        redshift_str = 'zg'
    elif redshift_type == 'spz':
        results_dir = savedir_spz
        redshift_str = 'spz'

    pz = np.load(results_dir + str(obj_field) + '_' + str(obj_id) + '_' + redshift_str + '_pz.npy')    
    zarr = np.load(results_dir + str(obj_field) + '_' + str(obj_id) + '_' + redshift_str + '_z_arr.npy')
    z_peak = zarr[np.argmax(pz)]

    # Get errors
    zerr_low, zerr_high = comp.get_z_errors(zarr, pz, z_minchi2)

    return z_peak, zerr_low, zerr_high

def main():
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # ------------------------------- Give galaxy data here ------------------------------- #
    # Only needs the ID and the field
    # And flag to modify LSF
    current_id = 126769
    current_field = 'GOODS-S'
    modify_lsf = True

    # ------------------------------- Get correct directories ------------------------------- #
    figs_data_dir = '/Volumes/Bhavins_backup/bc03_models_npy_spectra/'
    threedhst_datadir = "/Volumes/Bhavins_backup/3dhst_data/"
    cspout = "/Volumes/Bhavins_backup/bc03_models_npy_spectra/cspout_2016updated_galaxev/"
    # This is if working on the laptop. 
    # Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(figs_data_dir):
        figs_data_dir = figs_dir  # this path only exists on firstlight
        threedhst_datadir = home + "/Desktop/3dhst_data/"  # this path only exists on firstlight
        cspout = home + '/Documents/galaxev_bc03_2016update/bc03/src/cspout_2016updated_galaxev/'
        if not os.path.isdir(figs_data_dir):
            print "Model files not found. Exiting..."
            sys.exit(0)

    # ------------------------------ Get models ------------------------------ #
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample.txt', dtype=None, names=True)

    # ------------------------------ Get models ------------------------------ #
    # read in entire model set
    # To see how these arrays were created check the code:
    # $HOME/Desktop/test-codes/shared_memory_multiprocessing/shmem_parallel_proc.py
    # This part will fail if the arrays dont already exist.
    total_models = 37761 # get_total_extensions(bc03_all_spec_hdulist)

    log_age_arr = np.load(figs_data_dir + 'log_age_arr.npy', mmap_mode='r')
    metal_arr = np.load(figs_data_dir + 'metal_arr.npy', mmap_mode='r')
    nlyc_arr = np.load(figs_data_dir + 'nlyc_arr.npy', mmap_mode='r')
    tau_gyr_arr = np.load(figs_data_dir + 'tau_gyr_arr.npy', mmap_mode='r')
    tauv_arr = np.load(figs_data_dir + 'tauv_arr.npy', mmap_mode='r')
    ub_col_arr = np.load(figs_data_dir + 'ub_col_arr.npy', mmap_mode='r')
    bv_col_arr = np.load(figs_data_dir + 'bv_col_arr.npy', mmap_mode='r')
    vj_col_arr = np.load(figs_data_dir + 'vj_col_arr.npy', mmap_mode='r')
    ms_arr = np.load(figs_data_dir + 'ms_arr.npy', mmap_mode='r')
    mgal_arr = np.load(figs_data_dir + 'mgal_arr.npy', mmap_mode='r')

    model_lam_grid_withlines = np.load(figs_data_dir + 'model_lam_grid_withlines.npy', mmap_mode='r')
    model_comp_spec_withlines = np.load(figs_data_dir + 'model_comp_spec_withlines.npy', mmap_mode='r')

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

    # ---------------------------------- Read in look-up tables for model mags ------------------------------------- #
    # Using the look-up table now since it should be much faster
    # Again check the code --
    # $HOME/Desktop/test-codes/shared_memory_multiprocessing/shmem_parallel_proc.py
    # to see how this was created
    # This part will fail if the array does not already exist.
    all_model_flam = np.load(figs_data_dir + 'all_model_flam.npy', mmap_mode='r')

    # ------------------------------- Read in photometry catalogs ------------------------------- #
    # GOODS-N from 3DHST
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
    'f_U', 'e_U', 'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
    'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), skip_header=3)

    # Read in Vega spectrum and get it in the appropriate forms
    vega = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/' + 'vega_reference.dat', dtype=None, \
        names=['wav', 'flam'], skip_header=7)

    vega_lam = vega['wav']
    vega_spec_flam = vega['flam']
    vega_nu = speed_of_light / vega_lam
    vega_spec_fnu = vega_lam**2 * vega_spec_flam / speed_of_light

    # ------------------------------- Prep for getting photometry ------------------------------- #
    # Assign catalogs 
    if current_field == 'GOODS-N':
        phot_cat_3dhst = goodsn_phot_cat_3dhst
    elif current_field == 'GOODS-S':
        phot_cat_3dhst = goodss_phot_cat_3dhst

    final_sample_idx = int(np.where((final_sample['pearsid'] == current_id) & (final_sample['field'] == current_field))[0])
    current_specz = final_sample['zspec'][final_sample_idx]

    # Get RA, DEC
    current_ra = final_sample['ra'][final_sample_idx]
    current_dec = final_sample['dec'][final_sample_idx]

    print "At ID", current_id, "in", current_field, "with specz:", current_specz

    # ------------------------------- Get grism data and then match with photometry ------------------------------- #
    grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)

    if return_code == 0:
        print current_id, current_field
        print "Return code should not have been 0. Exiting."
        sys.exit(0)

    threed_ra = phot_cat_3dhst['ra']
    threed_dec = phot_cat_3dhst['dec']

    # Now match
    ra_lim = 0.3/3600  # arcseconds in degrees
    dec_lim = 0.3/3600
    threed_phot_idx = np.where((threed_ra >= current_ra - ra_lim) & (threed_ra <= current_ra + ra_lim) & \
        (threed_dec >= current_dec - dec_lim) & (threed_dec <= current_dec + dec_lim))[0]

    """
    If there are multiple matches with the photometry catalog 
    within 0.5 arseconds then choose the closest one.
    """
    if len(threed_phot_idx) > 1:
        print "Multiple matches found in Photmetry catalog. Choosing the closest one."

        ra_two = current_ra
        dec_two = current_dec

        dist_list = []
        for v in range(len(threed_phot_idx)):

            ra_one = threed_ra[threed_phot_idx][v]
            dec_one = threed_dec[threed_phot_idx][v]

            dist = np.arccos(np.cos(dec_one*np.pi/180) * np.cos(dec_two*np.pi/180) * np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + 
                np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
            dist_list.append(dist)

        dist_list = np.asarray(dist_list)
        dist_idx = np.argmin(dist_list)
        threed_phot_idx = threed_phot_idx[dist_idx]

    elif len(threed_phot_idx) == 0:
        print "Match not found in Photmetry catalog. Exiting."
        sys.exit(0)

    # ------------------------------- Get photometric fluxes and their errors ------------------------------- #
    flam_f435w = ff.get_flam('F435W', phot_cat_3dhst['f_F435W'][threed_phot_idx])
    flam_f606w = ff.get_flam('F606W', phot_cat_3dhst['f_F606W'][threed_phot_idx])
    flam_f775w = ff.get_flam('F775W', phot_cat_3dhst['f_F775W'][threed_phot_idx])
    flam_f850lp = ff.get_flam('F850LP', phot_cat_3dhst['f_F850LP'][threed_phot_idx])
    flam_f125w = ff.get_flam('F125W', phot_cat_3dhst['f_F125W'][threed_phot_idx])
    flam_f140w = ff.get_flam('F140W', phot_cat_3dhst['f_F140W'][threed_phot_idx])
    flam_f160w = ff.get_flam('F160W', phot_cat_3dhst['f_F160W'][threed_phot_idx])

    flam_U = ff.get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['f_U'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac1 = ff.get_flam_nonhst('irac1', phot_cat_3dhst['f_IRAC1'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac2 = ff.get_flam_nonhst('irac2', phot_cat_3dhst['f_IRAC2'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac3 = ff.get_flam_nonhst('irac3', phot_cat_3dhst['f_IRAC3'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac4 = ff.get_flam_nonhst('irac4', phot_cat_3dhst['f_IRAC4'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

    ferr_f435w = ff.get_flam('F435W', phot_cat_3dhst['e_F435W'][threed_phot_idx])
    ferr_f606w = ff.get_flam('F606W', phot_cat_3dhst['e_F606W'][threed_phot_idx])
    ferr_f775w = ff.get_flam('F775W', phot_cat_3dhst['e_F775W'][threed_phot_idx])
    ferr_f850lp = ff.get_flam('F850LP', phot_cat_3dhst['e_F850LP'][threed_phot_idx])
    ferr_f125w = ff.get_flam('F125W', phot_cat_3dhst['e_F125W'][threed_phot_idx])
    ferr_f140w = ff.get_flam('F140W', phot_cat_3dhst['e_F140W'][threed_phot_idx])
    ferr_f160w = ff.get_flam('F160W', phot_cat_3dhst['e_F160W'][threed_phot_idx])

    ferr_U = ff.get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['e_U'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac1 = ff.get_flam_nonhst('irac1', phot_cat_3dhst['e_IRAC1'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac2 = ff.get_flam_nonhst('irac2', phot_cat_3dhst['e_IRAC2'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac3 = ff.get_flam_nonhst('irac3', phot_cat_3dhst['e_IRAC3'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac4 = ff.get_flam_nonhst('irac4', phot_cat_3dhst['e_IRAC4'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

    # ------------------------------- Apply aperture correction ------------------------------- #
    # First interpolate the given filter curve on to the wavelength frid of the grism data
    # You only need the F775W filter here since you're only using this filter to get the 
    # aperture correction factor.
    f775w_filt_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f775w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    f775w_trans_interp = griddata(points=f775w_filt_curve['wav'], values=f775w_filt_curve['trans'], \
        xi=grism_lam_obs, method='linear')

    # multiply grism spectrum to filter curve
    num = 0
    den = 0
    for w in range(len(grism_flam_obs)):
        num += grism_flam_obs[w] * f775w_trans_interp[w]
        den += f775w_trans_interp[w]

    avg_f775w_flam_grism = num / den
    aper_corr_factor = flam_f775w / avg_f775w_flam_grism
    print "Aperture correction factor:", "{:.3}".format(aper_corr_factor)

    grism_flam_obs *= aper_corr_factor  # applying factor

    # ------------------------------- Make unified photometry arrays ------------------------------- #
    phot_fluxes_arr = np.array([flam_U, flam_f435w, flam_f606w, flam_f775w, flam_f850lp, flam_f125w, flam_f140w, flam_f160w,
        flam_irac1, flam_irac2, flam_irac3, flam_irac4])
    phot_errors_arr = np.array([ferr_U, ferr_f435w, ferr_f606w, ferr_f775w, ferr_f850lp, ferr_f125w, ferr_f140w, ferr_f160w,
        ferr_irac1, ferr_irac2, ferr_irac3, ferr_irac4])

    # Pivot wavelengths
    # From here --
    # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/#keywords
    # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
    # KPNO/MOSAIC U-band: https://www.noao.edu/kpno/mosaic/filters/k1001.html
    # Spitzer IRAC channels: http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/6/#_Toc410728283
    phot_lam = np.array([3582.0, 4328.2, 5921.1, 7692.4, 9033.1, 12486.0, 13923.0, 15369.0, 
    35500.0, 44930.0, 57310.0, 78720.0])  # angstroms

    # ------------------------------ Now start fitting ------------------------------ #
    # --------- Force dtype for cython code --------- #
    # Apparently this (i.e. for flam_obs and ferr_obs) has  
    # to be done to avoid an obscure error from parallel in joblib --
    # AttributeError: 'numpy.ndarray' object has no attribute 'offset'
    grism_lam_obs = grism_lam_obs.astype(np.float64)
    grism_flam_obs = grism_flam_obs.astype(np.float64)
    grism_ferr_obs = grism_ferr_obs.astype(np.float64)

    phot_lam = phot_lam.astype(np.float64)
    phot_fluxes_arr = phot_fluxes_arr.astype(np.float64)
    phot_errors_arr = phot_errors_arr.astype(np.float64)

    # Read in LSF
    if current_field == 'GOODS-N':
        lsf_filename = lsfdir + "north_lsfs/" + "n" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"
    elif current_field == 'GOODS-S':
        lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"

    # read in LSF file
    lsf = np.genfromtxt(lsf_filename)
    lsf = lsf.astype(np.float64)  # Force dtype for cython code

    # -------- Stetch the LSF ------- #
    if modify_lsf:
        # Stretch LSF instead of broadening
        lsf_length = len(lsf)
        x_arr = np.arange(lsf_length)
        num_interppoints = int(1.118 * lsf_length)
        stretched_lsf_arr = np.linspace(0, lsf_length, num_interppoints, endpoint=False)
        stretched_lsf = griddata(points=x_arr, values=lsf, xi=stretched_lsf_arr, method='linear')

        # Make sure that the new LSF does not have NaN values in ti
        stretched_lsf = stretched_lsf[np.isfinite(stretched_lsf)]

        # Area under stretched LSF should be 1.0
        current_area = simps(stretched_lsf)
        stretched_lsf *= (1/current_area)

        lsf_to_use = stretched_lsf

    # ------- Make new resampling grid ------- # 
    # extend lam_grid to be able to move the lam_grid later 
    avg_dlam = old_ref.get_avg_dlam(grism_lam_obs)

    lam_low_to_insert = np.arange(6000, grism_lam_obs[0], avg_dlam, dtype=np.float64)
    lam_high_to_append = np.arange(grism_lam_obs[-1] + avg_dlam, 10000, avg_dlam, dtype=np.float64)

    resampling_lam_grid = np.insert(grism_lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    # ------- Finite photometry values ------- # 
    # Make sure that the photometry arrays all have finite values
    # If any vlues are NaN then throw them out
    phot_fluxes_finite_idx = np.where(np.isfinite(phot_fluxes_arr))[0]
    phot_errors_finite_idx = np.where(np.isfinite(phot_errors_arr))[0]

    phot_fin_idx = reduce(np.intersect1d, (phot_fluxes_finite_idx, phot_errors_finite_idx))

    phot_fluxes_arr = phot_fluxes_arr[phot_fin_idx]
    phot_errors_arr = phot_errors_arr[phot_fin_idx]
    phot_lam = phot_lam[phot_fin_idx]

    # ------------------------------ Read in results from Agave ------------------------------ #
    fitting_results_dir = home + '/Documents/Papers/spz_paper/example_fits_data/'

    results_filename = fitting_results_dir + 'redshift_fitting_results_' + current_field + '_' + str(current_id) + '.txt'
    res = np.genfromtxt(results_filename, dtype=None, names=True, skip_header=1)

    # Get all results from the file
    # ----------- Photometric redshift ----------- #
    zp_minchi2 = float(res['zp_minchi2'])
    zp = float(res['zp'])
    zp_zerr_low = float(res['zp_zerr_low'])
    zp_zerr_up = float(res['zp_zerr_up'])
    zp_min_chi2 = float(res['zp_min_chi2'])
    zp_bestalpha = float(res['zp_bestalpha'])
    zp_model_idx = int(res['zp_model_idx'])
    zp_age = float(res['zp_age'])
    zp_tau = float(res['zp_tau'])
    zp_av = float(res['zp_av'])

    # ----------- SpectroPhotometric redshift ----------- #
    zspz_minchi2 = float(res['zspz_minchi2'])
    zspz = float(res['zspz'])
    zspz_zerr_low = float(res['zspz_zerr_low'])
    zspz_zerr_up = float(res['zspz_zerr_up'])
    zspz_min_chi2 = float(res['zspz_min_chi2'])
    zspz_bestalpha = float(res['zspz_bestalpha'])
    zspz_model_idx = int(res['zspz_model_idx'])
    zspz_age = float(res['zspz_age'])
    zspz_tau = float(res['zspz_tau'])
    zspz_av = float(res['zspz_av'])

    # ----------- Grism redshift ----------- #
    zg_minchi2 = float(res['zg_minchi2'])
    zg = float(res['zg'])
    zg_zerr_low = float(res['zg_zerr_low'])
    zg_zerr_up = float(res['zg_zerr_up'])
    zg_min_chi2 = float(res['zg_min_chi2'])
    zg_bestalpha = float(res['zg_bestalpha'])
    zg_model_idx = int(res['zg_model_idx'])
    zg_age = float(res['zg_age'])
    zg_tau = float(res['zg_tau'])
    zg_av = float(res['zg_av'])

    # Leave the following block commented out.
    # This is what I used previously where it actually does all the 
    # fitting again before doing the plot.
    """
    # ------------- Call fitting function for photo-z ------------- #
    print "\n", "Computing Photo-z now."
    
    zp_minchi2, zp, zp_zerr_low, zp_zerr_up, zp_min_chi2, zp_bestalpha, zp_model_idx, zp_age, zp_tau, zp_av = \
    photoz.do_photoz_fitting_lookup(phot_fluxes_arr, phot_errors_arr, phot_lam, \
        model_lam_grid_withlines, total_models, model_comp_spec_withlines, start,\
        current_id, current_field, all_model_flam, phot_fin_idx, current_specz, savedir_photoz, \
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr)
    
    # ------------- Call fitting function for SPZ ------------- #
    print "\n", "Photo-z done. Moving on to SPZ computation now."
    
    zspz_minchi2, zspz, zspz_zerr_low, zspz_zerr_up, zspz_min_chi2, zspz_bestalpha, zspz_model_idx, zspz_age, zspz_tau, zspz_av = \
    ff.do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_fluxes_arr, phot_errors_arr, phot_lam, \
        lsf_to_use, resampling_lam_grid, len(resampling_lam_grid), all_model_flam, phot_fin_idx, \
        model_lam_grid_withlines, total_models, model_comp_spec_withlines, start, current_id, current_field, current_specz, zp, \
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, \
        use_broadband=True, single_galaxy=False, for_loop_method='parallel')
    
    # ------------- Call fitting function for grism-z ------------- #
    # Essentially just calls the same function as above but switches off broadband for the fit
    print "\n", "SPZ done. Moving on to Grism-z computation now."
        
    zg_minchi2, zg, zg_zerr_low, zg_zerr_up, zg_min_chi2, zg_bestalpha, zg_model_idx, zg_age, zg_tau, zg_av = \
    ff.do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_fluxes_arr, phot_errors_arr, phot_lam, \
        lsf_to_use, resampling_lam_grid, len(resampling_lam_grid), all_model_flam, phot_fin_idx, \
        model_lam_grid_withlines, total_models, model_comp_spec_withlines, start, current_id, current_field, current_specz, zp, \
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, \
        use_broadband=False, single_galaxy=False, for_loop_method='parallel')
    """

    # ------------- Get z and errors ------------- #
    # Only works if you've used the full redshift grid i.e., np.arange(0.3, 1.5, 0.01)
    # Make sure you do not confuse yourself with this function
    # It returns the z at the peak of the p(z) curve but you should
    # be using the z_minchi2 and the errors it returns are based on the z_minchi2
    zp_peak, zp_zerr_low, zp_zerr_up = get_zpeak_and_zerr(current_id, current_field, zp_minchi2, redshift_type='photo-z')
    zg_peak, zg_zerr_low, zg_zerr_up = get_zpeak_and_zerr(current_id, current_field, zg_minchi2, redshift_type='grism-z')
    zspz_peak, zspz_zerr_low, zspz_zerr_up = get_zpeak_and_zerr(current_id, current_field, zspz_minchi2, redshift_type='spz')

    # ------------- Print results------------- #
    print "\n", "Results:"
    print "Ground-based spectroscopic redshift:", current_specz

    print "\n", "Photometric redshift from min chi2:", "{:.3f}".format(zp_minchi2)
    print "Photometric redshift from peak of p(z) curve:", zp_peak
    print "Weighted photometric redshift:", "{:.3f}".format(zp)

    print "\n", "Grism redshift from min chi2:", "{:.3f}".format(zg_minchi2)
    print "Grism redshift from peak of p(z) curve:", zg_peak
    print "Weighted Grism redshift:", "{:.3f}".format(zg)

    print "\n", "SPZ from min chi2:", "{:.3f}".format(zspz_minchi2)
    print "SPZ from peak of p(z) curve:", zspz_peak
    print "Weighted SPZ:", "{:.3f}".format(zspz)

    print "\n", "Time taken up to now--", str("{:.2f}".format(time.time() - start)), "seconds."
    print "Moving on to plotting fitting results."

    # ------------------------------- First get D4000. Only need to put D4000 info on plot. ------------------------------- # 
    # Get d4000 at SPZ
    lam_em = grism_lam_obs / (1 + current_specz)
    flam_em = grism_flam_obs * (1 + current_specz)
    ferr_em = grism_ferr_obs * (1 + current_specz)

    d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

    # ------------------------------- Get best fit model for plotting ------------------------------- #
    # Will have to do this at the photo-z and SPZ separtely otherwise the plots will not look right
    # ------------ Get best fit model for photo-z ------------ #
    zp_best_fit_model_fullres = model_comp_spec_withlines[zp_model_idx]
    zp_all_filt_flam_bestmodel = get_photometry_best_fit_model(zp_minchi2, zp_model_idx, phot_fin_idx, all_model_flam, total_models)

    # ------------ Get best fit model for SPZ ------------ #
    zspz_best_fit_model_in_objlamgrid, zspz_all_filt_flam_bestmodel, zspz_best_fit_model_fullres = \
    get_best_fit_model_spz(resampling_lam_grid, len(resampling_lam_grid), model_lam_grid_withlines, model_comp_spec_withlines, \
        grism_lam_obs, zspz_minchi2, zspz_model_idx, phot_fin_idx, all_model_flam, lsf_to_use, total_models)

    print "\n", "Chi2 checking for SPZ ---"
    print "Min Chi2 from fitting code:", zspz_min_chi2

    chi2_sum = 0
    for i in range(len(phot_lam)):
    	current_term = ((phot_fluxes_arr[i] - zspz_bestalpha*zp_all_filt_flam_bestmodel[i]) / phot_errors_arr[i])**2
        chi2_sum += current_term
    for j in range(len(grism_flam_obs)):
    	current_term = ((grism_flam_obs[j] - zspz_bestalpha*zspz_best_fit_model_in_objlamgrid[j]) / grism_ferr_obs[j])**2
        chi2_sum += current_term

    print "Explicit chi2:", chi2_sum
    dof = len(grism_lam_obs) + len(phot_lam) - 1
    print "Explicit reduced chi2:", chi2_sum / dof

    # ------------ Get best fit model for grism-z ------------ #
    zg_best_fit_model_in_objlamgrid, zg_best_fit_model_fullres = \
    get_best_fit_model_grismz(resampling_lam_grid, len(resampling_lam_grid), model_lam_grid_withlines, model_comp_spec_withlines, \
        grism_lam_obs, zg_minchi2, zg_model_idx, lsf_to_use, total_models)

    print "\n", "Chi2 checking for grism-z ---"
    print "Min Chi2 from fitting code:", zg_min_chi2

    chi2_sum = 0
    for i in range(len(grism_flam_obs)):
        current_term = ((grism_flam_obs[i] - zg_bestalpha*zg_best_fit_model_in_objlamgrid[i]) / (grism_ferr_obs[i]))**2
        chi2_sum += current_term

    print "Explicit chi2:", chi2_sum
    dof = len(grism_lam_obs) - 1
    print "Explicit reduced chi2:", chi2_sum / dof

    # ------------------------------- Plotting based on results from the above two codes ------------------------------- #
    """
    plot_photoz_fit(phot_lam, phot_fluxes_arr, phot_errors_arr, model_lam_grid_withlines, \
    zp_best_fit_model_fullres, zp_all_filt_flam_bestmodel, zp_bestalpha, \
    current_id, current_field, current_specz, zp, zp_zerr_low, zp_zerr_up, zp_min_chi2, \
    zp_age, zp_tau, zp_av, netsig_chosen, d4000, fitting_results_dir)
    """

    plot_spz_fit(grism_lam_obs, grism_flam_obs, grism_ferr_obs, phot_lam, phot_fluxes_arr, phot_errors_arr, \
    model_lam_grid_withlines, zspz_best_fit_model_fullres, zspz_best_fit_model_in_objlamgrid, zspz_all_filt_flam_bestmodel, zspz_bestalpha, \
    current_id, current_field, current_specz, zp_zerr_low, zp_zerr_up, zp, zspz_zerr_low, zspz_zerr_up, zspz, \
    zg_zerr_low, zg_zerr_up, zg, zspz_min_chi2, zspz_age, zspz_tau, zspz_av, netsig_chosen, d4000, fitting_results_dir)

    """
    plot_grismz_fit(grism_lam_obs, grism_flam_obs, grism_ferr_obs, \
    model_lam_grid_withlines, zg_best_fit_model_fullres, zg_best_fit_model_in_objlamgrid, zg_bestalpha, \
    current_id, current_field, current_specz, zp_zerr_low, zp_zerr_up, zp, zg_zerr_low, zg_zerr_up, zg, \
    zg_min_chi2, zg_age, zg_tau, zg_av, netsig_chosen, d4000, fitting_results_dir)
    """

    # Total time taken
    print "\n", "All done."
    print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds.", "\n"

    return None

if __name__ == '__main__':
    main()
    sys.exit()
