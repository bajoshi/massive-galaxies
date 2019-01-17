from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.integrate import simps

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"
threedhst_datadir = home + "/Desktop/3dhst_data/"
massive_figures_dir = figs_dir + 'massive-galaxies-figures/'
savedir = massive_figures_dir + 'single_galaxy_comparison/'  # Required to save p(z) curve

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

speed_of_light = 299792458e10  # angstroms per second

def makefig():
    # ---------- Create figure ---------- #
    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[:8,:])
    ax2 = fig.add_subplot(gs[8:,:])

    # ---------- labels ---------- #
    ax1.set_ylabel(r'$\mathrm{f_\lambda\ [erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]}$')
    ax2.set_xlabel(r'$\mathrm{Wavelength\, [\AA]}$')
    ax2.set_ylabel(r'$\mathrm{\frac{f^{obs}_\lambda\ - f^{mod}_\lambda}{f^{obs;err}_\lambda}}$')

    return fig, ax1, ax2

def plot_photoz_fit(phot_lam_obs, phot_flam_obs, phot_ferr_obs, model_lam_grid, \
    best_fit_model_fullres, all_filt_flam_bestmodel, bestalpha, \
    obj_id, obj_field, specz, zp, zp_minchi2, low_z_lim, upper_z_lim, chi2, age, tau, av, netsig, d4000, savedir):

    # Make figure and place on grid
    fig, ax1, ax2 = makefig()

    # ---------- plot data, model, and residual ---------- #
    # plot full res model but you'll have to redshift it
    ax1.plot(model_lam_grid * (1+zp), bestalpha*best_fit_model_fullres / (1+zp), color='mediumblue', alpha=0.3)
    # plot model photometry
    ax1.scatter(phot_lam_obs, bestalpha*all_filt_flam_bestmodel, s=20, color='indianred', zorder=20)

    # ----- plot data
    ax1.errorbar(phot_lam_obs, phot_flam_obs, yerr=phot_ferr_obs, fmt='.', color='midnightblue', markeredgecolor='midnightblue', \
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
    min_ylim = 0.75 * min_y_obs

    max_ylim = 1.25 * max_y_obs
    min_ylim = 0.75 * min_y_obs

    ax1.set_ylim(min_ylim, max_ylim)

    ax1.set_xlim(3000, 80000)
    ax2.set_xlim(3000, 80000)

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # ---------- minor ticks ---------- #
    ax1.minorticks_on()
    ax2.minorticks_on()

    # ---------- text for info ---------- #
    ax1.text(0.75, 0.45, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    low_zerr = zp_minchi2 - low_z_lim
    high_zerr = upper_z_lim - zp_minchi2

    ax1.text(0.75, 0.35, \
    r'$\mathrm{z_{p;min\,\chi^2}\, =\, }$' + "{:.4}".format(zp_minchi2) + \
    r'$\substack{+$' + "{:.3}".format(high_zerr) + r'$\\ -$' + "{:.3}".format(low_zerr) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.27, r'$\mathrm{z_{spec}\, =\, }$' + "{:.4}".format(specz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.22, r'$\mathrm{z_{p;wt}\, =\, }$' + "{:.4}".format(zp), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.75, 0.17, r'$\mathrm{\chi^2\, =\, }$' + "{:.3}".format(chi2), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.75, 0.12, r'$\mathrm{NetSig\, =\, }$' + mr.convert_to_sci_not(netsig), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=8)
    ax1.text(0.75, 0.07, r'$\mathrm{D4000(from\ z_{spz})\, =\, }$' + "{:.3}".format(d4000), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=8)


    ax1.text(0.47, 0.3,'log(Age[yr]) = ' + "{:.4}".format(age), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.47, 0.25, r'$\tau$' + '[Gyr] = ' + "{:.3}".format(tau), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    if av < 0:
        av = -99.0

    ax1.text(0.47, 0.2, r'$\mathrm{A_V}$' + ' = ' + "{:.3}".format(av), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    # ---------- Plot p(z) curve in an inset figure ---------- #
    # Solution for inset came from SO:
    # https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.1, 0.6, 0.3, 0.2]
    ax3 = fig.add_axes([left, bottom, width, height])

    # Read in p(z) curve. It should be in the same folder where all these figures are being saved.
    pz = np.load(savedir + obj_field + '_' + str(obj_id) + '_photoz_pz.npy')
    zarr = np.load(savedir + obj_field + '_' + str(obj_id) + '_photoz_z_arr.npy')

    ax3.plot(zarr, pz)
    ax3.axvline(x=specz, ls='--', color='darkred')
    ax3.minorticks_on()

    # ---------- Save figure ---------- #
    fig.savefig(savedir + obj_field + '_' + str(obj_id) + '_photoz_fit.png', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

def plot_spz_fit(grism_lam_obs, grism_flam_obs, grism_ferr_obs, phot_lam_obs, phot_flam_obs, phot_ferr_obs, \
    model_lam_grid, best_fit_model_fullres, best_fit_model_in_objlamgrid, all_filt_flam_bestmodel, bestalpha, \
    obj_id, obj_field, specz, zp, zg_minchi2, low_z_lim, upper_z_lim, zspz, chi2, age, tau, av, netsig, d4000, savedir):

    # Make figure and place on grid
    fig, ax1, ax2 = makefig()

    # ---------- plot data, model, and residual ---------- #
    # plot full res model but you'll have to redshift it
    ax1.plot(model_lam_grid * (1+zspz), bestalpha*best_fit_model_fullres / (1+zspz), color='mediumblue', alpha=0.3)

    # ----- plot data
    ax1.plot(grism_lam_obs, grism_flam_obs, 'o-', color='k', markersize=2, zorder=10)
    ax1.fill_between(grism_lam_obs, grism_flam_obs + grism_ferr_obs, grism_flam_obs - grism_ferr_obs, color='lightgray', zorder=10)

    ax1.errorbar(phot_lam_obs, phot_flam_obs, yerr=phot_ferr_obs, fmt='.', color='midnightblue', markeredgecolor='midnightblue', \
        capsize=2, markersize=10.0, elinewidth=2.0)

    # ----- plot best fit model
    ax1.plot(grism_lam_obs, bestalpha*best_fit_model_in_objlamgrid, ls='-', color='indianred', zorder=20)
    ax1.scatter(phot_lam_obs, bestalpha*all_filt_flam_bestmodel, s=20, color='indianred', zorder=20)

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
    min_ylim = 0.75 * min_y_obs

    max_ylim = 1.25 * max_y_obs
    min_ylim = 0.75 * min_y_obs

    ax1.set_ylim(min_ylim, max_ylim)

    ax1.set_xlim(3000, 80000)
    ax2.set_xlim(3000, 80000)

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # ---------- minor ticks ---------- #
    ax1.minorticks_on()
    ax2.minorticks_on()

    # ---------- text for info ---------- #
    ax1.text(0.75, 0.45, obj_field + ' ' + str(obj_id), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.75, 0.4, r'$\mathrm{z_{SPZ}\, =\, }$' + "{:.4}".format(zspz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    low_zerr = zg_minchi2 - low_z_lim
    high_zerr = upper_z_lim - zg_minchi2

    ax1.text(0.75, 0.35, \
    r'$\mathrm{z_{g;min\,\chi^2}\, =\, }$' + "{:.4}".format(zg_minchi2) + \
    r'$\substack{+$' + "{:.3}".format(high_zerr) + r'$\\ -$' + "{:.3}".format(low_zerr) + r'$}$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.27, r'$\mathrm{z_{spec}\, =\, }$' + "{:.4}".format(specz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.75, 0.22, r'$\mathrm{z_{p;wt}\, =\, }$' + "{:.4}".format(zp), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.75, 0.17, r'$\mathrm{\chi^2\, =\, }$' + "{:.3}".format(chi2), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    ax1.text(0.75, 0.12, r'$\mathrm{NetSig\, =\, }$' + mr.convert_to_sci_not(netsig), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=8)
    ax1.text(0.75, 0.07, r'$\mathrm{D4000(from\ z_{spz})\, =\, }$' + "{:.3}".format(d4000), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=8)


    ax1.text(0.47, 0.3,'log(Age[yr]) = ' + "{:.4}".format(age), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)
    ax1.text(0.47, 0.25, r'$\tau$' + '[Gyr] = ' + "{:.3}".format(tau), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    if av < 0:
        av = -99.0

    ax1.text(0.47, 0.2, r'$\mathrm{A_V}$' + ' = ' + "{:.3}".format(av), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=10)

    # ---------- Plot p(z) curve in an inset figure ---------- #
    # Solution for inset came from SO:
    # https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.1, 0.6, 0.3, 0.2]
    ax3 = fig.add_axes([left, bottom, width, height])

    # Read in p(z) curve. It should be in the same folder where all these figures are being saved.
    pz = np.load(savedir + obj_field + '_' + str(obj_id) + '_spz_pz.npy')
    zarr = np.load(savedir + obj_field + '_' + str(obj_id) + '_spz_z_arr.npy')

    ax3.plot(zarr, pz)
    ax3.axvline(x=specz, ls='--', color='darkred')
    ax3.minorticks_on()

    # ---------- Save figure ---------- #
    fig.savefig(savedir + obj_field + '_' + str(obj_id) + '_spz_fit.png', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

def get_best_fit_model_spz(resampling_lam_grid, resampling_lam_grid_length, model_lam_grid, model_comp_spec, \
    grism_lam_obs, redshift, model_idx, phot_fin_idx, all_model_flam, lsf, total_models):

    # ------------ Get best fit model at grism resolution ------------ #
    # First do the convolution with the LSF
    model_comp_spec_lsfconv = ff.lsf_convolve(model_comp_spec, lsf, total_models)

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
    model_comp_spec_modified = mm.redshift_and_resample(model_comp_spec_lsfconv, redshift, total_models, model_lam_grid, resampling_lam_grid, resampling_lam_grid_length)
    print "Model mods done (only for plotting purposes) at the new SPZ:", redshift

    best_fit_model_in_objlamgrid = model_comp_spec_modified[model_idx, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # ------------ Get photomtery for model ------------- #
    all_filt_flam_bestmodel = get_photometry_best_fit_model(redshift, model_idx, phot_fin_idx, all_model_flam, total_models)

    # ------------ Get best fit model at full resolution ------------ #
    best_fit_model_fullres = model_comp_spec[model_idx]

    return best_fit_model_in_objlamgrid, all_filt_flam_bestmodel, best_fit_model_fullres

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

def main():
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # ------------------------------- Give galaxy data here ------------------------------- #
    # Only needs the ID and the field
    # And flag to modify LSF
    current_id = 51522
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
    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_data_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 37761 # get_total_extensions(bc03_all_spec_hdulist)

    # Read in models with emission lines adn put in numpy array
    bc03_all_spec_hdulist_withlines = fits.open(figs_data_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample_withlines.fits')
    model_lam_grid_withlines = np.load(figs_data_dir + 'model_lam_grid_withlines.npy')
    model_comp_spec_withlines = np.zeros((total_models, len(model_lam_grid_withlines)), dtype=np.float64)
    for q in range(total_models):
        model_comp_spec_withlines[q] = bc03_all_spec_hdulist_withlines[q+1].data

    bc03_all_spec_hdulist_withlines.close()
    del bc03_all_spec_hdulist_withlines

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

    # ---------------------------------- Read in look-up tables for model mags ------------------------------------- #
    # Using the look-up table now since it should be much faster
    # First get them all into an appropriate shape
    u = np.load(figs_data_dir + 'all_model_mags_par_u.npy')
    f435w = np.load(figs_data_dir + 'all_model_mags_par_f435w.npy')
    f606w = np.load(figs_data_dir + 'all_model_mags_par_f606w.npy')
    f775w = np.load(figs_data_dir + 'all_model_mags_par_f775w.npy')
    f850lp = np.load(figs_data_dir + 'all_model_mags_par_f850lp.npy')
    f125w = np.load(figs_data_dir + 'all_model_mags_par_f125w.npy')
    f140w = np.load(figs_data_dir + 'all_model_mags_par_f140w.npy')
    f160w = np.load(figs_data_dir + 'all_model_mags_par_f160w.npy')
    irac1 = np.load(figs_data_dir + 'all_model_mags_par_irac1.npy')
    irac2 = np.load(figs_data_dir + 'all_model_mags_par_irac2.npy')
    irac3 = np.load(figs_data_dir + 'all_model_mags_par_irac3.npy')
    irac4 = np.load(figs_data_dir + 'all_model_mags_par_irac4.npy')

    # put them in a list since I need to iterate over it
    all_model_flam = [u, f435w, f606w, f775w, f850lp, f125w, f140w, f160w, irac1, irac2, irac3, irac4]

    # cnovert to numpy array
    all_model_flam = np.asarray(all_model_flam)

    # ----------------------------------------- Read in matched catalogs ----------------------------------------- #
    # Only reading this in to be able to loop over all matched galaxies
    # read in matched files to get photo-z
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    # Read in Specz comparison catalogs
    specz_goodsn = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-N.txt', dtype=None, names=True)
    specz_goodss = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-S.txt', dtype=None, names=True)

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
        cat = matched_cat_n
        spec_cat = specz_goodsn
        phot_cat_3dhst = goodsn_phot_cat_3dhst
    elif current_field == 'GOODS-S':
        cat = matched_cat_s
        spec_cat = specz_goodss
        phot_cat_3dhst = goodss_phot_cat_3dhst

    # Get specz if it exists as initial guess, otherwise get photoz
    specz_idx = np.where(spec_cat['pearsid'] == current_id)[0]

    # Since Im' only running this code on a single galaxy with a known ground based spec-z this check should never be trigerred
    if len(specz_idx) != 1:
        print "Match not found in specz catalog for ID", current_id, "in", current_field, "... Exiting."
        sys.exit(0)

    current_specz = float(spec_cat['specz'][specz_idx])

    print "At ID", current_id, "in", current_field, "with specz:", current_specz

    # ------------------------------- Get grism data and then match with photometry ------------------------------- #
    grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)

    # find grism obj ra,dec
    cat_idx = np.where(cat['pearsid'] == current_id)[0]
    if cat_idx.size:
        current_ra = float(cat['pearsra'][cat_idx])
        current_dec = float(cat['pearsdec'][cat_idx])

    threed_ra = phot_cat_3dhst['ra']
    threed_dec = phot_cat_3dhst['dec']

    # Now match
    ra_lim = 0.5/3600  # arcseconds in degrees
    dec_lim = 0.5/3600
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

    # ------------- Call fitting function for photo-z ------------- #
    print "\n", "Computing photo-z now."

    zp_minchi2, zp, zp_zerr_low, zp_zerr_up, zp_min_chi2, zp_bestalpha, zp_model_idx, zp_age, zp_tau, zp_av = \
    photoz.do_photoz_fitting_lookup(phot_fluxes_arr, phot_errors_arr, phot_lam, \
        model_lam_grid_withlines, total_models, model_comp_spec_withlines, bc03_all_spec_hdulist, start,\
        current_id, current_field, all_model_flam, phot_fin_idx, current_specz, savedir)

    # ------------- Call fitting function for SPZ ------------- #
    print "\n", "Moving on to SPZ computation now."

    zg_minchi2, zspz, zg_zerr_low, zg_zerr_up, zg_min_chi2, zg_bestalpha, zg_model_idx, zg_age, zg_tau, zg_av = \
    ff.do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_fluxes_arr, phot_errors_arr, phot_lam, \
        lsf_to_use, resampling_lam_grid, len(resampling_lam_grid), all_model_flam, phot_fin_idx, \
        model_lam_grid_withlines, total_models, model_comp_spec_withlines, bc03_all_spec_hdulist, start,\
        current_id, current_field, current_specz, zp, use_broadband=True, single_galaxy=True)

    print "\n", "Results from both codes:"
    print "Ground-based spectroscopic redshift:", current_specz
    print "Photometric redshift from min chi2:", "{:.3f}".format(zp_minchi2)
    print "Weighted photometric redshift:", "{:.3f}".format(zp)
    print "SPZ from min chi2:", "{:.3f}".format(zg_minchi2)
    print "Weighted SPZ:", "{:.3f}".format(zspz)

    print "\n", "Time taken up to now--", str("{:.2f}".format(time.time() - start)), "seconds."
    print "Moving on to plotting fitting results."

    # ------------------------------- First get D4000. Only need to put D4000 info on plot. ------------------------------- # 
    # Get d4000 at SPZ
    lam_em = grism_lam_obs / (1 + zspz)
    flam_em = grism_flam_obs * (1 + zspz)
    ferr_em = grism_ferr_obs * (1 + zspz)

    d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

    # ------------------------------- Get best fit model for plotting ------------------------------- #
    # Will have to do this at the photo-z and SPZ separtely otherwise the plots will not look right
    # ------------ Get best fit model for photo-z ------------ #
    zp_best_fit_model_fullres = model_comp_spec_withlines[zp_model_idx]
    zp_all_filt_flam_bestmodel = get_photometry_best_fit_model(zp, zp_model_idx, phot_fin_idx, all_model_flam, total_models)

    # ------------ Get best fit model for SPZ ------------ #
    zg_best_fit_model_in_objlamgrid, zg_all_filt_flam_bestmodel, zg_best_fit_model_fullres = \
    get_best_fit_model_spz(resampling_lam_grid, len(resampling_lam_grid), model_lam_grid_withlines, model_comp_spec_withlines, \
        grism_lam_obs, zspz, zg_model_idx, phot_fin_idx, all_model_flam, lsf_to_use, total_models)

    # ------------------------------- Plotting based on results from the above two codes ------------------------------- #
    plot_photoz_fit(phot_lam, phot_fluxes_arr, phot_errors_arr, model_lam_grid_withlines, \
    zp_best_fit_model_fullres, zp_all_filt_flam_bestmodel, zp_bestalpha, \
    current_id, current_field, current_specz, zp, zp_minchi2, zp_zerr_low, zp_zerr_up, zp_min_chi2, zp_age, zp_tau, zp_av, netsig_chosen, d4000, savedir)

    plot_spz_fit(grism_lam_obs, grism_flam_obs, grism_ferr_obs, phot_lam, phot_fluxes_arr, phot_errors_arr, \
    model_lam_grid_withlines, zg_best_fit_model_fullres, zg_best_fit_model_in_objlamgrid, zg_all_filt_flam_bestmodel, zg_bestalpha, \
    current_id, current_field, current_specz, zp, zg_minchi2, zg_zerr_low, zg_zerr_up, zspz, zg_min_chi2, zg_age, zg_tau, zg_av, netsig_chosen, d4000, savedir)

    # Total time taken
    print "\n", "All done."
    print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds.", "\n"

    return None

if __name__ == '__main__':
    main()
    sys.exit()