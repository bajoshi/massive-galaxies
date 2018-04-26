from __future__ import division

import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import d4000_mocksim as dm
import dn4000_catalog as dc
import mag_hist as mh

def overplot_model_mockspectra(model_lam_grid, model_flam, lam_em, flam_em, ferr_em, d4000_in, d4000_out, d4000_out_err, test_redshift):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    flam_em /= 1e-12  
    # divide by constant that you multiplied with initially
    # This is only to show the spectra on the same scale.

    model_lam_idx = np.where((model_lam_grid >= 6000/(1 + test_redshift)) & (model_lam_grid <= 9500/(1 + test_redshift)))[0]

    ax.plot(model_lam_grid[model_lam_idx], model_flam[model_lam_idx])

    ax.plot(lam_em, flam_em)
    ax.fill_between(lam_em, flam_em + ferr_em, flam_em - ferr_em, color='lightgray')
    ax.set_xlim(6000/(1 + test_redshift), 9500/(1 + test_redshift))

    # Draw shaded bands that show D4000 measurement
    shade_color = mh.rgb_to_hex(204,236,230)

    arg3850 = np.argmin(abs(lam_em - 3850))
    arg4150 = np.argmin(abs(lam_em - 4150))

    x_fill = np.arange(3750, 3951, 1)
    y0_fill = np.ones(len(x_fill)) * (flam_em[arg3850] - 3*ferr_em[arg3850])
    y1_fill = np.ones(len(x_fill)) * (flam_em[arg4150] + 3*ferr_em[arg4150])
    ax.fill_between(x_fill, y0_fill, y1_fill, color=shade_color, zorder=2)

    x_fill = np.arange(4050, 4251, 1)
    ax.fill_between(x_fill, y0_fill, y1_fill, color=shade_color, zorder=2)

    plt.show()

    return None

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # ---------------------------------------------- MODELS ----------------------------------------------- #
    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_lam_grid = model_lam_grid.astype(np.float64)
    model_comp_spec = np.zeros((total_models, len(model_lam_grid)), dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec_hdulist[j+1].data

    # total run time up to now
    print "All models put in numpy array. Total time taken up to now --", time.time() - start, "seconds."

    # ---------------------------------------------- Loop ----------------------------------------------- #
    d4000_in_list = []  # D4000 measured on model before doing model modifications
    d4000_out_list = []  # D4000 measured on model after doing model modifications
    d4000_out_err_list = []

    for i in range(30,40):

        # Measure D4000 before doing modifications
        model_flam = model_comp_spec[i]
        model_ferr = np.zeros(len(model_flam))

        d4000_in, d4000_in_err = dc.get_d4000(model_lam_grid, model_flam, model_ferr)

        test_redshift = 1.1

        # Modify model and create mock spectrum
        lam_obs, flam_obs, ferr_obs = dm.get_mock_spectrum(model_lam_grid, model_comp_spec[i], test_redshift)

        # Now de-redshift and find D4000
        lam_em = lam_obs / (1 + test_redshift)
        flam_em = flam_obs * (1 + test_redshift)
        ferr_em = ferr_obs * (1 + test_redshift)

        d4000_out, d4000_out_err = dc.get_d4000(lam_em, flam_em, ferr_em)

        # Append arrays
        d4000_in_list.append(d4000_in)
        d4000_out_list.append(d4000_out)
        d4000_out_err_list.append(d4000_out_err)

        ##### Plot old and new mock spectra overlaid to check #####
        overplot_model_mockspectra(model_lam_grid, model_flam, lam_em, flam_em, ferr_em, d4000_in, d4000_out, d4000_out_err, test_redshift)

    sys.exit(0)

    # Convert to numpy arrays
    d4000_in_list = np.asarray(d4000_in_list)
    d4000_out_list = np.asarray(d4000_out_list)

    np.save(massive_figures_dir + 'model_mockspectra_fits/all_d4000_in_list_z' + str(test_redshift).replace('.', 'p') + '.npy', d4000_in_list)
    np.save(massive_figures_dir + 'model_mockspectra_fits/all_d4000_out_list_z' + str(test_redshift).replace('.', 'p') + '.npy', d4000_out_list)

    #### Now make plot comparing D4000 before and after ####
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{D4000_{instrinsic}}$')
    ax.set_ylabel(r'$\mathrm{(D4000_{instrinsic} - D4000_{mock}) / D4000_{instrinsic}}$')

    ax.set_ylim(-0.3, 0.3)

    ax.plot(d4000_in_list, (d4000_in_list - d4000_out_list) / d4000_in_list, 'o', markersize=0.6, color='k', markeredgecolor='None')
    ax.axhline(y=0.0, ls='--', color='b')

    ax.text(0.87, 0.87, r'$\mathrm{z=}$' + str(test_redshift), \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=10)

    fig.savefig(massive_figures_dir + 'model_mockspectra_fits/d4000_comparison_model_z' + str(test_redshift).replace('.', 'p') + '.eps', dpi=100, boox_inches='tight')

    # Total time taken
    print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."
    sys.exit(0)