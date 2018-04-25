from __future__ import division

import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import refine_redshifts_dn4000 as old_ref
import dn4000_catalog as dc



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

    d4000_in = []  # D4000 measured on model before doing model modifications
    d4000_out = []  # D4000 measured on model after doing model modifications

    test_redshift = 0.8

    for i in range(total_models):

        # Measure D4000 before doing modifications
        model_flam = model_comp_spec[i]
        model_ferr = np.zeros(len(model_flam))

        d4000, d4000_err = dc.get_d4000(model_lam_grid, model_flam, model_ferr)
        d4000_in.append(d4000)

        # Do modifications 
        # redshift model
        lam_obs = model_lam_grid * (1 + test_redshift)
        flam_obs = model_comp_spec[i] / (1 + test_redshift)

        # Now restrict the spectrum to be within 6000 to 9500
        valid_lam_idx = np.where((lam_obs >= 6000) & (lam_obs <= 9500))[0]
        lam_obs = lam_obs[valid_lam_idx]
        flam_obs = flam_obs[valid_lam_idx]

        # Now use a fake LSF and also resample the spectrum to the grism resolution
        mock_lsf = Gaussian1DKernel(10.0)
        flam_obs = convolve(flam_obs, mock_lsf, boundary='extend')

        # resample 
        mock_resample_lam_grid = np.linspace(6000, 9500, 88)
        resampled_flam = np.zeros((len(mock_resample_lam_grid)))
        for k in range(len(mock_resample_lam_grid)):

            if k == 0:
                lam_step_high = mock_resample_lam_grid[k+1] - mock_resample_lam_grid[k]
                lam_step_low = lam_step_high
            elif k == len(mock_resample_lam_grid) - 1:
                lam_step_low = mock_resample_lam_grid[k] - mock_resample_lam_grid[k-1]
                lam_step_high = lam_step_low
            else:
                lam_step_high = mock_resample_lam_grid[k+1] - mock_resample_lam_grid[k]
                lam_step_low = mock_resample_lam_grid[k] - mock_resample_lam_grid[k-1]

            new_ind = np.where((lam_obs >= mock_resample_lam_grid[k] - lam_step_low) & \
                (lam_obs < mock_resample_lam_grid[k] + lam_step_high))[0]
            resampled_flam[k] = np.mean(flam_obs[new_ind])

        flam_obs = resampled_flam

        # multiply flam by a constant to get to some realistic flux levels
        flam_obs *= 1e-12

        # assign a constant 5% error bar
        ferr_obs = np.ones(len(flam_obs))
        ferr_obs = 0.05 * flam_obs

        # put in random noise in the model
        for k in range(len(flam_obs)):
            flam_obs[k] = np.random.normal(flam_obs[k], ferr_obs[k], 1)

        # plot to check it looks right
        # don't delete these lines for plotting
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Deredshift the wavelength array because the model by default is at zero redshift
        # You will also have to comment the line above that multiplies flam by a constant
        # for this plot to show up correctly 
        mock_resample_lam_grid /= (1 + test_redshift)
        ax.plot(model_lam_grid, model_flam)
        ax.plot(mock_resample_lam_grid, flam_obs)
        ax.fill_between(mock_resample_lam_grid, flam_obs + ferr_obs, flam_obs - ferr_obs, color='lightgray')
        ax.set_xlim(6000/(1 + test_redshift), 9500/(1 + test_redshift))
        plt.show()
        """

        # Now de-redshift and find D4000
        lam_em = mock_resample_lam_grid / (1 + test_redshift)
        flam_em = flam_obs * (1 + test_redshift)
        ferr_em = ferr_obs * (1 + test_redshift)

        d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)
        d4000_out.append(d4000)

    # Convert to numpy arrays
    d4000_in = np.asarray(d4000_in)
    d4000_out = np.asarray(d4000_out)

    #### Now make plot comparing D4000 before and after ####
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(d4000_in, d4000_in - d4000_out, 'o', markersize=1.0, color='k', markeredgecolor='k')
    ax.axhline(y=0.0, ls='--', color='b')

    plt.show()

    sys.exit(0)