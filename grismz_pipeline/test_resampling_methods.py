from __future__ import division

import numpy as np
from scipy.interpolate import griddata
from astropy.io import fits

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

home = os.getenv('HOME')
figs_dir = home + "/Desktop/FIGS/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import fullfitting_grism_broadband_emlines as fg
import refine_redshifts_dn4000 as old_ref
import model_mods as mm

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # ------------------------------ Add emission lines to models ------------------------------ #
    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_lam_grid = model_lam_grid.astype(np.float64)

    total_emission_lines_to_add = 12  # Make sure that this changes if you decide to add more lines to the models
    model_comp_spec_withlines = np.zeros((total_models, len(model_lam_grid) + total_emission_lines_to_add), dtype=np.float64)
    for j in range(total_models):
        nlyc = float(bc03_all_spec_hdulist[j+1].header['NLYC'])
        metallicity = float(bc03_all_spec_hdulist[j+1].header['METAL'])
        model_lam_grid_withlines, model_comp_spec_withlines[j] = \
        fg.emission_lines(metallicity, model_lam_grid, bc03_all_spec_hdulist[j+1].data, nlyc)
        # Also checked that in every case model_lam_grid_withlines is the exact same
        # SO i'm simply using hte output from the last model.

        # To include the models without the lines as well you will have to make sure that 
        # the two types of models (ie. with and without lines) are on the same lambda grid.
        # I guess you could simply interpolate the model without lines on to the grid of
        # the models wiht lines.

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

    # ------------------------------ Do model mods ------------------------------ #
    current_id = 85920
    current_field = 'GOODS-N'
    z = 1.014
    # read in some test LSF
    # Read in LSF
    if current_field == 'GOODS-N':
        lsf_filename = lsfdir + "north_lsfs/" + "n" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"
    elif current_field == 'GOODS-S':
        lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"

    # read in LSF file
    lsf = np.genfromtxt(lsf_filename)
    lsf = lsf.astype(np.float64)  # Force dtype for cython code

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

    # First do the convolution with the LSF
    model_comp_spec_lsfconv = fg.lsf_convolve(model_comp_spec_withlines, lsf_to_use, total_models)
    print "Convolution done.",
    print "Total time taken up to now --", time.time() - start_time, "seconds."

    # ------- Make new resampling grid ------- # 
    # extend lam_grid to be able to move the lam_grid later 
    avg_dlam = old_ref.get_avg_dlam(grism_lam_obs)

    lam_low_to_insert = np.arange(6000, grism_lam_obs[0], avg_dlam, dtype=np.float64)
    lam_high_to_append = np.arange(grism_lam_obs[-1] + avg_dlam, 10000, avg_dlam, dtype=np.float64)

    resampling_lam_grid = np.insert(grism_lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    # --------------- Redshift model --------------- #
    redshift_factor = 1.0 + z
    model_lam_grid_z = model_lam_grid_withlines * redshift_factor
    model_comp_spec_redshifted = model_comp_spec_lsfconv / redshift_factor

    # --------------- Do resampling --------------- #
    # Define array to save modified models
    resampling_lam_grid_length = len(resampling_lam_grid)
    model_comp_spec_modified = np.zeros((total_models, resampling_lam_grid_length), dtype=np.float64)
    model_comp_spec_modified_old = np.zeros((total_models, resampling_lam_grid_length), dtype=np.float64)

    # --------------- Get indices for resampling --------------- #
    # These indices are going to be different each time depending on the redshfit.
    # i.e. Since it uses the redshifted model_lam_grid_z to get indices.
    indices = []
    ### Zeroth element
    lam_step = resampling_lam_grid[1] - resampling_lam_grid[0]
    indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[0] - lam_step) & (model_lam_grid_z < resampling_lam_grid[0] + lam_step))[0])

    ### all elements in between
    for i in range(1, resampling_lam_grid_length - 1):
        indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[i-1]) & (model_lam_grid_z < resampling_lam_grid[i+1]))[0])

    ### Last element
    lam_step = resampling_lam_grid[-1] - resampling_lam_grid[-2]
    indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[-1] - lam_step) & (model_lam_grid_z < resampling_lam_grid[-1] + lam_step))[0])

    # Trying to do the resampling with griddata
    for k in range(total_models):
        model_comp_spec_modified[k] = \
        griddata(points=model_lam_grid_z, values=model_comp_spec_redshifted[k], xi=resampling_lam_grid, method='linear')

        # Now try resampling the old way
        model_comp_spec_modified_old[k] = [mm.simple_mean(model_comp_spec_redshifted[k][indices[q]]) for q in range(resampling_lam_grid_length)]

    # Now plot and check
    for i in range(total_models):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(resampling_lam_grid, model_comp_spec_modified[i])
        ax.plot(resampling_lam_grid, model_comp_spec_modified_old[i])

        plt.show()

        plt.clf()
        plt.cla()
        plt.close()

    sys.exit(0)