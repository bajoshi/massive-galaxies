# coding: utf-8
""" Code to run the SPZ pipeline on all of PEARS.

    It needs access to:
    1. PEARS master catalogs
    2. PEARS spectra, i.e., Wavelength, flux, and error arrays.
    3. 3D-HST photometry of GOODS
    4. Filter curves used for the photometry
    5. Model stuff:
        5a. BC03 templates (with/without emission lines added). 
            i.e., Wavelength and flux arrays.
        5b. Model parameter arrays (saved as npy files).
    6. Line spread functions for PEARS galaxies.

    What this code does, very briefly:
    Uses a redshift grid from z=0.0 to z=6.0 with a step size of 0.001,
    i.e., np.arange(0.0, 6.001, 0.001), for fitting a grid of BC03 templates.

    1. First runs save_model_mags.py to compute and save photometry
    for the templates. Makes sure that the redshift grid is consistent.

    2. It will read in all required info:
    --- Templates, model params, photometry, and lsfs.

    3. Starts fitting:
        For each galaxy -- 
        3a. Begin with convolving the models with the LSF.
        3b. Now for each redshift -- redshift and resample 
        the model and get a chi2 and alpha value for each
        model.
        3c. Find the best fit model by using the min chi2 
        while making sure that the age of the best fit 
        model is consistent with the age of the Universe 
        at that best fit z.

    4. Returns the best fit SPZ and its errors.
    Also returns the best fit model parameters.
    Also returns the p(z) curve.
    Also returns the min chi2 and its alpha value.

    Procedure to run this code:
    **** 

    ONLY RUN ON AGAVE! 
    This code is computationally very expensive.
    This code assumes that you will run it on Agave.
    
    ****

    You should only need to run the bash script on Agave.
    After logging in to Agave:
    >> interactive
    >> sbatch run_full_pears_spz.sh
    Check the number of cores in the script. It should be less than 28
    to make sure it runs on a single compute node.
    The bash script should know what to do after that.
    
"""

from __future__ import division

import numpy as np

import os
import sys
import time
import datetime

home = os.getenv('HOME')
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

import cosmology_calculator as cc
import cluster_do_fitting as cf

# ------------ Set constants ------------ #
# Get cosmology
H0, omega_m0, omega_r0, omega_lam0 = cc.get_cosmology_params()
print "Flat Universe assumed. Cosmology assumed is (from Planck 2018):"
print "Hubble constant [km/s/Mpc]:", H0
print "Omega Matter:", omega_m0
print "Omega Lambda", omega_lam0
print "Omega Radiation:", omega_r0
speed_of_light_kms = 299792.458  # in km/s
print "Speed of Light [km/s]:", speed_of_light_kms
speed_of_light_ang = speed_of_light_kms * 1e3 * 1e10  # In angstroms per second

def main():

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # ------------------------------- Read PEARS cats ------------------------------- #
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    
    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['pearsdec'] = pears_ncat['pearsdec'] - dec_offset_goodsn_v19

    # ------------------------------ Get models and photometry ------------------------------ #
    # read in entire model set
    total_models = 37761

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

    model_lam_grid_withlines_mmap = np.load(figs_data_dir + 'model_lam_grid_withlines.npy', mmap_mode='r')
    model_comp_spec_withlines_mmap = np.load(figs_data_dir + 'model_comp_spec_withlines.npy', mmap_mode='r')

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", 
    print time.time() - start, "seconds."

    all_model_flam_mmap = np.load(figs_data_dir + 'all_model_flam.npy', mmap_mode='r')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)