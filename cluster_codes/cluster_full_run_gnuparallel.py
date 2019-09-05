# coding: utf-8
""" Help on redshift fitting module. 

__author__ = "Bhavin Joshi"

This is the code to run the redshift pipeline on all of PEARS
OR 
on the final sample for the SPZ paper.

Inputs:
    Needs the user to specify:
    1. The sample to use. This is the first thing in main()
    after it prints out the starting time.

    Both of these samples (txt files) will have to be created
    before this code can run. See procedure section below.

    Needs access to:
    1. PEARS master catalogs
    2. PEARS spectra, i.e., wavelength, flux, and error arrays.
    3. 3D-HST photometry of GOODS
    4. Filter curves used for the photometry
    5. Model stuff:
        5a. BC03 templates (with/without emission lines added). 
            i.e., Wavelength and flux arrays.
        5b. Model parameter arrays (saved as .npy files).
    6. Line spread functions for PEARS galaxies.

Returns:
    Returns the best fit SPZ and its errors.
    Also returns the best fit model parameters.
    Also returns the p(z) curve.
    Also returns the min chi2 and its alpha value.

Methodology:
    What this code does, very briefly.
    For ALL of PEARS -- 
    Uses a redshift grid from z=0.0 to z=6.0 with a step size of 0.001,
    i.e., np.arange(0.0, 6.001, 0.001), for fitting a grid of BC03 templates.
    For SPZ paper -- 
    Uses a redshift grid from z=0.3 to z=1.5 with a step size of 0.01.

    1. First runs cluster_save_all_model_flam.py to compute and save photometry
    for the templates. Makes sure that the redshift grid is consistent.

    2. It will read in all required info.
    Templates, model params, photometry, and lsfs.

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

Procedure to run this code:

    1. Generate the txt file that contains the sample.

    For SPZ paper:
    >> python get_spz_galaxies_zrange.py

    For full PEARS:
    >> python create_full_pears_sample_file.py

    Both codes in $HOME/Desktop/FIGS/massive-galaxies/grismz_pipeline/ folder.

    2. 
    **** 

    ONLY RUN THE NEXT TWO STEPS ON AGAVE! 
    This code is computationally very expensive.
    This code assumes that you will run it on Agave.
    
    ****

    You should only need to run the bash script on Agave.
    For both steps, make sure that the cores and execution 
    time are set correctly in the bash script.

    Step 1.
    After logging in to Agave:
    >> interactive
    >> sbatch run_model_flam.sh  # to get model photometry  # has to be done first
    Needs >12 cores, tested with 14. 

    Step 2.
    After step 1 finishes (takes a long time ~days).

    Again log into Agave:
    >> interactive
    >> sbatch run_full_pears_spz.sh  # to run code on the selected sample
    Needs a LOT of cores, tested with 25. 
    Will update it to be able to run with even more.

    Check the number of cores in both scripts. It should be less than 28
    to make sure it runs on a single compute node.
    The bash script should know what to do after that.
    
"""

from __future__ import division,print_function

import numpy as np
from astropy.io import fits
import multiprocessing as mp

import os
import sys
import time
import datetime

figs_data_dir = "/home/bajoshi/models_and_photometry/"
threedhst_datadir = figs_data_dir
cluster_spz_scripts = "/home/bajoshi/spz_scripts/"

# Only for testing with firstlight
# Comment this out before copying code to Agave
# Uncomment above directory paths which are correct for Agave
#figs_data_dir = '/Users/baj/Desktop/FIGS/'
#threedhst_datadir = '/Users/baj/Desktop/3dhst_data/'
#cluster_spz_scripts = '/Users/baj/Desktop/FIGS/massive-galaxies/cluster_codes/'

sys.path.append(cluster_spz_scripts)
from cluster_get_all_redshifts import get_all_redshifts_v2

def main(arr_index):

    # Start time
    start = time.time()
    dt = datetime.datetime
    print("Starting at --", dt.now())

    """
    figs_data_dir = '/Users/bhavinjoshi/Desktop/FIGS/'
    threedhst_datadir = '/Volumes/Bhavins_backup/3dhst_data/'

    final_sample = np.genfromtxt(figs_data_dir + 'massive-galaxies/pears_full_sample.txt', dtype=None, names=True)

    log_age_arr = np.zeros(5)
    metal_arr = np.zeros(5)
    nlyc_arr = np.zeros(5)
    tau_gyr_arr = np.zeros(5)
    tauv_arr = np.zeros(5)
    ub_col_arr = np.zeros(5)
    bv_col_arr = np.zeros(5)
    vj_col_arr = np.zeros(5)
    ms_arr = np.zeros(5)
    mgal_arr = np.zeros(5)

    model_lam_grid_withlines_mmap = np.zeros(5)
    model_comp_spec_llam_withlines_mmap = np.zeros(5)
    all_model_flam_mmap = np.zeros(5)

    get_grismz = False
    """

    # ------------------------------- Get catalog for final sample ------------------------------- #
    # ------------- Also set other flags ------------- #
    # Flag for full sample run
    run_for_full_pears = False

    # Flag for SPZ 
    get_spz = True
    # Other IRAC photometry flags
    ignore_irac = False
    ignore_irac_ch3_ch4 = False

    # CHoose IMF
    # Either 'Salpeter' or 'Chabrier'
    chosen_imf = 'Salpeter'

    if chosen_imf == 'Salpeter':
        npy_end_str = '_salp.npy'
        csp_str = ''
    elif chosen_imf == 'Chabrier':
        npy_end_str = '_chab.npy'
        csp_str = '_chabrier'

    if run_for_full_pears:
        final_sample = np.genfromtxt(figs_data_dir + 'massive-galaxies/pears_full_sample.txt', dtype=None, names=True)
        get_grismz = False
    else:
        final_sample = np.genfromtxt(figs_data_dir + 'spz_paper_sample.txt', dtype=None, names=True)
        get_grismz = True

    # ------------------------------ Get models and photometry ------------------------------ #
    # read in entire model set
    total_models = 37761

    log_age_arr = np.load(figs_data_dir + 'log_age_arr' + npy_end_str, mmap_mode='r')
    metal_arr = np.load(figs_data_dir + 'metal_arr' + npy_end_str, mmap_mode='r')
    nlyc_arr = np.load(figs_data_dir + 'nlyc_arr' + npy_end_str, mmap_mode='r')
    tau_gyr_arr = np.load(figs_data_dir + 'tau_gyr_arr' + npy_end_str, mmap_mode='r')
    tauv_arr = np.load(figs_data_dir + 'tauv_arr' + npy_end_str, mmap_mode='r')
    ub_col_arr = np.load(figs_data_dir + 'ub_col_arr' + npy_end_str, mmap_mode='r')
    bv_col_arr = np.load(figs_data_dir + 'bv_col_arr' + npy_end_str, mmap_mode='r')
    vj_col_arr = np.load(figs_data_dir + 'vj_col_arr' + npy_end_str, mmap_mode='r')
    ms_arr = np.load(figs_data_dir + 'ms_arr' + npy_end_str, mmap_mode='r')
    mgal_arr = np.load(figs_data_dir + 'mgal_arr' + npy_end_str, mmap_mode='r')

    # Read model lambda grid # In agnstroms
    model_lam_grid_withlines_mmap = np.load(figs_data_dir + 'model_lam_grid_withlines' + csp_str + '.npy', mmap_mode='r')
    # Now read the model spectra # In erg s^-1 A^-1
    model_comp_spec_llam_withlines_mmap = np.load(figs_data_dir + 'model_comp_spec_llam_withlines' + csp_str + '.npy', mmap_mode='r')

    # total run time up to now
    print("All models now in numpy array and have emission lines. Total time taken up to now --",)
    print(time.time() - start, "seconds.")

    all_model_flam_mmap = np.load(figs_data_dir + 'all_model_flam' + csp_str + '.npy', mmap_mode='r')

    # ------------------------------- Read in photometry catalogs ------------------------------- #
    # GOODS photometry catalogs from 3DHST
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
    'f_U', 'e_U', 'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
    'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), \
        skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), \
        skip_header=3)

    # Read in Vega spectrum and get it in the appropriate forms
    vega = np.genfromtxt(figs_data_dir + 'vega_reference.dat', dtype=None, \
        names=['wav', 'flam'], skip_header=7)

    speed_of_light = 299792458e10  # angstroms per second
    vega_lam = vega['wav']
    vega_spec_flam = vega['flam']
    vega_nu = speed_of_light / vega_lam
    vega_spec_fnu = vega_lam**2 * vega_spec_flam / speed_of_light

    # ------------------------------ Parallel processing ------------------------------ #
    print("Starting parallel processing. Will run each galaxy on a separate core.")
    print("Total time taken up to now --", str("{:.2f}".format(time.time() - start)), "seconds.")
    total_final_sample = len(final_sample)
    max_cores = 3

    """
    Use the following code block to run only on the 4 galaxies to be shown in
    figure 3 of the paper. This should be run with 4 cores on Agave.
    Make sure to change the number of cores in the bash script as well.
    The results will be overwritten so make sure to save any old results 
    that are needed.
    Don't need to change max cores at all.
    Comment out the code block to be used for the full sample.
    """
    """
    # First get the properties of galaxies in Figure 3
    fig3_id_list = [82267, 48189, 100543, 126769]
    fig3_field_list = ['GOODS-N', 'GOODS-N', 'GOODS-S', 'GOODS-S']
    fig3_ra_list = []
    fig3_dec_list = []
    fig3_zspec_list = []

    for i in range(len(fig3_id_list)):

        c_id = fig3_id_list[i]
        c_field = fig3_field_list[i]

        id_idx = np.where((final_sample['pearsid'] == c_id) & (final_sample['field'] == c_field))[0]

        fig3_ra_list.append(final_sample['ra'][id_idx])
        fig3_dec_list.append(final_sample['dec'][id_idx])
        fig3_zspec_list.append(final_sample['zspec'][id_idx])
        
    # Now run the pipeline for the example galaxies
    processes = [mp.Process(target=get_all_redshifts_v2, args=(fig3_id_list[u], \
        fig3_field_list[u], fig3_ra_list[u], fig3_dec_list[u], 
        fig3_zspec_list[u], goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam, \
        model_lam_grid_withlines_mmap, model_comp_spec_llam_withlines_mmap, all_model_flam_mmap, total_models, start, \
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, \
        bv_col_arr, vj_col_arr, ms_arr, mgal_arr, get_spz, get_grismz, run_for_full_pears)) for u in range(len(fig3_id_list))]

    print processes

    # Run all example galaxies
    for p in processes:
        p.start()
        print "Current process ID:", p.pid
    for p in processes:
        p.join()
    """

    get_all_redshifts_v2(
        final_sample['pearsid'][arr_index], 
        final_sample['field'][arr_index], 
        final_sample['ra'][arr_index], 
        final_sample['dec'][arr_index], 
        final_sample['zspec'][arr_index], 
        goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, 
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam, 
        model_lam_grid_withlines_mmap, model_comp_spec_llam_withlines_mmap, 
        all_model_flam_mmap, total_models, start, 
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, 
        bv_col_arr, vj_col_arr, ms_arr, mgal_arr, 
        get_spz, get_grismz, run_for_full_pears, 
        ignore_irac, ignore_irac_ch3_ch4, chosen_imf 
    )

    print("Done with index {:d} galaxy. Exiting.".format(arr_index))
    print("Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds.")

    return None

if __name__ == '__main__':
    arr_index = int(sys.argv[1])
    main(arr_index)
    sys.exit(0)
