from __future__ import division

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

def main():

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # ------------------------------- Get catalog for final sample ------------------------------- #
    final_sample = np.genfromtxt(figs_data_dir + 'spz_paper_sample.txt', dtype=None, names=True)

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
    print "Starting parallel processing. Will run each galaxy on a separate core."
    print "Total time taken up to now --", str("{:.2f}".format(time.time() - start)), "seconds."
    total_final_sample = len(final_sample)
    max_cores = 270

    for i in range(int(np.ceil(total_final_sample/max_cores))):

        jmin = i*max_cores
        jmax = (i+1)*max_cores

        if jmax > total_final_sample:
            jmax = total_final_sample

        processes = [mp.Process(target=get_all_redshifts_v2, args=(final_sample['pearsid'][j], \
            final_sample['field'][j], final_sample['ra'][j], final_sample['dec'][j], 
            final_sample['zspec'][j], goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam, \
            model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap, all_model_flam_mmap, total_models, start, \
            log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, \
            bv_col_arr, vj_col_arr, ms_arr, mgal_arr)) for j in xrange(jmin, jmax)]
        for p in processes:
            p.start()
            print "Current process ID:", p.pid
        for p in processes:
            p.join(timeout=None)

    print "Done with all galaxies. Exiting."
    print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)