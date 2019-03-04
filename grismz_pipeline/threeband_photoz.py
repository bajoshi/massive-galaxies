from __future__ import division

import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import griddata, interp1d
from scipy.integrate import simps
from joblib import Parallel, delayed
import pysynphot

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

home = os.getenv('HOME')
figs_data_dir = home + "/Desktop/FIGS/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = figs_data_dir + 'massive-galaxies-figures/'
three_band_photoz_dir = massive_figures_dir + "three_band_photoz/"
threedhst_datadir = home + "/Desktop/3dhst_data/"  # this path only exists on firstlight

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
from fullfitting_grism_broadband_emlines import get_flam, get_flam_nonhst
import new_refine_grismz_gridsearch_parallel as ngp
from photoz import do_photoz_fitting_lookup

speed_of_light = 299792458e10  # angsroms per second

def main():

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # ------------------------------- Get catalog for final sample ------------------------------- #
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

    model_lam_grid_withlines_mmap = np.load(figs_data_dir + 'model_lam_grid_withlines.npy', mmap_mode='r')
    model_comp_spec_withlines_mmap = np.load(figs_data_dir + 'model_comp_spec_withlines.npy', mmap_mode='r')

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

    # ---------------------------------- Read in look-up tables for model mags ------------------------------------- #
    # Using the look-up table now since it should be much faster
    # Again check the code --
    # $HOME/Desktop/test-codes/shared_memory_multiprocessing/shmem_parallel_proc.py
    # to see how this was created
    # This part will fail if the array does not already exist.
    all_model_flam_mmap = np.load(figs_data_dir + 'all_model_flam.npy', mmap_mode='r')

    # ------------------------------- Get filters ------------------------------- #
    """
    f606w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f606w')
    f775w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f775w')
    f850lp_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f850lp')

    all_filters = [f606w_filt_curve, f775w_filt_curve, f850lp_filt_curve]

    print np.sum(f850lp_filt_curve(f850lp_filt_curve.binset))
    print np.sum(f775w_filt_curve(f775w_filt_curve.binset))
    print np.sum(f606w_filt_curve(f606w_filt_curve.binset))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(f850lp_filt_curve.binset, f850lp_filt_curve(f850lp_filt_curve.binset), color='seagreen')
    ax.plot(f606w_filt_curve.binset, f606w_filt_curve(f606w_filt_curve.binset), color='orange')
    ax.plot(f775w_filt_curve.binset, f775w_filt_curve(f775w_filt_curve.binset), color='rebeccapurple')
    plt.show()
    sys.exit(0)
    """

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

    # ------------------------------- Loop and fit ------------------------------- #
    # All the grism stuff in here is not used in the fit
    # It is only here to check that the photometry and grism data line up

    id_list = []
    field_list = []
    zspec_list = []
    chi2_list = []
    age_list = []
    tau_list = []
    av_list = []
    my_photoz_wt_list = []
    my_photoz_minchi2_list = []

    total_final_sample = len(final_sample)
    for j in range(total_final_sample):

        current_id = final_sample['pearsid'][j]
        current_field = final_sample['field'][j]
        current_specz = final_sample['zspec'][j]
        current_ra = final_sample['ra'][j]
        current_dec = final_sample['dec'][j]

        #grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)

        print "\n", "Working on:", current_field, current_id, "at", current_specz

        modify_lsf = True

        # Assign catalogs 
        if current_field == 'GOODS-N':
            phot_cat_3dhst = goodsn_phot_cat_3dhst
        elif current_field == 'GOODS-S':
            phot_cat_3dhst = goodss_phot_cat_3dhst

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
        flam_f435w = get_flam('F435W', phot_cat_3dhst['f_F435W'][threed_phot_idx])
        flam_f606w = get_flam('F606W', phot_cat_3dhst['f_F606W'][threed_phot_idx])
        flam_f775w = get_flam('F775W', phot_cat_3dhst['f_F775W'][threed_phot_idx])
        flam_f850lp = get_flam('F850LP', phot_cat_3dhst['f_F850LP'][threed_phot_idx])
        flam_f125w = get_flam('F125W', phot_cat_3dhst['f_F125W'][threed_phot_idx])
        flam_f140w = get_flam('F140W', phot_cat_3dhst['f_F140W'][threed_phot_idx])
        flam_f160w = get_flam('F160W', phot_cat_3dhst['f_F160W'][threed_phot_idx])

        flam_U = get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['f_U'][threed_phot_idx], \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
        flam_irac1 = get_flam_nonhst('irac1', phot_cat_3dhst['f_IRAC1'][threed_phot_idx], \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
        flam_irac2 = get_flam_nonhst('irac2', phot_cat_3dhst['f_IRAC2'][threed_phot_idx], \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
        flam_irac3 = get_flam_nonhst('irac3', phot_cat_3dhst['f_IRAC3'][threed_phot_idx], \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
        flam_irac4 = get_flam_nonhst('irac4', phot_cat_3dhst['f_IRAC4'][threed_phot_idx], \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

        ferr_f435w = get_flam('F435W', phot_cat_3dhst['e_F435W'][threed_phot_idx])
        ferr_f606w = get_flam('F606W', phot_cat_3dhst['e_F606W'][threed_phot_idx])
        ferr_f775w = get_flam('F775W', phot_cat_3dhst['e_F775W'][threed_phot_idx])
        ferr_f850lp = get_flam('F850LP', phot_cat_3dhst['e_F850LP'][threed_phot_idx])
        ferr_f125w = get_flam('F125W', phot_cat_3dhst['e_F125W'][threed_phot_idx])
        ferr_f140w = get_flam('F140W', phot_cat_3dhst['e_F140W'][threed_phot_idx])
        ferr_f160w = get_flam('F160W', phot_cat_3dhst['e_F160W'][threed_phot_idx])

        ferr_U = get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['e_U'][threed_phot_idx], \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
        ferr_irac1 = get_flam_nonhst('irac1', phot_cat_3dhst['e_IRAC1'][threed_phot_idx], \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
        ferr_irac2 = get_flam_nonhst('irac2', phot_cat_3dhst['e_IRAC2'][threed_phot_idx], \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
        ferr_irac3 = get_flam_nonhst('irac3', phot_cat_3dhst['e_IRAC3'][threed_phot_idx], \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
        ferr_irac4 = get_flam_nonhst('irac4', phot_cat_3dhst['e_IRAC4'][threed_phot_idx], \
            vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

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

        # ------------------------------- Apply aperture correction ------------------------------- #
        """
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

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(grism_lam_obs, grism_flam_obs, 'o-', color='k', markersize=2)
        ax.fill_between(grism_lam_obs, grism_flam_obs + grism_ferr_obs, grism_flam_obs - grism_ferr_obs, color='lightgray')
        plt.show(block=False)

        ff.check_spec_plot(current_id, current_field, grism_lam_obs, grism_flam_obs, grism_ferr_obs, \
        phot_lam, phot_fluxes_arr, phot_errors_arr)
        """

        # ------------------------------ Now start fitting ------------------------------ #
        # --------- Force dtype for cython code --------- #
        # Apparently this (i.e. for flam_obs and ferr_obs) has  
        # to be done to avoid an obscure error from parallel in joblib --
        # AttributeError: 'numpy.ndarray' object has no attribute 'offset'
        phot_lam = phot_lam.astype(np.float64)
        phot_fluxes_arr = phot_fluxes_arr.astype(np.float64)
        phot_errors_arr = phot_errors_arr.astype(np.float64)

        # Since I want only the three bands that overlap with the grism coverage
        # I'm going to use the phot_fin_idx variable to only select the three bands.
        # And also select only those that are finite within those three.
        phot_filt_idx = np.array([2, 3, 4])
        # This is essentially a way of only selecting filters 
        # at the positions (0-indexed) 2, 3, and 4.
        # So make sure the order of the filters passed is consistent with this.

        # ------- Finite photometry values ------- # 
        # Make sure that the photometry arrays all have finite values
        # If any vlues are NaN then throw them out
        phot_fluxes_finite_idx = np.where(np.isfinite(phot_fluxes_arr))[0]
        phot_errors_finite_idx = np.where(np.isfinite(phot_errors_arr))[0]

        phot_fin_idx = reduce(np.intersect1d, (phot_fluxes_finite_idx, phot_errors_finite_idx, phot_filt_idx))

        phot_fluxes_arr = phot_fluxes_arr[phot_fin_idx]
        phot_errors_arr = phot_errors_arr[phot_fin_idx]
        phot_lam = phot_lam[phot_fin_idx]

        # ------------- Call actual fitting function ------------- #
        zp_minchi2, zp, zp_zerr_low, zp_zerr_up, zp_min_chi2, zp_bestalpha, zp_model_idx, zp_age, zp_tau, zp_av = \
        do_photoz_fitting_lookup(phot_fluxes_arr, phot_errors_arr, phot_lam, \
            model_lam_grid_withlines_mmap, total_models, model_comp_spec_withlines_mmap, start,\
            current_id, current_field, all_model_flam_mmap, phot_fin_idx, current_specz, three_band_photoz_dir, \
            log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr)

        # ---------------------------------------------- SAVE PARAMETERS ----------------------------------------------- #
        id_list.append(current_id)
        field_list.append(current_field)
        zspec_list.append(current_specz)
        chi2_list.append(zp_min_chi2)
        age_list.append(zp_age)
        tau_list.append(zp_tau)
        av_list.append(zp_av)
        my_photoz_wt_list.append(zp)
        my_photoz_minchi2_list.append(zp_minchi2)

    # Save files
    np.save(three_band_photoz_dir + 'id_list.npy', id_list)
    np.save(three_band_photoz_dir + 'field_list.npy', field_list)
    np.save(three_band_photoz_dir + 'zspec_list.npy', zspec_list)
    np.save(three_band_photoz_dir + 'zp_min_chi2_list.npy', chi2_list)
    np.save(three_band_photoz_dir + 'zp_age_list.npy', age_list)
    np.save(three_band_photoz_dir + 'zp_tau_list.npy', tau_list)
    np.save(three_band_photoz_dir + 'zp_av_list.npy', av_list)
    np.save(three_band_photoz_dir + 'zp_wt_list.npy', my_photoz_wt_list)
    np.save(three_band_photoz_dir + 'zp_minchi2_list.npy', my_photoz_minchi2_list)

    return None

if __name__ == '__main__':
    main()
    sys.exit()