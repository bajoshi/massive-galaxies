from __future__ import division

import numpy as np
from numpy import nansum
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo
from joblib import Parallel, delayed
from scipy.interpolate import griddata, interp1d
from scipy.signal import fftconvolve
import pysynphot
from scipy.integrate import simps

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
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"
threedhst_datadir = home + "/Desktop/3dhst_data/"
massive_figures_dir = figs_dir + 'massive-galaxies-figures/'

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
import fullfitting_grism_broadband_emlines as ff

speed_of_light = 299792458e10  # angsroms per second

def get_chi2_alpha_at_z_photoz(z, phot_flam_obs, phot_ferr_obs, phot_lam_obs, model_lam_grid, model_comp_spec, all_filters, total_models, start_time):

    print "\n", "Currently at redshift:", z

    # ------------------------------------ Now compute model filter magnitudes ------------------------------------ #
    all_filt_flam_model = np.zeros((len(all_filters), total_models), dtype=np.float64)

    # Redshift the base models
    model_comp_spec_z = model_comp_spec / (1+z)
    model_lam_grid_z = model_lam_grid * (1+z)

    filt_count = 0
    for filt in all_filters:

        # first interpolate the transmission curve to the model lam grid
        # Check if the filter is an HST filter or not
        # It is an HST filter if it comes from pysynphot
        # IF it is a non-HST filter then it is a simple ascii file
        if type(filt) == pysynphot.obsbandpass.ObsModeBandpass:
            # Interpolate using the attributes of pysynphot filters
            filt_interp = griddata(points=filt.binset, values=filt(filt.binset), xi=model_lam_grid_z, method='linear')

        elif type(filt) == np.ndarray:
            filt_interp = griddata(points=filt['wav'], values=filt['trans'], xi=model_lam_grid_z, method='linear')

        # multiply model spectrum to filter curve
        for i in xrange(total_models):

            num = np.sum(model_comp_spec_z[i] * filt_interp)
            den = np.sum(filt_interp)

            filt_flam_model = num / den
            all_filt_flam_model[filt_count,i] = filt_flam_model

        # This one-liner acomplishes the same thing as the small for loop above.
        # But for some reason it is much slower. So I'm going to stick with the 
        # explicit for loop which is ~1.6x faster.
        # all_filt_flam_model[filt_count] = np.nansum(model_comp_spec_z * filt_interp, axis=1) / np.nansum(filt_interp)
        filt_count += 1

    # transverse array to make shape consistent with others
    # I did it this way so that in the above for loop each filter is looped over only once
    # i.e. minimizing the number of times each filter is gridded on to the model grid
    all_filt_flam_model_t = all_filt_flam_model.T

    print "Filter f_lam for models computed."
    print "Total time taken up to now --", time.time() - start_time, "seconds."

    # ------------------------------------ Now do the chi2 computation ------------------------------------ #
    # compute alpha and chi2
    alpha_ = np.sum(phot_flam_obs * all_filt_flam_model_t / (phot_ferr_obs**2), axis=1) / np.sum(all_filt_flam_model_t**2 / phot_ferr_obs**2, axis=1)
    chi2_ = np.sum(((phot_flam_obs - (alpha_ * all_filt_flam_model).T) / phot_ferr_obs)**2, axis=1)

    print "Min chi2 for redshift:", min(chi2_)

    return chi2_, alpha_

def do_photoz_fitting(phot_flam_obs, phot_ferr_obs, phot_lam_obs,\
    model_lam_grid, total_models, model_comp_spec, bc03_all_spec_hdulist, start_time,\
    obj_id, obj_field, specz, photoz, all_filters):
    """
    All models are redshifted to each of the redshifts in the list defined below,
    z_arr_to_check. Then the model modifications are done at that redshift.

    For each iteration through the redshift list it computes a chi2 for each model.
    """

    # Set up redshift grid to check
    z_arr_to_check = np.linspace(start=0.3, stop=1.5, num=61, dtype=np.float64)
    print "Will check the following redshifts:", z_arr_to_check

    ####### ------------------------------------ Main loop through redshfit array ------------------------------------ #######
    # Loop over all redshifts to check
    # set up chi2 and alpha arrays
    chi2 = np.empty((len(z_arr_to_check), total_models))
    alpha = np.empty((len(z_arr_to_check), total_models))

    # looping
    """
    num_cores = 3
    chi2_alpha_list = Parallel(n_jobs=num_cores)(delayed(get_chi2_alpha_at_z_photoz)(z, \
        phot_flam_obs, phot_ferr_obs, phot_lam_obs, model_lam_grid, model_comp_spec, all_filters, total_models, start_time) \
    for z in z_arr_to_check)

    # the parallel code seems to like returning only a list
    # so I have to unpack the list
    for i in range(len(z_arr_to_check)):
        chi2[i], alpha[i] = chi2_alpha_list[i]
    """

    count = 0
    for z in z_arr_to_check:
        chi2[count], alpha[count] = get_chi2_alpha_at_z_photoz(z, phot_flam_obs, phot_ferr_obs, phot_lam_obs, model_lam_grid, model_comp_spec, all_filters, total_models, start_time)
        count += 1

    ####### -------------------------------------- Min chi2 and best fit params -------------------------------------- #######
    # Sort through the chi2 and make sure that the age is physically meaningful
    sortargs = np.argsort(chi2, axis=None)  # i.e. it will use the flattened array to sort

    for k in xrange(len(chi2.ravel())):

        # Find the minimum chi2
        min_idx = sortargs[k]
        min_idx_2d = np.unravel_index(min_idx, chi2.shape)

        # Get the best fit model parameters
        # first get the index for the best fit
        model_idx = int(min_idx_2d[1])

        age = float(bc03_all_spec_hdulist[model_idx + 1].header['LOG_AGE'])
        current_z = z_arr_to_check[min_idx_2d[0]]
        age_at_z = cosmo.age(current_z).value * 1e9  # in yr

        # now check if the best fit model is an ssp or csp 
        # only the csp models have tau and tauV parameters
        # so if you try to get these keywords for the ssp fits files
        # it will fail with a KeyError
        if 'TAU_GYR' in list(bc03_all_spec_hdulist[model_idx + 1].header.keys()):
            tau = float(bc03_all_spec_hdulist[model_idx + 1].header['TAU_GYR'])
            tauv = float(bc03_all_spec_hdulist[model_idx + 1].header['TAUV'])
        else:
            # if the best fit model is an SSP then assign -99.0 to tau and tauV
            tau = -99.0
            tauv = -99.0

        # now check if the age is meaningful
        if (age < np.log10(age_at_z - 1e8)) and (age > 9 + np.log10(0.1)):
            # If the age is meaningful then you don't need to do anything
            # more. Just break out of the loop. the best fit parameters have
            # already been assigned to variables. This assignment is done before 
            # the if statement to make sure that there are best fit parameters 
            # even if the loop is broken out of in the first iteration.
            break

    print "Minimum chi2:", "{:.4}".format(chi2[min_idx_2d])
    zp_minchi2 = z_arr_to_check[min_idx_2d[0]]

    print "Current best fit log(age [yr]):", "{:.4}".format(age)
    print "Current best fit Tau [Gyr]:", "{:.4}".format(tau)
    print "Current best fit Tau_V:", tauv

    ############# -------------------------- Errors on z and other derived params ----------------------------- #############
    min_chi2 = chi2[min_idx_2d]
    # See Andrae+ 2010;arXiv:1012.3754. The number of d.o.f. for non-linear models 
    # is not well defined and reduced chi2 should really not be used.
    # Seth's comment: My model is actually linear. Its just a factor 
    # times a set of fixed points. And this is linear, because each
    # model is simply a function of lambda, which is fixed for a given 
    # model. So every model only has one single free parameter which is
    # alpha i.e. the vertical scaling factor; that's true since alpha is 
    # the only one I'm actually solving for to get a min chi2. I'm not 
    # varying the other parameters - age, tau, av, metallicity, or 
    # z - within a given model. Therefore, I can safely use the 
    # methods described in Andrae+ 2010 for linear models.
    dof = len(phot_lam_obs) - 1  # i.e. total data points minus the single fitting parameter

    chi2_red = chi2 / dof
    chi2_red_error = np.sqrt(2/dof)
    min_chi2_red = min_chi2 / dof
    print "Error in reduced chi-square:", chi2_red_error
    chi2_red_2didx = np.where((chi2_red >= min_chi2_red - chi2_red_error) & (chi2_red <= min_chi2_red + chi2_red_error))
    print "Indices within 1-sigma of reduced chi-square:", chi2_red_2didx

    # use first dimension indices to get error on zphot
    z_range = z_arr_to_check[chi2_red_2didx[0]]
    print "z range", z_range

    low_z_lim = np.min(z_range)
    upper_z_lim = np.max(z_range)
    print "Min z within 1-sigma error:", low_z_lim
    print "Max z within 1-sigma error:", upper_z_lim

    # Save chi2 map
    np.save(massive_figures_dir + 'large_diff_specz_sample/' + obj_field + '_' + str(obj_id) + '_chi2_map.npy', chi2/dof)
    np.save(massive_figures_dir + 'large_diff_specz_sample/' + obj_field + '_' + str(obj_id) + '_z_arr.npy', z_arr_to_check)

    pz = get_pz_and_plot_photoz(chi2/dof, z_arr_to_check, specz, photoz, zp_minchi2, low_z_lim, upper_z_lim, obj_id, obj_field)

    # Save p(z)
    np.save(massive_figures_dir + 'large_diff_specz_sample/' + obj_field + '_' + str(obj_id) + '_pz.npy', pz)
    zp = np.sum(z_arr_to_check * pz)
    print "Ground-based spectroscopic redshift [-99.0 if it does not exist]:", specz
    print "Previous photometric redshift from 3DHST:", photoz
    print "Photometric redshift from min chi2 from this code:", "{:.2}".format(zp_minchi2)
    print "Photometric redshift (weighted) from this code:", "{:.3}".format(zp)

    return zp_minchi2, zp, low_z_lim, upper_z_lim, min_chi2_red, age, tau, (tauv/1.086)

def get_pz_and_plot_photoz(chi2_map, z_arr_to_check, specz, photoz, grismz, low_z_lim, upper_z_lim, obj_id, obj_field):

    # Convert chi2 to likelihood
    likelihood = np.exp(-1 * chi2_map / 2)

    # Normalize likelihood function
    norm_likelihood = likelihood / np.sum(likelihood)

    # Get p(z)
    pz = np.zeros(len(z_arr_to_check))

    for i in range(len(z_arr_to_check)):
        pz[i] = np.sum(norm_likelihood[i])

    return pz

def main():
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Flag to turn on-off emission lines in the fit
    use_emlines = True
    num_filters = 12

    # ------------------------------ Add emission lines to models ------------------------------ #
    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542

    # create fits file to save models with emission lines
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)    

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_lam_grid = model_lam_grid.astype(np.float64)

    total_emission_lines_to_add = 12  # Make sure that this changes if you decide to add more lines to the models
    model_comp_spec_withlines = np.zeros((total_models, len(model_lam_grid) + total_emission_lines_to_add), dtype=np.float64)
    # ----------------------------------- #
    #### DO NOT delete this code block ####
    # ----------------------------------- #
    """
    for j in range(total_models):
        nlyc = float(bc03_all_spec_hdulist[j+1].header['NLYC'])
        metallicity = float(bc03_all_spec_hdulist[j+1].header['METAL'])
        model_lam_grid_withlines, model_comp_spec_withlines[j] = \
        ff.emission_lines(metallicity, model_lam_grid, bc03_all_spec_hdulist[j+1].data, nlyc)
        # Also checked that in every case model_lam_grid_withlines is the exact same
        # SO i'm simply using hte output from the last model.

        # To include the models without the lines as well you will have to make sure that 
        # the two types of models (ie. with and without lines) are on the same lambda grid.
        # I guess you could simply interpolate the model without lines on to the grid of
        # the models wiht lines.

        hdr = fits.Header()
        hdr['NLYC'] = str(nlyc)
        hdr['METAL'] = str(metallicity)
        hdulist.append(fits.ImageHDU(data=model_comp_spec_withlines[j], header=hdr))

    # Save models with emission lines
    hdulist.writeto(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample_withlines.fits', overwrite=True)
    # Also save the model lam grid for models with emission lines
    np.save(figs_dir + 'model_lam_grid_withlines.npy', model_lam_grid_withlines)
    """

    # Read in models with emission lines adn put in numpy array
    bc03_all_spec_hdulist_withlines = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample_withlines.fits')
    model_lam_grid_withlines = np.load(figs_dir + 'model_lam_grid_withlines.npy')
    for q in range(total_models):
        model_comp_spec_withlines[q] = bc03_all_spec_hdulist_withlines[q+1].data

    bc03_all_spec_hdulist_withlines.close()
    del bc03_all_spec_hdulist_withlines

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

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
        dtype=None, names=photometry_names, usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), skip_header=3)

    # Read in Vega spectrum and get it in the appropriate forms
    vega = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/' + 'vega_reference.dat', dtype=None, \
        names=['wav', 'flam'], skip_header=7)

    speed_of_light = 3e18  # in Angstroms per second
    vega_lam = vega['wav']
    vega_spec_flam = vega['flam']
    vega_nu = speed_of_light / vega_lam
    vega_spec_fnu = vega_lam**2 * vega_spec_flam / speed_of_light

    # Lists to loop over
    all_speccats =  [specz_goodsn, specz_goodss]
    all_match_cats = [matched_cat_n]#, matched_cat_s]

    # save lists for comparing after code is done
    id_list = []
    field_list = []
    zspec_list = []
    zphot_list = []
    chi2_list = []
    age_list = []
    tau_list = []
    av_list = []
    my_photoz_list = []
    my_photoz_minchi2_list = []
    my_photoz_lowerr_list = []
    my_photoz_uperr_list = []

    # start looping
    catcount = 0
    galaxy_count = 0
    for cat in all_match_cats:

        for i in xrange(len(cat)):

            # --------------------------------------------- GET OBS DATA ------------------------------------------- #
            current_id = cat['pearsid'][i]

            if catcount == 0:
                current_field = 'GOODS-N'
                spec_cat = specz_goodsn
                phot_cat_3dhst = goodsn_phot_cat_3dhst
            elif catcount == 1: 
                current_field = 'GOODS-S'
                spec_cat = specz_goodss
                phot_cat_3dhst = goodss_phot_cat_3dhst
    
            threed_ra = phot_cat_3dhst['ra']
            threed_dec = phot_cat_3dhst['dec']

            # Get specz and photoz
            specz_idx = np.where(spec_cat['pearsid'] == current_id)[0]

            # For now running this code only on the specz sample
            if len(specz_idx) != 1:
                print "Match not found in specz catalog for ID", current_id, "in", current_field, "... Skipping."
                continue

            if len(specz_idx) == 1:
                current_specz = float(spec_cat['specz'][specz_idx])
                current_photz = float(cat['zphot'][i])
            elif len(specz_idx) == 0:
                current_specz = -99.0
                current_photz = float(cat['zphot'][i])
            else:
                print "Got other than 1 or 0 matches for the specz for ID", current_id, "in", current_field
                print "This much be fixed. Check why it happens. Exiting now."
                sys.exit(0)

            print "Galaxies done so far:", galaxy_count
            print "At ID", current_id, "in", current_field, "with specz and photo-z:", current_specz, current_photz
            print "Total time taken:", time.time() - start, "seconds."
            if galaxy_count > 2:
                break

            # ------------------------------- Match and get photometry data ------------------------------- #
            # find obj ra,dec
            cat_idx = np.where(cat['pearsid'] == current_id)[0]
            if cat_idx.size:
                current_ra = float(cat['pearsra'][cat_idx])
                current_dec = float(cat['pearsdec'][cat_idx])

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
                print "Match not found in Photmetry catalog. Skipping."
                continue

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

            # ------------------------------- Read in filter curves ------------------------------- #
            f435w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f435w')
            f606w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f606w')
            f775w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f775w')
            f850lp_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f850lp')

            f125w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f125w')
            f140w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f140w')
            f160w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f160w')

            # non-HST filter curves
            # IRac wavelengths are in mixrons # convert to angstroms
            uband_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/kpno_mosaic_u.txt', dtype=None, \
                names=['wav', 'trans'], skip_header=14)
            irac1_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac1.txt', dtype=None, \
                names=['wav', 'trans'], skip_header=3)
            irac2_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac2.txt', dtype=None, \
                names=['wav', 'trans'], skip_header=3)
            irac3_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac3.txt', dtype=None, \
                names=['wav', 'trans'], skip_header=3)
            irac4_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac4.txt', dtype=None, \
                names=['wav', 'trans'], skip_header=3)

            irac1_curve['wav'] *= 1e4
            irac2_curve['wav'] *= 1e4
            irac3_curve['wav'] *= 1e4
            irac4_curve['wav'] *= 1e4

            all_filters = [uband_curve, f435w_filt_curve, f606w_filt_curve, f775w_filt_curve, f850lp_filt_curve, \
            f125w_filt_curve, f140w_filt_curve, f160w_filt_curve, irac1_curve, irac2_curve, irac3_curve, irac4_curve]

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
            phot_lam = np.array([3582.0, 4328.2, 5921.1, 7692.4, 9033.1, 12486, 13923, 15369, 
            35500.0, 44930.0, 57310.0, 78720.0])  # angstroms

            # ------------------------------ Now start fitting ------------------------------ #
            # --------- Force dtype for cython code --------- #
            # Apparently this (i.e. for flam_obs and ferr_obs) has  
            # to be done to avoid an obscure error from parallel in joblib --
            # AttributeError: 'numpy.ndarray' object has no attribute 'offset'
            phot_lam = phot_lam.astype(np.float64)
            phot_fluxes_arr = phot_fluxes_arr.astype(np.float64)
            phot_errors_arr = phot_errors_arr.astype(np.float64)

            # ------- Finite photometry values ------- # 
            # Make sure that the photometry arrays all have finite values
            # If any vlues are NaN then throw them out
            phot_fluxes_finite_idx = np.where(np.isfinite(phot_fluxes_arr))[0]
            phot_errors_finite_idx = np.where(np.isfinite(phot_errors_arr))[0]

            phot_fin_idx = reduce(np.intersect1d, (phot_fluxes_finite_idx, phot_errors_finite_idx))

            phot_fluxes_arr = phot_fluxes_arr[phot_fin_idx]
            phot_errors_arr = phot_errors_arr[phot_fin_idx]
            phot_lam = phot_lam[phot_fin_idx]

            all_filters = np.asarray(all_filters)
            all_filters = all_filters[phot_fin_idx]
            num_filters = len(all_filters)

            # ------------- Call actual fitting function ------------- #
            zp_minchi2, zp, zerr_low, zerr_up, min_chi2, age, tau, av = \
            do_photoz_fitting(phot_fluxes_arr, phot_errors_arr, phot_lam, \
                model_lam_grid_withlines, total_models, model_comp_spec_withlines, bc03_all_spec_hdulist, start,\
                current_id, current_field, current_specz, current_photz, all_filters)

            galaxy_count += 1

            # ---------------------------------------------- SAVE PARAMETERS ----------------------------------------------- #
            id_list.append(current_id)
            field_list.append(current_field)
            zspec_list.append(current_specz)
            zphot_list.append(current_photz)
            chi2_list.append(min_chi2)
            age_list.append(age)
            tau_list.append(tau)
            av_list.append(av)
            my_photoz_list.append(zp)
            my_photoz_minchi2_list.append(zp_minchi2)
            my_photoz_lowerr_list.append(zerr_low)
            my_photoz_uperr_list.append(zerr_up)

            # Save files
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_id_list.npy', id_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_field_list.npy', field_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_zspec_list.npy', zspec_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_3dzphot_list.npy', zphot_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_chi2_list.npy', chi2_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_age_list.npy', age_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_tau_list.npy', tau_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_av_list.npy', av_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_list.npy', my_photoz_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_minchi2_list.npy', my_photoz_minchi2_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_lowerr_list.npy', my_photoz_lowerr_list)
            np.save(figs_dir + 'massive-galaxies-figures/my_photoz_uperr_list.npy', my_photoz_uperr_list)

        catcount += 1

    print "Total galaxies considered:", galaxy_count

    # Close HDUs
    bc03_all_spec_hdulist.close()

    # Total time taken
    print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."
    return None

if __name__ == '__main__':
    main()
