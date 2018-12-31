from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.convolution import convolve_fft, convolve, Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo

import os
import sys
import glob
import time
import datetime

#import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
import grid_coadd as gd
import dn4000_catalog as dc
import fast_chi2_jackknife_massive_galaxies as fcjm
import fast_chi2_jackknife as fcj
import refine_redshifts_dn4000 as old_ref
import create_fsps_miles_libraries as ct

def plotspectrum(lam, flam, ferr):

    #gauss_kernel = Gaussian1DKernel(0.9)
    #flam = convolve(flam, gauss_kernel, boundary='extend')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(lam, flam)
    ax.fill_between(lam, flam + ferr, flam - ferr, color='lightgray')

    plt.show()

    return None

def plotlsf(lsf):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(np.arange(len(lsf)), lsf)

    plt.show()

    return None

def plot_interp_and_nointerp_lsf_comparison(modellam, currentspec, lsf, interplsf):

    # plot spectrum to check before and after convolution
    # and also checking between the two lsfs
    fig = plt.figure()
    gs = gridspec.GridSpec(2,2)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    ax1.plot(modellam, currentspec)

    spec_conv_nointerplsf = convolve_fft(currentspec, lsf)
    ax2.plot(modellam, spec_conv_nointerplsf)

    spec_conv_interplsf = convolve_fft(currentspec, interplsf)

    ax3.plot(modellam, currentspec)
    ax4.plot(modellam, spec_conv_interplsf)

    ax1.set_xlim(2000, 6000)
    ax2.set_xlim(2000, 6000)
    ax3.set_xlim(2000, 6000)
    ax4.set_xlim(2000, 6000)

    plt.show()

    return None

def get_model_set():
    """
    All that this function does is it reads in all models 
    with different parameters and puts them into one single
    fits file. Can't use a numpy array instead of a fits
    file because I need a header with info for each extension
    ie. each model spectrum's parameters.

    It is set up right now to always include SSPs with all 
    metallicities and CSPs restricted to solar
    """

    print "Running code to generate set of models for comparison."

    ssp_dir = home + "/Documents/galaxev_bc03_2016update/bc03/Miles_Atlas/Salpeter_IMF/"

    # Find total ages (and their indices in the individual fitfile's extensions) that are to be used in the fits
    example_filename = "bc2003_hr_xmiless_m22_salp_ssp.fits"
    example = fits.open(ssp_dir + example_filename)
    ages = example[2].data
    age_ind = np.where((ages/1e9 > 0.01) & (ages/1e9 < 13))[0]
    total_ages = int(len(age_ind))  # 123 for SSPs and CSPs

    # FITS file where the reduced number of spectra will be saved
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)
    
    for filename in glob.glob(ssp_dir + '*.fits'):        
        h = fits.open(filename)
        modellam = h[1].data

        """
        I'm not restricting the model wav range to something like 100 A to 10 micron
        which would be more useful because most of the model points are withing this 
        range anyway so it doesn't help save much space.
        """

        # Open corresponding *.*color files.
        # These files contain useful info avbout the spectra which I will 
        # put in the header for each fits extension containing a spectrum.
        # *.1color gives (U-B) and (B-V) colors 
        # *.2color gives (V-J) color if you need to construct the UVJ diagram 
        # *.3color gives NLyc
        # *.4color gives M_stellar and M_galaxy (where M_galaxy = M_stellar + M_gas)
        onecolor = np.genfromtxt(filename.replace('.fits','.1color'), dtype=None, \
            names=['log_age','UB_col','BV_col'], usecols=(0,13,14), skip_header=30)
        twocolor = np.genfromtxt(filename.replace('.fits','.2color'), dtype=None, \
            names=['log_age','VJ_col'], usecols=(0,6), skip_header=30)
        threecolor = np.genfromtxt(filename.replace('.fits','.3color'), dtype=None, \
            names=['log_age','log_nlyc'], usecols=(0,5), skip_header=30)
        fourcolor = np.genfromtxt(filename.replace('.fits','.4color'), dtype=None, \
            names=['log_age','ms','mgal'], usecols=(0,5,8), skip_header=30)
    
        # define and initialize numpy array
        current_model_set_ssp = np.zeros([total_ages, len(modellam)], dtype=np.float64)
        for i in range(total_ages):

            # spectrum to save
            current_model_set_ssp[i] = h[age_ind[i]+3].data

            # Other info to put in header 
            # ---- Age
            hdr = fits.Header()
            current_log_age = np.log10(ages[age_ind[i]])
            hdr['LOG_AGE'] = str(current_log_age)

            # ---- Rate of Lyman continuum photons [# per second]
            # Get idx corresponding to age first. This is valid for all color files.
            color_idx = np.argmin(abs(onecolor['log_age'] - current_log_age))
            nlyc = 10**(threecolor['log_nlyc'][color_idx])
            #print "Current Log(age):", current_log_age
            #print "Rate of Lyman continuum photons:", nlyc, "s^-1"
            hdr['NLYC'] = str(nlyc)

            # ---- Colors
            # ---- U-B
            ub_col = onecolor['UB_col'][color_idx]
            hdr['UB_col'] = str(ub_col)
            # ---- B-V
            bv_col = onecolor['BV_col'][color_idx]
            hdr['BV_col'] = str(bv_col)
            # ---- V-J
            vj_col = twocolor['VJ_col'][color_idx]
            hdr['VJ_col'] = str(vj_col)

            # ---- Masses
            # ---- Stellar mass
            mstar = fourcolor['ms'][color_idx]
            hdr['ms'] = str(mstar)

            # ---- Galaxy mass = Stellar mass + Gas mass
            mgalaxy = fourcolor['mgal'][color_idx]
            hdr['mgal'] = str(mgalaxy)

            # ---- Metallicity
            if 'm22' in filename:
                metal_val = 0.0001
            elif 'm32' in filename:
                metal_val = 0.0004
            elif 'm42' in filename:
                metal_val = 0.004
            elif 'm52' in filename:
                metal_val = 0.008
            elif 'm62' in filename:
                metal_val = 0.02
            elif 'm72' in filename:
                metal_val = 0.05

            hdr['METAL'] = str(metal_val)

            # Append
            hdulist.append(fits.ImageHDU(data=current_model_set_ssp[i], header=hdr))

        h.close()

    # ---------- Currently restricted to solar metallicity ----------- #
    # Check directories for the models
    cspout = "/Volumes/Bhavins_backup/bc03_models_npy_spectra/cspout_2016updated_galaxev/"
    # This is if working on the laptop. 
    # Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(cspout):
        # On firstlight
        cspout = home + '/Documents/galaxev_bc03_2016update/bc03/src/cspout_2016updated_galaxev/'
        if not os.path.isdir(cspout):
            print "Model directory not found. Exiting..."
            sys.exit(0)

    # Get valid ages i.e. between 10 Myr and 13 Gyr
    # Find total ages (and their indices in the individual fitfile's extensions) that are to be used in the fits
    example_filename = "bc2003_hr_m62_tauV0_csp_tau100_salp.fits"
    metalfolder = "m62/"
    model_dir = cspout + metalfolder
    example = fits.open(model_dir + example_filename)
    ages = example[2].data
    age_ind = np.where((ages/1e9 > 0.01) & (ages/1e9 < 13))[0]
    total_ages = int(len(age_ind))  # 123 for SSPs and CSPs

    # model grid parameters
    # they are being redefined here to be able to identify 
    # the different filenames which were set up in the exact same way.
    # I've restricted tauV, tau, lambda, and ages in distinguishing spectra
    tauVarr = np.arange(0.0, 3.0, 0.2)
    logtauarr = np.arange(-2, 2, 0.2)
    # This tauVarr and logtauarr are a bit more (2x) coarsely sampled than the 
    # generated models to be less computationally expensive. 
    # I'm choosing every second value as compared to the original grid.
    tauarr = np.empty(len(logtauarr)).astype(np.str)
    
    for i in range(len(logtauarr)):
        tauarr[i] = str(int(float(str(10**logtauarr[i])[0:6])*10000))

    # Read in each individual spectrum and append it to the FITS HDU List
    for tauVarrval in tauVarr:
        for tauval in tauarr:
            filename = model_dir + 'bc2003_hr_m62_tauV' + str(int(tauVarrval*10)) + '_csp_tau' + tauval + '_salp.fits'

            # define and initialize fits file
            h = fits.open(filename)
            modellam = h[1].data

            # Open corresponding *.3color, *.4color, and *.1color files
            # See comment in the SSP part above for explanation
            onecolor = np.genfromtxt(filename.replace('.fits','.1color'), dtype=None, \
                names=['log_age','UB_col','BV_col'], usecols=(0,13,14), skip_header=30)
            twocolor = np.genfromtxt(filename.replace('.fits','.2color'), dtype=None, \
                names=['log_age','VJ_col'], usecols=(0,6), skip_header=30)
            threecolor = np.genfromtxt(filename.replace('.fits','.3color'), dtype=None, \
                names=['log_age','log_nlyc'], usecols=(0,5), skip_header=30)
            fourcolor = np.genfromtxt(filename.replace('.fits','.4color'), dtype=None, \
                names=['log_age','ms','mgal'], usecols=(0,5,8), skip_header=30)

            current_model_set_csp = np.zeros([total_ages, len(modellam)], dtype=np.float64)
            for i in range(total_ages):
                # Spectrum to save
                current_model_set_csp[i] = h[age_ind[i]+3].data

                # Add info to header
                hdr = fits.Header()

                # ---- Age 
                current_log_age = np.log10(ages[age_ind[i]])
                hdr['LOG_AGE'] = str(current_log_age)

                # ---- Metallicity
                metal_val = 0.02
                hdr['METAL'] = str(metal_val)

                # ---- Tau and TauV
                hdr['TAU_GYR'] = str(float(tauval)/1e4)
                hdr['TAUV'] = str(tauVarrval)

                # ---- Rate of Lyman continuum photons [# per second]
                # Get idx corresponding to age first. This is valid for all color files.
                color_idx = np.argmin(abs(onecolor['log_age'] - current_log_age))
                nlyc = 10**(threecolor['log_nlyc'][color_idx])
                #print "Current Log(age):", current_log_age
                #print "Rate of Lyman continuum photons:", nlyc, "s^-1"
                hdr['NLYC'] = str(nlyc)

                # ---- Colors
                # ---- U-B
                ub_col = onecolor['UB_col'][color_idx]
                hdr['UB_col'] = str(ub_col)
                # ---- B-V
                bv_col = onecolor['BV_col'][color_idx]
                hdr['BV_col'] = str(bv_col)
                # ---- V-J
                vj_col = twocolor['VJ_col'][color_idx]
                hdr['VJ_col'] = str(vj_col)

                # ---- Masses
                # ---- Stellar mass
                mstar = fourcolor['ms'][color_idx]
                hdr['ms'] = str(mstar)

                # ---- Galaxy mass = Stellar mass + Gas mass
                mgalaxy = fourcolor['mgal'][color_idx]
                hdr['mgal'] = str(mgalaxy)

                # Append
                hdulist.append(fits.ImageHDU(data=current_model_set_csp[i], header=hdr))

            h.close()

    final_fitsname = 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits'
    hdulist.writeto(figs_dir + final_fitsname, overwrite=True)
    print "All models (SSP + CSP_solar) saved to one fits file in directory:", figs_dir
    print "Exiting."
    sys.exit(0)

    return None

def do_model_modifications(object_lam_obs, model_lam_grid, model_comp_spec, resampling_lam_grid, total_models, lsf, z):

    # Before fitting
    # 0. get lsf and models (supplied as arguments to this function)
    # 1. redshift the models
    # 2. convolve the models with the lsf
    # 3. resample the models

    # create empty array in which final modified models will be stored
    model_comp_spec_modified = np.empty([total_models, len(resampling_lam_grid)])

    # redshift lambda grid for model 
    # this is the lambda grid at the model's native resolution
    model_lam_grid_z = model_lam_grid * (1 + z)

    # redshift flux
    model_comp_spec = model_comp_spec / (1 + z)    # figure added to be able to debug

    # ---------------- Mask potential emission lines ----------------- #
    # Will mask one point on each side of line center i.e. approx 80 A masked
    # These are all vacuum wavelengths
    oiii_4363 = 4364.44
    oiii_5007 = 5008.24
    oiii_4959 = 4960.30
    hbeta = 4862.69
    hgamma = 4341.69
    oii_3727 = 3728.5  
    # these two lines (3727 and 3729) are so close to each other 
    # that the line will always blend in grism spectra. 
    # avg wav of the two written here

    # Set up line mask 
    line_mask = np.zeros(len(resampling_lam_grid))

    # Get redshifted wavelengths and mask
    oii_3727_idx = np.argmin(abs(resampling_lam_grid - oii_3727*(1 + z)))
    oiii_5007_idx = np.argmin(abs(resampling_lam_grid - oiii_5007*(1 + z)))
    oiii_4959_idx = np.argmin(abs(resampling_lam_grid - oiii_4959*(1 + z)))
    oiii_4363_idx = np.argmin(abs(resampling_lam_grid - oiii_4363*(1 + z)))

    line_mask[oii_3727_idx-1 : oii_3727_idx+2] = 1.0
    line_mask[oiii_5007_idx-1 : oiii_5007_idx+2] = 1.0
    line_mask[oiii_4959_idx-1 : oiii_4959_idx+2] = 1.0
    line_mask[oiii_4363_idx-1 : oiii_4363_idx+2] = 1.0

    for k in range(total_models):

        # ALL OF THE PLOTTING CODE IS IMPORTANT FOR DEBUGGING. 
        # DO NOT DELETE. UNCOMMENT IF NOT NEEDED.
        # figure added to be able to debug
        #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        #ax1.plot(model_lam_grid_z, model_comp_spec[k])
        #ax1.set_xlim(5000, 10500)

        #model_comp_spec[k] = convolve_fft(model_comp_spec[k], lsf)#, boundary='extend')
        # seems like boundary='extend' is not implemented 
        # currently for convolve_fft(). It works with convolve() though.

        #ax2.plot(model_lam_grid_z, model_comp_spec[k])
        #ax2.set_xlim(5000, 10500)

        # using a broader lsf just to see if that can do better
        interppoints = np.linspace(0, len(lsf), int(len(lsf)*10))
        # just making the lsf sampling grid longer # i.e. sampled at more points 
        broad_lsf = np.interp(interppoints, xp=np.arange(len(lsf)), fp=lsf)
        temp_broadlsf_model = convolve_fft(model_comp_spec[k], broad_lsf)

        # resample to object resolution
        #resampled_flam = np.zeros((len(resampling_lam_grid)))
        resampled_flam_broadlsf = np.zeros((len(resampling_lam_grid)))

        for i in range(len(resampling_lam_grid)):

            if i == 0:
                lam_step_high = resampling_lam_grid[i+1] - resampling_lam_grid[i]
                lam_step_low = lam_step_high
            elif i == len(resampling_lam_grid) - 1:
                lam_step_low = resampling_lam_grid[i] - resampling_lam_grid[i-1]
                lam_step_high = lam_step_low
            else:
                lam_step_high = resampling_lam_grid[i+1] - resampling_lam_grid[i]
                lam_step_low = resampling_lam_grid[i] - resampling_lam_grid[i-1]

            new_ind = np.where((model_lam_grid_z >= resampling_lam_grid[i] - lam_step_low) & \
                (model_lam_grid_z < resampling_lam_grid[i] + lam_step_high))[0]
            #resampled_flam[i] = np.mean(model_comp_spec[k][new_ind])

            resampled_flam_broadlsf[i] = np.mean(temp_broadlsf_model[new_ind])

        # Now mask the flux at these wavelengths using the mask generated before the for loop
        resampled_flam_broadlsf = ma.array(resampled_flam_broadlsf, mask=line_mask)

        model_comp_spec_modified[k] = resampled_flam_broadlsf

        #ax3.plot(resampling_lam_grid, resampled_flam_broadlsf)
        #ax3.set_xlim(5000, 10500)

        #ax4.plot(resampling_lam_grid, model_comp_spec_modified[k])
        #ax4.set_xlim(5000, 10500)

        #plt.show()
        #plt.cla()
        #plt.clf()
        #plt.close()

    return model_comp_spec_modified

def get_best_fit_model(flam, ferr, object_lam_grid, model_lam_grid, model_comp_spec, current_z, model_spec_hdu):

    # chop the model to be consistent with the objects lam grid
    model_lam_grid_indx_low = np.argmin(abs(model_lam_grid - object_lam_grid[0]))
    model_lam_grid_indx_high = np.argmin(abs(model_lam_grid - object_lam_grid[-1]))
    model_spec_in_objlamgrid = model_comp_spec[:, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # make sure that the arrays are the same length
    if int(model_spec_in_objlamgrid.shape[1]) != len(object_lam_grid):
        print "Arrays of unequal length. Must be fixed before moving forward. Exiting..."
        sys.exit(0)

    # create chi2 array
    # i.e. there is one chi2 value for each model 
    # therefore the chi2 array has as many elements as there are models to compare
    # the vertical scaling factor, alpha, is computed with a formula 
    # determined analytically. There is also one alpha value for each model.
    #num_models = int(model_spec_in_objlamgrid.shape[0])
    #chi2 = np.zeros(num_models, dtype=np.float64)
    alpha = np.sum(flam * model_spec_in_objlamgrid / (ferr**2), axis=1) / np.sum(model_spec_in_objlamgrid**2 / ferr**2, axis=1)
    chi2 = np.sum(((flam - (alpha * model_spec_in_objlamgrid.T).T) / ferr)**2, axis=1)

    # This is to get only physical ages
    sortargs = np.argsort(chi2)
    for k in range(len(chi2)):
        best_age = float(model_spec_hdu[sortargs[k] + 1].header['LOG_AGE'])
        # now check if the best fit model is an ssp or csp 
        # only the csp models have tau and tauV parameters
        # so if you try to get these keywords for the ssp fits files
        # it will fail with a KeyError
        if 'TAU_GYR' in list(model_spec_hdu[sortargs[k] + 1].header.keys()):
            best_tau = float(model_spec_hdu[sortargs[k] + 1].header['TAU_GYR'])
            best_tauv = float(model_spec_hdu[sortargs[k] + 1].header['TAUV'])
        else:
            # if the best fit model is an SSP then assign -99.0 to tau and tauV
            best_tau = -99.0
            best_tauv = -99.0
        age_at_z = cosmo.age(current_z).value * 1e9 # in yr
        if (best_age < np.log10(age_at_z)) & (best_age > 9 + np.log10(0.1)):
            #fitages.append(best_age)
            #fitmetals.append(model_spec_hdu[sortargs[k] + 1].header['METAL'])
            #best_exten.append(sortargs[k] + 1)
            #bestalpha_plot.append(alpha[sortargs[k]])
            #old_chi2.append(chi2[sortargs[k]])
            current_best_fit_model = model_spec_in_objlamgrid[sortargs[k]]
            current_best_fit_model_whole = model_comp_spec[sortargs[k]]
            break

    return current_best_fit_model, current_best_fit_model_whole, alpha[sortargs[k]], np.nanmin(chi2), best_age, best_tau, best_tauv

def shift_in_wav_get_new_z(flam, ferr, lam_obs, model_lam_grid, previous_z, spec_elem_model_fit, best_fit_model_whole):

    # now shift in wavelength space to get best fit on wavelength grid and correspongind redshift
    low_lim_for_comp = 5000
    high_lim_for_comp = 10500

    start_low_indx = np.argmin(abs(model_lam_grid - low_lim_for_comp))
    start_high_indx = start_low_indx + spec_elem_model_fit - 1
    # these starting low and high indices are set up in a way that the dimensions of
    # best_fit_model_in_objlamgrid and flam and ferr are the same to be able to 
    # compute alpha and chi2 below.
    # the -1 is for ...?

    chi2_redshift_arr = []
    count = 0 
    while 1:
        current_low_indx = start_low_indx + count
        current_high_indx = start_high_indx + count

        if current_high_indx == len(model_lam_grid):
            break

        if model_lam_grid[current_high_indx] >= high_lim_for_comp:
            break

        # do the fitting again for each shifted lam grid
        best_fit_model_in_objlamgrid = best_fit_model_whole[current_low_indx:current_high_indx+1]

        alpha = np.sum(flam * best_fit_model_in_objlamgrid / (ferr**2)) / np.sum(best_fit_model_in_objlamgrid**2 / ferr**2)
        chi2 = np.sum(((flam - (alpha * best_fit_model_in_objlamgrid)) / ferr)**2)

        chi2_redshift_arr.append(chi2)
        #print "In shifting loop. Iteration #", count, current_low_indx, current_high_indx, len(model_lam_grid), chi2
        # Above line useful for debugging. Do not remove. Just uncomment.

        count += 1

    chi2_redshift_arr = np.asarray(chi2_redshift_arr)
    new_chi2_minindx = np.argmin(chi2_redshift_arr)

    new_lam_grid = model_lam_grid[start_low_indx+new_chi2_minindx : start_high_indx+new_chi2_minindx+1]

    new_z_minchi2 = (lam_obs[0] * (1 + previous_z) / new_lam_grid[0]) - 1

    # ---------
    #print chi2_redshift_arr
    #print new_chi2_minindx
    #print chi2_redshift_arr[new_chi2_minindx]

    #print model_lam_grid
    #print lam_obs
    #print new_lam_grid

    #print new_z_minchi2
    # All printing lines useful for debugging. Do not remove. Just uncomment.

    return new_z_minchi2

def do_fitting(flam, ferr, object_lam_obs, lsf, starting_z, resampling_lam_grid, \
    model_lam_grid, total_models, model_comp_spec, model_spec_hdu):

    """
    flam, ferr, object_lam_grid are in observed wavelength space. Need to deredshift
    before fitting.

    # first find best fit by deredshifting the grism spectrum using the old redshift
    # iterate 
    """

    # Get resampled spectra for bootstrap based 
    # on errors and assuming gaussian errors.
    # Resampling is done in observed frame.
    num_samp_to_draw = 1

    if num_samp_to_draw > 1:
        resampled_spec = get_resamples(flam, ferr, num_samp_to_draw)

    # loop over bootstrap runs
    for i in range(int(num_samp_to_draw)):
 
        # find if you're only doing one run or multiple bootstrap runs
        if num_samp_to_draw > 1:
            flam = resampled_spec[i]

        print "Starting at redshift", starting_z

        previous_z = starting_z
        num_iter = 0
        while 1:
            
            print "\n", "Currently testing redshift:", previous_z

            # modify the model to be able to compare with data
            model_comp_spec_modified = \
            do_model_modifications(object_lam_obs, model_lam_grid, model_comp_spec, resampling_lam_grid, total_models, lsf, previous_z)
            print "Model mods done at current z:", previous_z, "\n", "Total time taken up to now --", time.time() - start, "seconds."

            # now get teh best fit model spectrum
            best_fit_model_in_objlamgrid, best_fit_model_whole, bestalpha, min_chi2, age, tau, tauv = \
            get_best_fit_model(flam, ferr, object_lam_obs, resampling_lam_grid, \
                model_comp_spec_modified, previous_z, model_spec_hdu)

            print "Current min Chi2:", "{:.2}".format(min_chi2)
            print "Current best fit log(age [yr]):", "{:.2}".format(age)
            print "Current best fit Tau [Gyr]:", "{:.2}".format(tau)
            print "Current best fit Tau_V:", tauv
            #plot_fit_and_residual(object_lam_obs, flam, ferr, best_fit_model_in_objlamgrid, bestalpha)

            #print "Current best fit model parameters are:"
            #print "Age:"
            #print "Metallicity: Solar (this was kept fixed)"
            #print "Tau (i.e. exponential SFH time scale):"
            #print "Tau_V:", , "A_V:",

            # now shift in wavelength range and get new_z
            new_z = \
            shift_in_wav_get_new_z(flam, ferr, object_lam_obs, resampling_lam_grid, previous_z, \
                len(best_fit_model_in_objlamgrid), best_fit_model_whole)
            print "Current Old and New redshifts", previous_z, new_z

            # Need to add code to stop it from getting worse
            # Can keep track of the minimum chi2 and if it increases after any run 
            # then you know you've gone too far in that direction.

            # Stop if maximum iterations are reached
            num_iter += 1
            if num_iter >= 20:
                print "Maximum iterations reached. Current grism redshift is", new_z
                print "Exiting out of iterative loop."
                break 

            # if the relative error is less than 1% then stop
            if abs(new_z - previous_z) < 0.01 * new_z:
                break

            previous_z = new_z

    plot_fit_and_residual(object_lam_obs, flam, ferr, best_fit_model_in_objlamgrid, bestalpha)

    print "New refined redshift", new_z

    return None

def plot_fit_and_residual(object_lam_obs, flam, ferr, best_fit_model_in_objlamgrid, bestalpha):

    #gauss_kernel = Gaussian1DKernel(1.6)
    #best_fit_model_in_objlamgrid = convolve(best_fit_model_in_objlamgrid, gauss_kernel, boundary='extend')

    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[:8,:])
    ax2 = fig.add_subplot(gs[8:,:])

    ax1.plot(object_lam_obs, flam, ls='-', color='k')
    ax1.plot(object_lam_obs, bestalpha*best_fit_model_in_objlamgrid, ls='-', color='r')
    ax1.fill_between(object_lam_obs, flam + ferr, flam - ferr, color='lightgray')

    resid_fit = (flam - bestalpha*best_fit_model_in_objlamgrid) / ferr
    ax2.plot(object_lam_obs, resid_fit, ls='-', color='k')

    plt.show()

    return None

def test_plot():

    # Open spectra
    #spec_hdu_old = fits.open(pears_datadir + 'h_pears_s_id65620.fits')
    spec_hdu_old = fits.open(pears_datadir + 'h_pears_s_id61447.fits')
    spec_hdu_young = fits.open(pears_datadir + 'h_pears_n_id40498.fits')
    #spec_hdu_old = fits.open(pears_datadir + 'h_pears_n_id83499.fits')

    # assign arrays
    flam_old = spec_hdu_old[1].data['FLUX']
    ferr_old = spec_hdu_old[1].data['FERROR']
    contam_old = spec_hdu_old[1].data['CONTAM']
    lam_old = spec_hdu_old[1].data['LAMBDA']
    
    flam_young = spec_hdu_young[1].data['FLUX']
    ferr_young = spec_hdu_young[1].data['FERROR']
    contam_young = spec_hdu_young[1].data['CONTAM']
    lam_young = spec_hdu_young[1].data['LAMBDA']

    # Subtract contamination
    flam_old_consub = flam_old - contam_old
    flam_young_consub = flam_young - contam_young

    # Select only reliable data and plot. Select only data within 6000A to 9500A
    # old pop
    lam_low_idx = np.argmin(abs(lam_old - 6000))
    lam_high_idx = np.argmin(abs(lam_old - 9500))
    
    lam_old = lam_old[lam_low_idx:lam_high_idx+1]
    flam_old_consub = flam_old_consub[lam_low_idx:lam_high_idx+1]
    ferr_old = ferr_old[lam_low_idx:lam_high_idx+1]
    
    # young pop
    lam_low_idx = np.argmin(abs(lam_young - 6000))
    lam_high_idx = np.argmin(abs(lam_young - 9500))
    
    lam_young = lam_young[lam_low_idx:lam_high_idx+1]
    flam_young_consub = flam_young_consub[lam_low_idx:lam_high_idx+1]

    # Plot
    if verbose:
        #plotspectrum(lam_young, flam_young_consub, ferr_young)
        plotspectrum(lam_old, flam_old_consub, ferr_old)

    return None

def get_mock_spec():
    """
    Choose and return a mock spectrum.
    """

    try_solar = False
    # I've added this flag to be able to test with a mock model spectrum that is outside of the current
    # available set of models for comparison. i.e. its trying to see that if you had a galaxy
    # spectrum that wasn't matched well by any of the templates then how good of a match will the code 
    # come up with. This is because the current set of models for comparison are restricted to solar
    # metallicity. Turning this flag off (i.e. False) will supply a mock spectrum drawn from a model
    # that does not have solar metallicity.
    # You will have to change the model and metallicty by hand in the else part of the if statement
    # below. This is where it looks for a different mock spectrum.

    # I just chose some random file from the model directory
    if try_solar:
        # first check where the models are
        model_dir = '/Volumes/Bhavins_backup/bc03_models_npy_spectra/m62/'
        # this is if working on the laptop. Then you must be using the external hard drive where the models are saved.
        if not os.path.isdir(model_dir):
            model_dir = home + '/Documents/GALAXEV_BC03/bc03/src/cspout_new/m62/'  # this path only exists on firstlight
        else:
            print "Model files not found. Exiting..."
            sys.exit(0)

        example_allspectra = np.load(model_dir + 'bc2003_hr_m62_tauV10_csp_tau1995_salp_allspectra.npy')
        example_lamgrid = np.load(model_dir + 'bc2003_hr_m62_tauV10_csp_tau1995_salp_lamgrid.npy')
        example_ages = np.load(model_dir + 'bc2003_hr_m62_tauV10_csp_tau1995_salp_ages.npy')
    else:
        model_dir = home + '/Documents/GALAXEV_BC03/bc03/src/cspout_new/m52/'
        single_csp_set_dir = home + '/Documents/GALAXEV_BC03/'
        example_allspectra = np.load(single_csp_set_dir + 'bc2003_hr_m32_tauV10_csp_tau200000_salp_allspectra.npy')
        example_lamgrid = np.load(single_csp_set_dir + 'bc2003_hr_m32_tauV10_csp_tau200000_salp_lamgrid.npy')
        example_ages = np.load(single_csp_set_dir + 'bc2003_hr_m32_tauV10_csp_tau200000_salp_ages.npy')

    # Just make sure to pick a valid age
    age_ind = np.where((example_ages/1e9 < 8) & (example_ages/1e9 > 0.1))[0]
    total_ages = int(len(age_ind))  # 57 for both SSP and CSP
    # for refrence example_ages[age_ind[30]] is  ~2.1 Gyr
    #print example_ages[age_ind]
    #print total_ages
    #print "Age value chosen,", "{:.3}".format(example_ages[age_ind[20]])

    # read in spectrum
    # define mock redshift first
    mock_redshift = 1.08
    example_lamgrid_z = example_lamgrid * (1 + mock_redshift)

    # now redshift model
    example_lamgrid_z_idx = np.where((example_lamgrid_z >= 6000) & (example_lamgrid_z <= 9500))[0]
    flam_obs = example_allspectra[age_ind[20]] / (1 + mock_redshift)

    # now get only the part of the model within the valid wavelengths
    lam_obs = example_lamgrid_z[example_lamgrid_z_idx]
    flam_obs = flam_obs[example_lamgrid_z_idx]

    # Now use a fake LSF and also resample the spectrum to the grism resolution
    mock_lsf = Gaussian1DKernel(15.0)
    flam_obs = convolve(flam_obs, mock_lsf, boundary='extend')

    # resample 
    mock_resample_lam_grid = np.linspace(6000, 9500, 88)
    resampled_flam = np.zeros((len(mock_resample_lam_grid)))
    for i in range(len(mock_resample_lam_grid)):

        if i == 0:
            lam_step_high = mock_resample_lam_grid[i+1] - mock_resample_lam_grid[i]
            lam_step_low = lam_step_high
        elif i == len(mock_resample_lam_grid) - 1:
            lam_step_low = mock_resample_lam_grid[i] - mock_resample_lam_grid[i-1]
            lam_step_high = lam_step_low
        else:
            lam_step_high = mock_resample_lam_grid[i+1] - mock_resample_lam_grid[i]
            lam_step_low = mock_resample_lam_grid[i] - mock_resample_lam_grid[i-1]

        new_ind = np.where((lam_obs >= mock_resample_lam_grid[i] - lam_step_low) & \
            (lam_obs < mock_resample_lam_grid[i] + lam_step_high))[0]
        resampled_flam[i] = np.median(flam_obs[new_ind])

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
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    ###ax.plot(lam_obs, flam_obs)  # Use this line if you're plotting before resampling
    ###ax.fill_between(lam_obs, flam_obs + ferr_obs, flam_obs - ferr_obs, color='lightgray')  
    ### Use these above lines if you're plotting before resampling
    #ax.plot(mock_resample_lam_grid, flam_obs)
    #ax.fill_between(mock_resample_lam_grid, flam_obs + ferr_obs, flam_obs - ferr_obs, color='lightgray')
    #plt.show()

    return mock_resample_lam_grid, flam_obs, ferr_obs, mock_lsf

def test_mock_spec():
    """
    Testing module. The structure here is exactly the same as the main code.
    """

    # try fitting a mock spectrum which you know has a match in the model grid
    lam_obs, flam_obs, ferr_obs, mock_lsf = get_mock_spec()

    # read in entire model set
    bc03_all_spec = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')

    # prep for fitting
    total_models = 34542  # if ssp + csp

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    # first check where the models are
    model_dir = '/Volumes/Bhavins_backup/bc03_models_npy_spectra/m62/'
    # this is if working on the laptop. Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(model_dir):
        model_dir = home + '/Documents/GALAXEV_BC03/bc03/src/cspout_new/m62/'  # this path only exists on firstlight
        if not os.path.isdir(model_dir):
            print "Model files not found. Exiting..."
            sys.exit(0)

    example_filename_lamgrid = 'bc2003_hr_m62_tauV0_csp_tau100_salp_lamgrid.npy'
    model_lam_grid = np.load(model_dir + example_filename_lamgrid)
    model_comp_spec = np.zeros([total_models, len(model_lam_grid)], dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec[j+1].data

    # total run time up to now
    print "All models put in numpy array. Total time taken up to now --", time.time() - start, "seconds."

    # now call the actual fitting function
    mock_redshift_estimate = 1.0  # this behaves like the photo-z

    # extend lam_grid to be able to move the lam_grid later 
    avg_dlam = old_ref.get_avg_dlam(lam_obs)

    lam_low_to_insert = np.arange(5000, lam_obs[0], avg_dlam)
    lam_high_to_append = np.arange(lam_obs[-1] + avg_dlam, 10500, avg_dlam)

    resampling_lam_grid = np.insert(lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    # call actual fitting function
    do_fitting(flam_obs, ferr_obs, lam_obs, mock_lsf, mock_redshift_estimate, resampling_lam_grid, \
        model_lam_grid, total_models, model_comp_spec, bc03_all_spec)

    # total time
    print "Total time taken up to now --", time.time() - start, "seconds."
    sys.exit(0)

    return None

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    get_model_set()
    sys.exit(0)

    # -------------------------------------- TESTING WITH A MOCK SPECTRUM ----------------------------------------- #
    # UNCOMMENT IF YOU'RE DONE TESTING
    #test_mock_spec()

    # --------------------------------------------- GET OBS DATA ------------------------------------------- #
    # if this is True then the code will show you all kinds of plots and print info to the screen
    verbose = False

    # start fitting
    # get original photo-z first
    current_id = 43997
    current_field = 'GOODS-S'
    redshift = 0.64
    # for 61447 GOODS-S
    # 0.84 Ferreras+2009  # candels 0.976 # 3dhst 0.9198
    # for 65620 GOODS-S
    # 0.97 Ferreras+2009  # candels 0.972 # 3dhst 0.9673 # spec-z 0.97
    lam_em, flam_em, ferr_em, specname, pa_chosen, netsig_chosen = gd.fileprep(current_id, redshift, current_field)
    d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

    # the original function (fileprep in grid_coadd) will give quantities in rest frame
    # but I need these to be in the observed frame so I will redshift them again.
    # It gives them in the rest frame because getting the d4000 and the average 
    # lambda separation needs to be figured out in the rest frame.
    flam_obs = flam_em / (1 + redshift)
    ferr_obs = ferr_em / (1 + redshift)
    lam_obs = lam_em * (1 + redshift)

    # plot to check # Line useful for debugging. Do not remove. Just uncomment.
    #plotspectrum(lam_obs, flam_obs, ferr_obs)

    if verbose:
        print current_field, current_id, pa_chosen, netsig_chosen  # GOODS-S 65620 PA200 657.496906164
        print d4000  # 1.69

    # ---------------------------------------------- MODELS ----------------------------------------------- #
    # put all models into one single fits file
    #get_model_set()

    # read in entire model set
    bc03_all_spec = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')

    # Read in LSF
    if current_field == 'GOODS-N':
        lsf_filename = lsfdir + "north_lsfs/" + "n" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"
    elif current_field == 'GOODS-S':
        lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"

    # read in LSF file
    lsf = np.loadtxt(lsf_filename)
    #interplsf = fcjm.get_interplsf(current_id, redshift, current_field, pa_chosen)
    #plotlsf(lsf)
    #plotlsf(interplsf)

    # ---------------------------------------------- FITTING ----------------------------------------------- #
    #total_models = fcj.get_total_extensions(bc03_all_spec)
    # this function gives the total number of extensions minus the zeroth extension
    # I used the above line to get this number which is hard coded in now.
    #print "Total models for comparison:", total_models
    total_models = 34542
    # I've hard coded the number in because the function to figure out the total
    # number of extensions takes way too long to run (approx 2-3 mins).

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    # first check where the models are
    #model_dir = '/Volumes/Bhavins_backup/bc03_models_npy_spectra/m62/'
    ## this is if working on the laptop. Then you must be using the external hard drive where the models are saved.
    #if not os.path.isdir(model_dir):
    #    model_dir = home + '/Documents/GALAXEV_BC03/bc03/src/cspout_new/m62/'  # this path only exists on firstlight
    #    if not os.path.isdir(model_dir):
    #        print "Model files not found. Exiting..."
    #        sys.exit(0)

    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_comp_spec = np.zeros([total_models, len(model_lam_grid)], dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec[j+1].data

    # total run time up to now
    print "All models put in numpy array. Total time taken up to now --", time.time() - start, "seconds."

    # now call the actual fitting function
    # extend lam_grid to be able to move the lam_grid later 
    avg_dlam = old_ref.get_avg_dlam(lam_obs)

    lam_low_to_insert = np.arange(5000, lam_obs[0], avg_dlam)
    lam_high_to_append = np.arange(lam_obs[-1] + avg_dlam, 10500, avg_dlam)

    resampling_lam_grid = np.insert(lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    # call actual fitting function
    do_fitting(flam_obs, ferr_obs, lam_obs, lsf, redshift, resampling_lam_grid, \
        model_lam_grid, total_models, model_comp_spec, bc03_all_spec)

    # total time
    print "Total time taken up to now --", time.time() - start, "seconds."
    sys.exit(0)

