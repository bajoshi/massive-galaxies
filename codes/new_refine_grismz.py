from __future__ import division

import numpy as np
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

def plotspectrum(lam, flam):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(lam, flam)

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
    """
    
    # currently restricted to solar metallicity
    # first check where the models are
    model_dir = '/Volumes/Bhavins_backup/bc03_models_npy_spectra/m62/'
    # this is if working on the laptop. Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(model_dir):
        model_dir = home + '/Documents/GALAXEV_BC03/bc03/src/cspout_new/m62/'  # this path only exists on firstlight
    else:
        print "Model files not found. Exiting..."
        sys.exit(0)

    # ONly using SSP for now
    # get valid ages i.e. between 100 Myr and 8 Gyr
    # Find total ages (and their indices in the individual fitfile's extensions) that are to be used in the fits
    example_filename='bc2003_hr_m62_tauV0_csp_tau100_salp_ages.npy'
    ages = np.load(model_dir + example_filename)
    age_ind = np.where((ages/1e9 < 8) & (ages/1e9 > 0.1))[0]
    total_ages = int(len(age_ind))  # 57 for both SSP and CSP

    # FITS file where the model number of spectra will be saved
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)

    # model grid parameters
    # they are being redefined here to be able to identify 
    # the different filenames which were set up in the exact same way.
    # I've restricted tauV, tau, lambda, and ages in distinguishing spectra
    tauVarr = np.arange(0.0, 2.0, 0.1)
    logtauarr = np.arange(-2, 2, 0.2)
    tauarr = np.empty(len(logtauarr)).astype(np.str)
    
    for i in range(len(logtauarr)):
        tauarr[i] = str(int(float(str(10**logtauarr[i])[0:6])*10000))

    # you'll need the following block if you ever use the entire metallicty range
    """
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
    """

    # Read in each individual spectrum and convolve it first and then chop and resample it
    # The convolution has to be done first otherwise the ends of the spectra look weird
    # because of the way convolution is done by astropy.convolution.convolve_fft
    # So i'll convolve first to get the correct result from convolution at the ends of hte spectra (that I want)
    for tauVarrval in tauVarr:
        for tauval in tauarr:
            filename = model_dir + 'bc2003_hr_m62_tauV' + str(int(tauVarrval*10)) + '_csp_tau' + tauval + '_salp_allspectra.npy'

            # define and initialize numpy array so that you can resample all the spectra at once.
            # It also does the convolution in the for loop below because 
            # I wasn't sure if I gave a 2D array to convolve_fft if it would convolve each row separately.
            # I thought that it might think of the 2D ndarray as an image and convolve it that way which I don't want.
            current_model_set_array = np.load(filename)
            modellam = np.load(filename.replace('_allspectra.npy','_lamgrid.npy'))

            current_model_set = np.zeros([total_ages, len(modellam)], dtype=np.float64)
            for i in range(total_ages):
                current_model_set[i] = current_model_set_array[age_ind[i]+3]
                #current_model_set[i] = convolve_fft(current_model_spec, interplsf, boundary='extend')
                #plot_interp_and_nointerp_lsf_comparison(modellam, current_model_set, lsf, interplsf)
                # if you want to use the above line to compare then you will 
                # have to pass the interplsf as an additional argument.

            #current_model_set = ct.resample(modellam, current_model_set, lam_grid, total_ages)
            # modellam is the wavelength grid of the models at their native sampling i.e. very high resolution
            # lam_grid is the wavelength grid that I want to downgrade the models to.

            for i in range(total_ages):
                hdr = fits.Header()
                hdr['LOG_AGE'] = str(np.log10(ages[age_ind[i]]))

                metal_val = 0.02

                hdr['METAL'] = str(metal_val)
                hdr['TAU_GYR'] = str(float(tauval)/1e4)
                hdr['TAUV'] = str(tauVarrval)
                hdulist.append(fits.ImageHDU(data=current_model_set[i], header=hdr))

    final_fitsname = 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits'
    hdulist.writeto(massive_galaxies_dir + final_fitsname, overwrite=True)

    return None

def do_model_modifications(model_lam_grid, model_comp_spec, resampling_lam_grid, total_models, lsf, z):

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
    model_comp_spec = model_comp_spec / (1 + z)

    for k in range(total_models):

        # now convolve with lsf
        model_comp_spec[k] = convolve_fft(model_comp_spec[k], lsf)#, boundary='extend')
        # seems like boundary='extend' is not implemented 
        # currently for convolve_fft(). It works with convolve() though.

        # now resample to object resolution
        resampled_flam = np.zeros((len(resampling_lam_grid)))

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
            resampled_flam[i] = np.median(model_comp_spec[k][new_ind])

        model_comp_spec_modified[k] = resampled_flam

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

    return current_best_fit_model, current_best_fit_model_whole, alpha[sortargs[k]]

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

        count += 1

    chi2_redshift_arr = np.asarray(chi2_redshift_arr)
    new_chi2_minindx = np.argmin(chi2_redshift_arr)

    new_lam_grid = model_lam_grid[start_low_indx+new_chi2_minindx : start_high_indx+new_chi2_minindx+1]
    new_z_minchi2 = ( new_lam_grid[0] / (lam_obs[0] / (1 + previous_z))) - 1

    return new_z_minchi2

def do_fitting(flam, ferr, object_lam_obs, lsf, starting_z, resampling_lam_grid, \
    model_lam_grid, total_models, model_comp_spec, model_spec_hdu):

    #flam_obs, ferr_obs, lam_obs, lsf, redshift, resampling_lam_grid, \
    #model_lam_grid, total_models, model_comp_spec, bc03_all_spec

    """
    flam, ferr, object_lam_grid are in observed wavelength space. Need to deredshift
    before fitting.

    # first find best fit by deredshifting the grism spectrum using the old redshift
    # iterate 
    """

    # modify the model to be able to compare with data
    model_comp_spec_modified = \
    do_model_modifications(model_lam_grid, model_comp_spec, resampling_lam_grid, total_models, lsf, starting_z)
    print "Model mods done at current z", starting_z

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
        while 1: 

            # now get teh best fit model spectrum
            best_fit_model_in_objlamgrid, best_fit_model_whole, bestalpha = \
            get_best_fit_model(flam, ferr, object_lam_obs, resampling_lam_grid, \
                model_comp_spec_modified, previous_z, model_spec_hdu)

            # now shift in wavelength range and get new_z
            new_z = \
            shift_in_wav_get_new_z(flam, ferr, object_lam_obs, resampling_lam_grid, previous_z, \
                len(best_fit_model_in_objlamgrid), best_fit_model_whole)
            print "Current Old and New redshifts", previous_z, new_z

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

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # if this is True then the code will show you all kinds of plots and print info to the screen
    verbose = False

    # Open spectra
    spec_hdu_old = fits.open(pears_datadir + 'h_pears_s_id65620.fits')
    #spec_hdu_old = fits.open(pears_datadir + 'h_pears_s_id61447.fits')
    spec_hdu_young = fits.open(pears_datadir + 'h_pears_n_id40498.fits')

    # assign arrays
    flam_old = spec_hdu_old[4].data['FLUX']
    ferr_old = spec_hdu_old[4].data['FERROR']
    contam_old = spec_hdu_old[4].data['CONTAM']
    lam_old = spec_hdu_old[4].data['LAMBDA']
    
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
    
    # young pop
    lam_low_idx = np.argmin(abs(lam_young - 6000))
    lam_high_idx = np.argmin(abs(lam_young - 9500))
    
    lam_young = lam_young[lam_low_idx:lam_high_idx+1]
    flam_young_consub = flam_young_consub[lam_low_idx:lam_high_idx+1]

    # Plot
    if verbose:
        plotspectrum(lam_young, flam_young_consub)
        plotspectrum(lam_old, flam_old_consub)

    # now start fitting
    # get original photo-z first
    current_id = 65620
    current_field = 'GOODS-S'
    redshift = 0.9673
    # for 65620 GOODS-S
    # 0.97 Ferreras+2009  # candels 0.972 # 3dhst 0.9673
    lam_em, flam_em, ferr_em, specname, pa_chosen, netsig_chosen = gd.fileprep(current_id, redshift, current_field)
    d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

    if verbose:
        print current_field, current_id, pa_chosen, netsig_chosen  # GOODS-S 65620 PA200 657.496906164
        print d4000  # 1.69

    # ---------- Models --------- #    
    # put all models into one single fits file
    #get_model_set()

    # read in entire model set
    bc03_all_spec = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')

    # Read in LSF
    if current_field == 'GOODS-N':
        lsf_filename = lsfdir + "north_lsfs/" + "n" + str(current_id) + "_avg_lsf.txt"
    elif current_field == 'GOODS-S':
        lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"

    # read in LSF file
    lsf = np.loadtxt(lsf_filename)
    #interplsf = fcjm.get_interplsf(current_id, redshift, current_field, pa_chosen)
    #plotlsf(lsf)
    #plotlsf(interplsf)

    # ---------- Fitting --------- #
    #total_models = fcj.get_total_extensions(bc03_all_spec)  
    # this function gives the total number of extensions minus the zeroth extension
    # I used the above line to get this number which is hard coded in now.
    total_models = 22800
    # I've hard coded the number in because 

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    # first check where the models are
    model_dir = '/Volumes/Bhavins_backup/bc03_models_npy_spectra/m62/'
    # this is if working on the laptop. Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(model_dir):
        model_dir = home + '/Documents/GALAXEV_BC03/bc03/src/cspout_new/m62/'  # this path only exists on firstlight
    else:
        print "Model files not found. Exiting..."
        sys.exit(0)

    example_filename_lamgrid = 'bc2003_hr_m62_tauV0_csp_tau100_salp_lamgrid.npy'
    model_lam_grid = np.load(model_dir + example_filename_lamgrid)
    model_comp_spec = np.zeros([total_models, len(model_lam_grid)], dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec[j+1].data

    # total run time up to now
    print "Total time taken up to now --", time.time() - start, "seconds."

    # now call the actual fitting function
    # the original function (fileprep in grid_coadd) will give quantities in rest frame
    # but I need these to be in the observed frame so I will redshift them again.
    # It gives them in the rest frame because getting the d4000 and the average 
    # lambda separation needs to be figured out in the rest frame.
    flam_obs = flam_em / (1 + redshift)
    ferr_obs = ferr_em / (1 + redshift)
    lam_obs = lam_em * (1 + redshift)
    # I need them to be in the observed frame because the iterative process for 
    # finding a new redshift will de-redshift them each time a new redshift is
    # found. 

    # extend lam_grid to be able to move the lam_grid later 
    avg_dlam = old_ref.get_avg_dlam(lam_obs)

    lam_low_to_insert = np.arange(5000, lam_obs[0], avg_dlam)
    lam_high_to_append = np.arange(lam_obs[-1] + avg_dlam, 10500, avg_dlam)

    resampling_lam_grid = np.insert(lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)
    print resampling_lam_grid
    print len(resampling_lam_grid)
    sys.exit(0)

    # call actual fitting function
    do_fitting(flam_obs, ferr_obs, lam_obs, lsf, redshift, resampling_lam_grid, \
        model_lam_grid, total_models, model_comp_spec, bc03_all_spec)

    sys.exit(0)

