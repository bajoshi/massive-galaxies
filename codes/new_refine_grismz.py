from __future__ import division

import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, convolve, Gaussian1DKernel

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

def convolve_models_with_lsf(pearsid, field, lam_grid, lsf):

    # ONly using SSP for now
    # get valid ages i.e. between 100 Myr and 8 Gyr
    total_ages, age_ind, ages = old_ref.get_valid_ages_in_model(pop='ssp', example_filename='bc2003_hr_m22_salp_ssp.fits')

    # FITS file where the reduced number of spectra will be saved
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)
    hdulist.append(fits.ImageHDU(data=lam_grid))
    
    for filename in glob.glob(home + '/Documents/GALAXEV_BC03/bc03/models/Padova1994/salpeter/' + '*.fits'):
        
        h = fits.open(filename, memmap=False)
        modellam = h[1].data
    
        # define and initialize numpy array so that you can resample all the spectra at once.
        # It also does the convolution in the for loop below because 
        # I wasn't sure if I gave a 2D array to convolve_fft if it would convolve each row separately.
        # I thought that it might think of the 2D ndarray as an image and convolve it that way which I don't want.
        convolvedspec = np.zeros([total_ages, len(modellam)], dtype=np.float64)
        for i in range(total_ages):
            currentspec = h[age_ind[i]+3].data
            convolvedspec[i] = convolve_fft(currentspec, lsf)
            #plot_interp_and_nointerp_lsf_comparison(modellam, currentspec, lsf, interplsf)
            # if you want to use the above line to compare then you will 
            # have to pass the interplsf as an additional argument.

        convolvedspec = ct.resample(modellam, convolvedspec, lam_grid, total_ages)
        # modellam is the wavelength grid of the models at their native sampling i.e. very high resolution
        # lam_grid is the wavelength grid that I want to downgrade the models to.

        for i in range(total_ages):
            hdr = fits.Header()
            hdr['LOG_AGE'] = str(np.log10(ages[age_ind[i]]))

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
            hdulist.append(fits.ImageHDU(data=convolvedspec[i], header=hdr))

    final_fitsname = 'all_comp_spectra_bc03_ssp_withlsf_' + field + '_' + str(pearsid) + '.fits'
    hdulist.writeto(massive_galaxies_dir + final_fitsname, overwrite=True)

    return None

def get_best_fit_model(flam, ferr, object_lam_grid, model_lam_grid, model_comp_spec):

    # chop the model to be consistent with the objects lam grid
    model_lam_grid_indx_low = np.where(model_lam_grid == object_lam_grid[0])[0][0]
    model_lam_grid_indx_high = np.where(model_lam_grid == object_lam_grid[-1])[0][0]
    model_spec_in_objlamgrid = model_comp_spec[:, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # create chi2 array
    # i.e. there is one chi2 value for each model 
    # therefore the chi2 array has as many elements as there are models to compare
    # the vertical scaling factor, alpha, is computed with a formula 
    # determined analytically. There is also one alpha value for each model
    num_models = int(model_spec_in_objlamgrid.shape[0])
    chi2 = np.zeros(num_models, dtype=np.float64)
    alpha = np.sum(flam * model_spec_in_objlamgrid / (ferr**2), axis=1) / np.sum(model_spec_in_objlamgrid**2 / ferr**2, axis=1)
    chi2 = np.sum(((flam - (alpha * model_spec_in_objlamgrid.T).T) / ferr)**2, axis=1)

    # This is to get only physical ages
    sortargs = np.argsort(chi2)
    for k in range(len(chi2)):
        best_age = float(bc03_spec[sortargs[k] + 2].header['LOG_AGE'])
        age_at_z = cosmo.age(old_z).value * 1e9 # in yr
        if (best_age < np.log10(age_at_z)) & (best_age > 9 + np.log10(0.1)):
            fitages.append(best_age)
            fitmetals.append(bc03_spec[sortargs[k] + 2].header['METAL'])
            best_exten.append(sortargs[k] + 2)
            bestalpha_plot.append(alpha[sortargs[k]])
            old_chi2.append(chi2[sortargs[k]])
            current_best_fit_model = model_spec_in_objlamgrid[sortargs[k]]
            current_best_fit_model_whole = model_comp_spec[sortargs[k]]
            break

    return current_best_fit_model, current_best_fit_model_whole

def shift_in_wav_get_new_z(flam, ferr, model_lam_grid, spec_elem_model_fit, best_fit_model_whole):

    # now shift in wavelength space to get best fit on wavelength grid and correspongind redshift
    low_lim_for_comp = 1500
    high_lim_for_comp = 7500

    start_low_indx = np.argmin(abs(model_lam_grid - low_lim_for_comp))
    start_high_indx = start_low_indx + spec_elem_model_fit - 1
    # these starting low and high indices are set up in a way that the dimensions of
    # best_fit_model_in_objlamgrid and flam and ferr are the same to be able to 
    # compute alpha and chi2 below.

    chi2_redshift_arr = []
    count = 0 
    while 1:
        current_low_indx = start_low_indx + count
        current_high_indx = start_high_indx + count

        # do the fitting again for each shifted lam grid
        best_fit_model_in_objlamgrid = best_fit_model_whole[current_low_indx:current_high_indx+1]

        alpha = np.sum(flam * best_fit_model_in_objlamgrid / (ferr**2)) / np.sum(best_fit_model_in_objlamgrid**2 / ferr**2)
        chi2 = np.sum(((flam - (alpha * best_fit_model_in_objlamgrid)) / ferr)**2)

        chi2_redshift_arr.append(chi2)

        count += 1
        if model_lam_grid[current_high_indx] >= high_lim_for_comp:
            break

    new_chi2_minindx = np.argmin(new_chi2)
    new_z_minchi2 = new_z[new_chi2_minindx]

    return new_z_minchi2

def do_fitting(flam, ferr, object_lam_grid, starting_z, model_lam_grid, model_comp_spec):
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

        previous_z = starting_z
        while 1: 
            # de-redshift
            flam_em = flam * (1 + previous_z)
            ferr_em = ferr * (1 + previous_z)
            object_lam_grid_rest = object_lam_grid / (1 + previous_z)

            # now get teh best fit model spectrum
            best_fit_model_in_objlamgrid, best_fit_model_whole = \
            get_best_fit_model(flam_em, ferr_em, object_lam_grid_rest, model_lam_grid, model_comp_spec)

            # now shift in wavelength range and get new_z
            new_z = shift_in_wav_get_new_z(flam_em, ferr_em, model_lam_grid, len(best_fit_model_in_objlamgrid), best_fit_model_whole)

            if abs(new_z - previous_z) < 0.01 * new_z:
                break

            previous_z = new_z

    return None

if __name__ == '__main__':

    # if this is True then the code will show you all kinds of plots and print info to the screen
    verbose = False

    # Open spectra
    spec_hdu_old = fits.open(pears_datadir + 'h_pears_s_id65620.fits')
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
    redshift = 0.972  # candels 0.972 # 3dhst 0.9673
    lam_em, flam_em, ferr_em, specname, pa_chosen, netsig_chosen = gd.fileprep(current_id, redshift, current_field)
    d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

    if verbose:
        print current_field, current_id, pa_chosen, netsig_chosen  # GOODS-S 65620 PA200 657.496906164
        print d4000  # 1.69

    # ---------- Models --------- # 
    # read in bc03 models
    #bc03_spec_old = fits.open(savefits_dir + 'all_comp_spectra_bc03_ssp_withlsf_' + current_field + '_' + str(current_id) + '.fits')
    
    # get lsf and convolve the models with the lsf
    if current_field == 'GOODS-N':
        lsf_filename = lsfdir + "north_lsfs/" + "n" + str(current_id) + "_avg_lsf.txt"
    elif current_field == 'GOODS-S':
        lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"

    # read in LSF file
    lsf = np.loadtxt(lsf_filename)
    interplsf = fcjm.get_interplsf(current_id, redshift, current_field, pa_chosen)
    #plotlsf(lsf)
    #plotlsf(interplsf)

    # extend lam_grid to be able to move the lam_grid later 
    avg_dlam = old_ref.get_avg_dlam(lam_em)

    lam_low_to_insert = np.arange(1500, lam_em[0], avg_dlam)
    lam_high_to_append = np.arange(lam_em[-1] + avg_dlam, 7500, avg_dlam)

    resampling_lam_grid = np.insert(lam_em, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

    # get models convolved with lsf
    #convolve_models_with_lsf(current_id, current_field, resampling_lam_grid, lsf)

    # read in models convolved with LSF
    bc03_spec_new = fits.open(massive_galaxies_dir + 'all_comp_spectra_bc03_ssp_withlsf_' + current_field + '_' + str(current_id) + '.fits')

    # ---------- Fitting --------- #
    # first arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    bc03_extens = fcj.get_total_extensions(bc03_spec_new)  # this function gives the total number of extensions minus the zeroth extension

    model_comp_spec = np.zeros([bc03_extens, len(resampling_lam_grid)], dtype=np.float64)
    for j in range(bc03_extens-1):  # the -1 takes into account that the #1 extension is the model wavelength grid
        model_comp_spec[j] = bc03_spec_new[j+2].data

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
    do_fitting(flam_obs, ferr_obs, lam_obs, redshift, resampling_lam_grid, model_comp_spec)

    sys.exit(0)