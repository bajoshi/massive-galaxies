from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy.cosmology import Planck15 as cosmo

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
new_codes_dir = home + "/Desktop/FIGS/new_codes/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd
import create_fsps_miles_libraries as ct
import fast_chi2_jackknife as fcj
import fast_chi2_jackknife_massive_galaxies as fcjm

def create_bc03_lib(pearsid, redshift, field, lam_grid):

    final_fitsname = 'all_comp_spectra_bc03_ssp_withlsf_' + str(pearsid) + '.fits'

    interplsf = fcjm.get_interplsf(pearsid, redshift, field)

    if interplsf is None:
        return None

    # Find total ages (and their indices in the individual fitfile's extensions) that are to be used in the fits
    example = fits.open(home + '/Documents/GALAXEV_BC03/bc03/models/Padova1994/salpeter/bc2003_hr_m22_salp_ssp.fits')
    ages = example[2].data
    age_ind = np.where((ages/1e9 < 8) & (ages/1e9 > 0.1))[0]
    total_ages = int(len(age_ind))  # 57 for SSPs

    # FITS file where the reduced number of spectra will be saved
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)
    hdulist.append(fits.ImageHDU(data=lam_grid))
    
    for filename in glob.glob(home + '/Documents/GALAXEV_BC03/bc03/models/Padova1994/salpeter/' + '*.fits'):
        
        h = fits.open(filename, memmap=False)
        currentlam = h[1].data
    
        # define and initialize numpy array so that you can resample all the spectra at once.
        # It also does the convolution in the for loop below because 
        # I wasn't sure if I gave a 2D array to convolve_fft if it would convolve each row separately.
        # I thought that it might think of the 2D ndarray as an image and convolve it that way which I don't want.
        currentspec = np.zeros([total_ages, len(currentlam)], dtype=np.float64)
        for i in range(total_ages):
            currentspec[i] = h[age_ind[i]+3].data
            currentspec[i] = convolve_fft(currentspec[i], interplsf)

        currentspec = ct.resample(currentlam, currentspec, lam_grid, total_ages)
        currentlam = lam_grid

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
            hdulist.append(fits.ImageHDU(data=currentspec[i], header=hdr))

    hdulist.writeto(savefits_dir + final_fitsname, clobber=True)

    return None

def fit_chi2_redshift(orig_lam_grid, orig_lam_grid_model, resampled_spec, ferr, num_samp_to_draw, comp_spec, nexten, spec_hdu, old_z, pearsid):

    # first find best fit assuming old redshift is ok
    fitages = []
    fitmetals = []
    best_exten = []
    bestalpha = []

    for i in range(int(num_samp_to_draw)):  # loop over jackknife runs
        if num_samp_to_draw == 1:
            flam = resampled_spec
        elif num_samp_to_draw > 1:
            flam = resampled_spec[i]
        orig_lam_grid_model_indx_low = np.where(orig_lam_grid_model == orig_lam_grid[0])[0][0]
        orig_lam_grid_model_indx_high = np.where(orig_lam_grid_model == orig_lam_grid[-1])[0][0]
        currentspec = comp_spec[:,orig_lam_grid_model_indx_low:orig_lam_grid_model_indx_high+1]

        chi2 = np.zeros(nexten, dtype=np.float64)
        alpha = np.sum(flam * currentspec / (ferr**2), axis=1) / np.sum(currentspec**2 / ferr**2, axis=1)
        chi2 = np.sum(((flam - (alpha * currentspec.T).T) / ferr)**2, axis=1)
        
        bc03_spec = spec_hdu
        # This is to get only physical ages
        sortargs = np.argsort(chi2)
        for k in range(len(chi2)):
            best_age = float(bc03_spec[sortargs[k] + 2].header['LOG_AGE'])
            age_at_z = cosmo.age(old_z).value * 1e9 # in yr
            if (best_age < np.log10(age_at_z)) & (best_age > 9 + np.log10(0.1)):
                fitages.append(best_age)
                fitmetals.append(bc03_spec[sortargs[k] + 2].header['METAL'])
                best_exten.append(sortargs[k] + 2)
                bestalpha.append(alpha[sortargs[k]])
                current_best_fit_model = currentspec[sortargs[k]]
                current_best_fit_model_whole = comp_spec[sortargs[k]]
                print "Old     Chi2", chi2[sortargs[k]]
                break

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(orig_lam_grid, current_best_fit_model*bestalpha, '-', color='k')
    ax.plot(orig_lam_grid, flam, '-', color='g')
    #ax.fill_between(orig_lam_grid, flam + ferr, flam - ferr, color='mediumseagreen')
    #ax.errorbar(orig_lam_grid, flam, yerr=ferr, fmt='-', color='g')
    #sys.exit(0)

    # now shift in wavelength space to get best fit on wavelength grid and correspongind redshift
    low_lim_for_comp = 2500
    high_lim_for_comp = 6500

    start_low_indx = np.argmin(abs(orig_lam_grid_model - low_lim_for_comp))
    start_high_indx = start_low_indx + len(current_best_fit_model) - 1

    chi2_redshift_arr = []
    count = 0 
    while 1:
        current_low_indx = start_low_indx + count
        current_high_indx = start_high_indx + count

        # do the fitting again for each shifted lam grid
        current_best_fit_model_chopped = current_best_fit_model_whole[current_low_indx:current_high_indx+1]

        alpha = np.sum(flam * current_best_fit_model_chopped / (ferr**2)) / np.sum(current_best_fit_model_chopped**2 / ferr**2)
        chi2 = np.sum(((flam - (alpha * current_best_fit_model_chopped)) / ferr)**2)

        chi2_redshift_arr.append(chi2)

        count += 1
        if orig_lam_grid_model[current_high_indx] >= high_lim_for_comp:
            break

    print "Refined Chi2", np.min(chi2_redshift_arr)
    refined_chi2_indx = np.argmin(chi2_redshift_arr)

    # plot the newer shifted spectrum
    ax.plot(orig_lam_grid_model[start_low_indx+refined_chi2_indx:start_high_indx+refined_chi2_indx+1],\
     flam, '-', color='red')
    #ax.fill_between(orig_lam_grid_model[start_low_indx+refined_chi2_indx:start_high_indx+refined_chi2_indx+1],\
    # flam + ferr, flam - ferr, color='lightred')
    #ax.errorbar(orig_lam_grid_model[start_low_indx+refined_chi2_indx:start_high_indx+refined_chi2_indx+1],\
    # flam, yerr=ferr, fmt='-', color='red')

    # shade region for dn4000 bands
    arg3900 = np.argmin(abs(orig_lam_grid_model - 3900))
    arg4050 = np.argmin(abs(orig_lam_grid_model - 4050))
    bestalpha = bestalpha[0]

    x_fill = np.arange(3850,3951,1)
    y0_fill = np.ones(len(x_fill)) * \
    (current_best_fit_model_whole[arg3900]*bestalpha - 3*0.05*current_best_fit_model_whole[arg3900]*bestalpha)
    y1_fill = np.ones(len(x_fill)) * \
    (current_best_fit_model_whole[arg4050]*bestalpha + 3*0.05*current_best_fit_model_whole[arg4050]*bestalpha)
    ax.fill_between(x_fill, y0_fill, y1_fill, color='lightsteelblue')

    x_fill = np.arange(4000,4101,1)
    ax.fill_between(x_fill, y0_fill, y1_fill, color='lightsteelblue')

    fig.savefig(new_codes_dir + 'plots_from_refining_z_code/' + 'refined_z_' + str(pearsid) + '.eps', dpi=150)

    sys.exit(0)

    return None

def get_avg_dlam(lam):

    dlam = 0
    for i in range(len(lam) - 1):
        dlam += lam[i+1] - lam[i]

    avg_dlam = dlam / (len(lam) - 1)

    return avg_dlam

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # read in dn4000 catalogs 
    pears_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/pears_4000break_catalog.txt',\
     dtype=None, names=True, skip_header=1)
    #gn1_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/figs_gn1_4000break_catalog.txt',\
    # dtype=None, names=True, skip_header=1)
    #gn2_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/figs_gn2_4000break_catalog.txt',\
    # dtype=None, names=True, skip_header=1)
    #gs1_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/figs_gs1_4000break_catalog.txt',\
    # dtype=None, names=True, skip_header=1)

    #### PEARS ####
    #pears_redshift_indices = np.where((pears_cat['redshift'] >= 0.558) & (pears_cat['redshift'] <= 1.317))[0]

    # galaxies in the possible redshift range
    #print len(pears_redshift_indices)  # 2318

    # galaxies that are outside the redshift range
    # not sure how these originally got into the pears and 3dhst matched sample....need to check again
    # these were originally selected to be within the above written redshift range
    #print np.setdiff1d(np.arange(len(pears_cat)), pears_redshift_indices)  # [1136 2032 2265]

    # galaxies with significant breaks
    sig_4000break_indices_pears = np.where(((pears_cat['dn4000'] / pears_cat['dn4000_err']) >= 3.0))[0]

    #arg = np.where((pears_cat['field'] == 'GOODS-S') & (pears_cat['pears_id'] == 16836))[0]
    #print pears_cat[arg]
    
    # there are 1226 galaxies with SNR on dn4000 greater than 3sigma
    # there are 492 galaxies with SNR on dn4000 greater than 3sigma and less than 20sigma

    # Galaxies with believable breaks; im calling them proper breaks
    prop_4000break_indices_pears = \
    np.where((pears_cat['dn4000'][sig_4000break_indices_pears] >= 1.05) & (pears_cat['dn4000'][sig_4000break_indices_pears] <= 3.0))[0]

    # there are now 483 galaxies in this dn4000 range

    all_pears_ids = pears_cat['pears_id'][sig_4000break_indices_pears][prop_4000break_indices_pears]
    all_pears_fields = pears_cat['field'][sig_4000break_indices_pears][prop_4000break_indices_pears]
    all_pears_redshifts = pears_cat['redshift'][sig_4000break_indices_pears][prop_4000break_indices_pears]

    total_galaxies = len(all_pears_ids)
    print total_galaxies

    for i in range(total_galaxies):
        print i 

        current_id = all_pears_ids[i]
        current_redshift = all_pears_redshifts[i]
        current_field = all_pears_fields[i]

        lam_em, flam_em, ferr, specname = gd.fileprep(current_id, current_redshift, current_field)
 
        # extend lam_grid to be able to move the lam_grid later 
        avg_dlam = get_avg_dlam(lam_em)

        lam_low_to_insert = np.arange(1500, lam_em[0], avg_dlam)
        lam_high_to_append = np.arange(lam_em[-1] + avg_dlam, 7500, avg_dlam)

        resampling_lam_grid = np.insert(lam_em, obj=0, values=lam_low_to_insert)
        resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

        create_bc03_lib(current_id, current_redshift, current_field, resampling_lam_grid)
        continue

        # Open fits files with comparison spectra
        try:
            bc03_spec = fits.open(savefits_dir + 'all_comp_spectra_bc03_ssp_withlsf_' + str(current_id) + '.fits', memmap=False)        
        except IOError as e:
            print e
            print "LSF was not taken into account for this galaxy. Moving on to next galaxy for now."
            continue

        # Find number of extensions in each
        bc03_extens = fcj.get_total_extensions(bc03_spec)
        bc03_extens -= 1  # because the first extension is just the resampling grid for the model

        # get lam grid for model and put in spectra for all ages in a properly shaped numpy array for faster computations
        orig_lam_grid_model = bc03_spec[1].data

        comp_spec_bc03 = np.zeros([bc03_extens, len(resampling_lam_grid)], dtype=np.float64)
        for i in range(bc03_extens):
            comp_spec_bc03[i] = bc03_spec[i+2].data

        # Get random samples by jackknifing
        num_samp_to_draw = int(1)
        if num_samp_to_draw == 1:
            resampled_spec = flam_em
        else:
            print "Running over", num_samp_to_draw, "random jackknifed samples."
            resampled_spec = ma.empty((len(flam_em), num_samp_to_draw))
            for i in range(len(flam_em)):
                if flam_em[i] is not ma.masked:
                    resampled_spec[i] = np.random.normal(flam_em[i], ferr[i], num_samp_to_draw)
                else:
                    resampled_spec[i] = ma.masked
            resampled_spec = resampled_spec.T

        # Run the actual fitting function
        ages_bc03, metals_bc03, refined_z = \
        fit_chi2_redshift(lam_em, orig_lam_grid_model, resampled_spec, ferr,\
         num_samp_to_draw, comp_spec_bc03, bc03_extens, bc03_spec, current_redshift, current_id)

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)





