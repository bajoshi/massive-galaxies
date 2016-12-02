from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.convolution import convolve_fft

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

def fit_chi2_redshift(orig_lam_grid, orig_lam_grid_model, flam, ferr, comp_spec, nexten, resampled_spec, num_samp_to_draw, spec_hdu, old_z):

    delta_lam = 
    step_array = np.arange(-400, 400, delta_lam)

    for step in step_array:




    return None

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
    
    # there are 1226 galaxies with SNR on dn4000 greater than 3sigma
    # there are 492 galaxies with SNR on dn4000 greater than 3sigma and less than 20sigma

    # Galaxies with believable breaks; im calling them proper breaks
    prop_4000break_indices_pears = \
    np.where((pears_cat['dn4000'][sig_4000break_indices_pears] >= 1.2) & (pears_cat['dn4000'][sig_4000break_indices_pears] <= 3.0))[0]

    # there are now 483 galaxies in this dn4000 range

    all_pears_ids = pears_cat['pears_id'][sig_4000break_indices_pears][prop_4000break_indices_pears]
    all_pears_fields = pears_cat['field'][sig_4000break_indices_pears][prop_4000break_indices_pears]
    all_pears_redshifts = pears_cat['redshift'][sig_4000break_indices_pears][prop_4000break_indices_pears]

    total_galaxies = len(all_pears_ids)

    for i in range(total_galaxies):

        print i

        current_id = all_pears_ids[i]
        current_redshift = all_pears_redshifts[i]
        current_field = all_pears_fields[i]

        lam_em, flam_em, ferr, specname = gd.fileprep(current_id, current_redshift, current_field)
        resampling_lam_grid = lam_em 
 
        #create_bc03_lib(current_id, current_redshift, current_field, lam_em)

        # Open fits files with comparison spectra
        try:
            bc03_spec = fits.open(savefits_dir + 'all_comp_spectra_bc03_ssp_withlsf_' + str(current_id) + '.fits', memmap=False)        
        except IOError as e:
            print e
            print "LSF was not taken into account for this galaxy. Moving on to next galaxy for now."
            continue

        # Find number of extensions in each
        bc03_extens = fcj.get_total_extensions(bc03_spec)

        comp_spec_bc03 = np.zeros([bc03_extens, len(resampling_lam_grid)], dtype=np.float64)
        for i in range(bc03_extens):
            comp_spec_bc03[i] = bc03_spec[i+1].data        

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
        ages_bc03, metals_bc03, refined_z = fit_chi2_redshift()

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)





