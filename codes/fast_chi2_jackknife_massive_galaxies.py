from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.convolution import convolve_fft

import os
import sys
import time 
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/fits_comp_spectra/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd
import fast_chi2_jackknife as fcj
import create_fsps_miles_libraries as ct

def find_matches_in_ferreras2009(pears_cat, ferreras_prop_cat, ferreras_cat):

    pears_ids = pears_cat['pearsid']
    ferreras_ids = ferreras_cat['id']

    for i in range(len(pears_ids[massive_galaxies_indices])):
        if pears_ids[massive_galaxies_indices][i] in ferreras_ids:
            ferreras_ind = np.where(ferreras_ids == pears_ids[massive_galaxies_indices][i])[0]
            print ferreras_cat['ra'][ferreras_ind], ',', ferreras_cat['dec'][ferreras_ind], ferreras_ids[ferreras_ind], pears_ids[massive_galaxies_indices][i], " matched."
            #print pears_cat['mstar'][massive_galaxies_indices][i], ferreras_prop_cat['mstar'][ferreras_ind]
            #print pears_cat['threedzphot'][massive_galaxies_indices][i], ferreras_cat['z'][ferreras_ind]
        else:
            data_path = home + "/Documents/PEARS/data_spectra_only/"
            # Get the correct filename and the number of extensions
            filename = data_path + 'h_pears_n_id' + str(pears_ids[massive_galaxies_indices][i]) + '.fits'
            if not os.path.isfile(filename):
                filename = data_path + 'h_pears_s_id' + str(pears_ids[massive_galaxies_indices][i]) + '.fits'

            fitsfile = fits.open(filename)
            print fitsfile[0].header['ra'], ',', fitsfile[0].header['dec'], " did not match."
            #print pears_ids[massive_galaxies_indices][i], " did not match."

    return None

def create_bc03_lib_main(lam_grid, pearsid):

    cspout = home + '/Documents/GALAXEV_BC03/bc03/src/cspout_new/'
    metals = ['m62']  # fixed at solar

    final_fits_name = 'all_comp_spectra_bc03_solar_withlsf_' + str(pearsid) + '.fits'

    # find the corresponding LSF
    pears_data_path = home + "/Documents/PEARS/data_spectra_only/"

    filename_n = pears_data_path + 'h_pears_n_id' + str(pearsid) + '.fits'
    filename_s = pears_data_path + 'h_pears_s_id' + str(pearsid) + '.fits'
    if (os.path.isfile(filename_n)) and (os.path.isfile(filename_s)):
        print "this galaxy's id repeats in SOUTH and NORTH. Pick another galaxy for now."
        sys.exit(0)
    if not os.path.isfile(filename_n):
        filename = pears_data_path + 'h_pears_s_id' + str(pearsid) + '.fits'
    else:
        filename = filename_n

    specname = os.path.basename(filename)
    field = specname.split('_')[2]

    if field == 'n':
        lsf = np.loadtxt(home + '/Desktop/FIGS/new_codes/pears_lsfs/north_lsfs/n' + str(pearsid) + '_avg_lsf.txt')
    elif field == 's':
        lsf = np.loadtxt(home + '/Desktop/FIGS/new_codes/pears_lsfs/south_lsfs/s' + str(pearsid) + '_avg_lsf.txt')

    interplsf = np.interp(np.linspace(0,len(lsf),len(lsf)*24), xp=np.arange(len(lsf)), fp=lsf)

    # Find total ages (and their indices in the individual fitfile's extensions) that are to be used in the fits
    example = fits.open(home + '/Documents/GALAXEV_BC03/bc03/src/cspout_new/m62/bc2003_hr_m62_tauV0_csp_tau100_salp.fits')
    ages = example[2].data[1:]
    age_ind = np.where((ages/1e9 < 8) & (ages/1e9 > 0.1))[0]
    total_ages = int(len(age_ind))
    
    # FITS file where the reduced number of spectra will be saved
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)
    
    # read in BC03 spectra
    # I've restricted tauV, tau, lambda, and ages in distinguishing spectra
    tauVarr = np.arange(0.0, 2.0, 0.1)
    logtauarr = np.arange(-2, 2, 0.2)
    tauarr = np.empty(len(logtauarr)).astype(np.str)
    
    for i in range(len(logtauarr)):
        tauarr[i] = str(int(float(str(10**logtauarr[i])[0:6])*10000))
    
    # Read in each individual spectrum and convolve it first and then chop and resample it
    # The convolution has to be done first otherwise the ends of the spectra look weird
    # because of the way convolution is done by astropy.convolution.convolve_fft
    # So i'll convolve first to get the correct result from convolution at the ends of hte spectra (that I want)
    for metallicity in metals:
        metalfolder = metallicity + '/'
        for tauVarrval in tauVarr:
            for tauval in tauarr:
                filename = cspout + metalfolder + 'bc2003_hr_' + metallicity + '_tauV' + str(int(tauVarrval*10)) + '_csp_tau' + tauval + '_salp.fits'
                #print filename
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

                    metal_val = 0.02

                    hdr['METAL'] = str(metal_val)
                    hdr['TAU_GYR'] = str(float(tauval)/1e4)
                    hdr['TAUV'] = str(float(tauVarrval)/10)
                    hdulist.append(fits.ImageHDU(data=currentspec[i], header=hdr))

    hdulist.writeto(savefits_dir + final_fits_name, clobber=True)

    return None

def create_model_fits(libname, lam_grid, pearsid):

    if libname == 'bc03':

        create_bc03_lib_main(lam_grid, final_fits_name, pearsid)

    if libname == 'miles':

        final_fits_name = 'all_comp_spectra_miles_' + str(pearsid) + '.fits'
        ct.create_miles_lib_main(lam_grid, final_fits_name, pearsid)

    if libname == 'fsps':

        final_fits_name = 'all_comp_spectra_fsps_' + str(pearsid) + '.fits'
        ct.create_fsps_lib_main(lam_grid, final_fits_name, pearsid, np.array([0.02]))  # metallicity fixed at solar

    return None

def create_models_wrapper(resampling_lam_grid, pearsid):

    # Create consolidated fits files for faster array comparisons
    create_model_fits('bc03', resampling_lam_grid, pearsid)
    print "BCO3 library saved for PEARS ID:", pearsid
    create_model_fits('miles', resampling_lam_grid, pearsid)
    print "MILES library saved for PEARS ID:", pearsid
    create_model_fits('fsps', resampling_lam_grid, pearsid)
    print "FSPS library saved for PEARS ID:", pearsid

    return None

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Save info for the massive galaxies into a separate file
    # You will need this to put in a tabular form in your paper

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"
    #lsf_path = massive_galaxies_dir + "north_lsfs/"

    # Read pears + 3dhst catalog
    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt',
                       dtype=None, names=True, skip_header=2)

    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    # Find indices for massive galaxies
    massive_galaxies_indices = np.where(stellarmass >= 10.5)[0] 
    # 102 massive galaxies with log(M/M_sol)>10^10.5
    # 17 massive galaxies with log(M/M_sol)>11

    # Match with Ferreras et al. 2009
    #ferreras_cat = np.genfromtxt(massive_galaxies_dir + 'ferreras_2009_ETG_cat.txt', dtype=None,\
    #                             names=['id', 'ra', 'dec', 'z'], usecols=(0,1,2,5), skip_header=23)
    #ferreras_prop_cat = np.genfromtxt(massive_galaxies_dir + 'ferreras_2009_ETG_prop_cat.txt', dtype=None,\
    #                             names=['id', 'mstar'], usecols=(0,1), skip_header=23)    
    #
    #find_matches_in_ferreras2009(cat, ferreras_prop_cat, ferreras_cat)
    #sys.exit(0)
    # There are 12 galaxies that matched using ids
    # Also match using RA and DEC
    # Also match colors, stellar masses, and redshifts given in the Ferreras et al. 2009 paper

    # Loop over all spectra 
    for u in range(len(pears_id[massive_galaxies_indices])):

        print "\n", "Currently working with PEARS object id: ", pears_id[massive_galaxies_indices][u], "with log(M/M_sol) =", stellarmass[massive_galaxies_indices][u]
        print "At --", dt.now()

        redshift = photz[massive_galaxies_indices][u]
        lam_em, flam_em, ferr, specname = gd.fileprep(pears_id[massive_galaxies_indices][u], redshift)

        resampling_lam_grid = lam_em
        # define resampling grid for model spectra. i.e. resampling_lam_grid = lam_em
        # This will be different for each galaxy because they are all at different redshifts
        # so when unredshifted the lam grid is different for each.
        #create_models_wrapper(lam_em, pears_id[massive_galaxies_indices][u])

        # Skip galaxy if it was already analyzed before i.e. these are the galaxies with M > 10^11 M_sol
        if os.path.isfile(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_ages_bc03.txt'):
            continue

        # Open fits files with comparison spectra
        bc03_spec = fits.open(savefits_dir + 'all_comp_spectra_bc03_solar_' + str(pears_id[massive_galaxies_indices][u]) + '.fits', memmap=False)
        miles_spec = fits.open(savefits_dir + 'all_comp_spectra_miles_' + str(pears_id[massive_galaxies_indices][u]) + '.fits', memmap=False)
        fsps_spec = fits.open(savefits_dir + 'all_comp_spectra_fsps_' + str(pears_id[massive_galaxies_indices][u]) + '.fits', memmap=False)

        # Find number of extensions in each
        bc03_extens = fcj.get_total_extensions(bc03_spec)
        miles_extens = fcj.get_total_extensions(miles_spec)
        fsps_extens = fcj.get_total_extensions(fsps_spec)

        # Convolve model spectra with the line spread function for each galaxy
        # and also
        # set up numpy comparison spectra arrays for faster array computations at the same time
        #filename = lsf_path + 'n' + str(pears_id[massive_galaxies_indices][u]) + '_avg_lsf.txt'
        #if os.path.isfile(filename):
        #    lsf = np.loadtxt(filename)
        #else:
        #    filename = lsf_path + 's' + str(pears_id[massive_galaxies_indices][u]) + '_avg_lsf.txt'
        # Remove the try except blocks below once you have both north and south lsfs

        comp_spec_bc03 = np.zeros([bc03_extens, len(resampling_lam_grid)], dtype=np.float64)
        for i in range(bc03_extens):
            comp_spec_bc03[i] = bc03_spec[i+1].data
            #try:
            #    comp_spec_bc03[i] = np.convolve(comp_spec_bc03[i], lsf)
            #except NameError:
            #    pass
    
        comp_spec_miles = ma.zeros([miles_extens, len(resampling_lam_grid)], dtype=np.float64)
        for i in range(miles_extens):
            comp_spec_miles[i] = miles_spec[i+1].data
            mask_indices = np.isnan(miles_spec[i+1].data)
            comp_spec_miles[i] = ma.masked_array(comp_spec_miles[i], mask=mask_indices)
            #try:
            #    comp_spec_miles[i] = np.convolve(comp_spec_miles[i], lsf)
            #except NameError:
            #    pass
    
        comp_spec_fsps = np.zeros([fsps_extens, len(resampling_lam_grid)], dtype=np.float64)
        for i in range(fsps_extens):
            comp_spec_fsps[i] = fsps_spec[i+1].data
            #try:
            #    comp_spec_fsps[i] = np.convolve(comp_spec_fsps[i], lsf)
            #except NameError:
            #    pass

        # Get random samples by jackknifing
        num_samp_to_draw = int(1e3)
        print "Running over", num_samp_to_draw, "random jackknifed samples."
        resampled_spec = ma.empty((len(flam_em), num_samp_to_draw))
        for i in range(len(flam_em)):
            if flam_em[i] is not ma.masked:
                resampled_spec[i] = np.random.normal(flam_em[i], ferr[i], num_samp_to_draw)
            else:
                resampled_spec[i] = ma.masked
        resampled_spec = resampled_spec.T

        # Files to save distribution of best params in
        f_ages_bc03 = open(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_ages_bc03.txt', 'wa')
        f_logtau_bc03 = open(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_logtau_bc03.txt', 'wa')
        f_tauv_bc03 = open(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_tauv_bc03.txt', 'wa')
        f_exten_bc03 = open(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_exten_bc03.txt', 'wa')
    
        f_ages_miles = open(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_ages_miles.txt', 'wa')
        f_metals_miles = open(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_metals_miles.txt', 'wa')
        f_exten_miles = open(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_exten_miles.txt', 'wa')
    
        f_ages_fsps = open(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_ages_fsps.txt', 'wa')
        f_logtau_fsps = open(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_logtau_fsps.txt', 'wa')
        f_exten_fsps = open(savefits_dir + 'jackknife' + str(pears_id[massive_galaxies_indices][u]) + '_exten_fsps.txt', 'wa')

        # Run the actual fitting function
        ages_bc03, metals_bc03, tau_bc03, tauv_bc03, exten_bc03 = fcj.fit_chi2(flam_em, ferr, comp_spec_bc03, bc03_extens, resampled_spec, num_samp_to_draw, 'bc03', bc03_spec)
        ages_miles, metals_miles, exten_miles = fcj.fit_chi2(flam_em, ferr, comp_spec_miles, miles_extens, resampled_spec, num_samp_to_draw, 'miles', miles_spec)
        ages_fsps, metals_fsps, tau_fsps, exten_fsps = fcj.fit_chi2(flam_em, ferr, comp_spec_fsps, fsps_extens, resampled_spec, num_samp_to_draw, 'fsps', fsps_spec)

        logtau_bc03 = np.log10(tau_bc03)
        logtau_fsps = np.log10(tau_fsps)

        # Save the data from the runs
        ### BC03 ### 
        for k in range(len(ages_bc03)):
            f_ages_bc03.write(str(ages_bc03[k]) + ' ')

        for k in range(len(logtau_bc03)):
            f_logtau_bc03.write(str(logtau_bc03[k]) + ' ')

        for k in range(len(tauv_bc03)):
            f_tauv_bc03.write(str(tauv_bc03[k]) + ' ')

        for k in range(len(exten_bc03)):
            f_exten_bc03.write(str(int(exten_bc03[k])) + ' ')

        ### MILES ###
        for k in range(len(ages_miles)):
            f_ages_miles.write(str(ages_miles[k]) + ' ')

        for k in range(len(metals_miles)):
            f_metals_miles.write(str(metals_miles[k]) + ' ')

        for k in range(len(exten_miles)):
            f_exten_miles.write(str(int(exten_miles[k])) + ' ')

        ### FSPS ###
        for k in range(len(ages_fsps)):
            f_ages_fsps.write(str(ages_fsps[k]) + ' ')

        for k in range(len(logtau_fsps)):
            f_logtau_fsps.write(str(logtau_fsps[k]) + ' ')

        for k in range(len(exten_fsps)):
            f_exten_fsps.write(str(int(exten_fsps[k])) + ' ')

        # Close all files to write them -- 
        f_ages_bc03.close()
        f_logtau_bc03.close()
        f_tauv_bc03.close()
        f_exten_bc03.close()
    
        f_ages_miles.close()
        f_metals_miles.close()
        f_exten_miles.close()
    
        f_ages_fsps.close()
        f_logtau_fsps.close()
        f_exten_fsps.close()

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)