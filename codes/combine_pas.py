from __future__ import division

import numpy as np
from astropy.io import fits

import glob
import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
pears_spectra_dir = home + "/Documents/PEARS/data_spectra_only/"
figures_dir = stacking_analysis_dir + "figures/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd
import fast_chi2_jackknife as fcj

def add_position_angle(lam, flux, fluxerr, old_flux, old_fluxerr, lam_grid):
    
    for i in range(len(lam)):

        new_ind = np.where(lam == lam_grid[i])[0]

        sig = flux[new_ind]
        noise = fluxerr[new_ind]
        
        if sig > 0: # only append those points where the signal is positive
            if noise/sig < 0.20: # only append those points that are less than 20% contaminated
                old_flux[i].append(sig)
                old_fluxerr[i].append(noise**2) # adding errors in quadrature
        else:
            continue

    return old_flux, old_fluxerr

def combine_all_position_angles_pears(pears_index):

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # Get the correct filename and the number of extensions
    filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    if not os.path.isfile(filename):
        filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

    spec_hdu = fits.open(filename)
    spec_extens = fcj.get_total_extensions(spec_hdu)
    specname = os.path.basename(filename)

    print "Working on PEARS ID ", pears_index, "with ", spec_extens, " extensions."

    # Loop over all extensions and combine them
    # First find where the largest lam array is and also create the arrays for storing medians
    lam_obs_arr = []
    for j in range(spec_extens):

        lam_obs = spec_hdu[j + 1].data['LAMBDA']
        lam_obs_arr.append(len(lam_obs))

    lam_obs_arr = np.asarray(lam_obs_arr)

    max_lam_ind = np.argmax(lam_obs_arr)
    lam_grid = spec_hdu[max_lam_ind + 1].data['LAMBDA']

    old_flam = np.zeros(len(lam_grid))
    old_flamerr = np.zeros(len(lam_grid))
    
    old_flam = old_flam.tolist()
    old_flamerr = old_flamerr.tolist()

    for count in range(spec_extens):
         
        flam = spec_hdu[count + 1].data['FLUX']
        ferr = spec_hdu[count + 1].data['FERROR']
        contam = spec_hdu[count + 1].data['CONTAM']
        lam_obs = spec_hdu[count + 1].data['LAMBDA']

        if count == 0.0:
            for x in range(len(lam_grid)):
                old_flam[x] = []
                old_flamerr[x] = []

        # Reject spectrum if it is more than 30% contaminated
        if np.sum(abs(ferr)) > 0.3 * np.sum(abs(flam)):
            print "Skipped extension # ", count + 1, "in ", specname , " because of excess contamination."
            continue
        else:
            old_flam, old_flamerr = add_position_angle(lam_obs, flam, ferr, old_flam, old_flamerr, lam_grid)
    
    for y in range(len(lam_grid)):
        if old_flam[y]:
            old_flamerr[y] = 1.253 * np.std(old_flam[y]) / np.sqrt(len(old_flam[y]))
            old_flam[y] = np.median(old_flam[y])
        else:
            old_flam[y] = 0.0
            old_flamerr[y] = 0.0

    old_flam = np.asarray(old_flam)
    old_flamerr = np.asarray(old_flamerr)

    return lam_grid, old_flam, old_flamerr

if __name__ == '__main__':

    # Read pears + 3dhst catalog
    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt',
                       dtype=None, names=True, skip_header=2)

    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    # Find indices for massive galaxies
    massive_galaxies_indices = np.where(stellarmass >= 11.0)[0]

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # Plot all PAs and plot the median
    for u in range(len(pears_id[massive_galaxies_indices])):

        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        # Get the correct filename and the number of extensions
        filename = data_path + 'h_pears_n_id' + str(pears_id[massive_galaxies_indices][u]) + '.fits'
        if not os.path.isfile(filename):
            filename = data_path + 'h_pears_s_id' + str(pears_id[massive_galaxies_indices][u]) + '.fits'
    
        fitsfile = fits.open(filename)
        fits_extens = fcj.get_total_extensions(fitsfile)

        for j in range(fits_extens):
            ax.plot(fitsfile[j+1].data['LAMBDA'], fitsfile[j+1].data['FLUX'], 'b')

        lam_obs, combined_spec, combined_spec_err = combine_all_position_angles_pears(pears_id[massive_galaxies_indices][u])
        ax.plot(lam_obs, combined_spec, 'k')
        ax.fill_between(lam_obs, combined_spec + combined_spec_err, combined_spec - combined_spec_err, color='lightgray')

        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')

        plt.show()

    sys.exit(0)