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

        if new_ind.size:

            sig = flux[new_ind]
            noise = fluxerr[new_ind]
        
            if sig > 0: # only append those points where the signal is positive
                if noise/sig < 0.20: # only append those points that are less than 20% contaminated
                    old_flux[i].append(sig)
                    old_fluxerr[i].append(noise**2) # adding errors in quadrature
            else:
                continue
        else:
            continue

    return old_flux, old_fluxerr

def combine_all_position_angles_pears(pears_index, field):

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # Get the correct filename and the number of extensions
    if field == 'GOODS-N':
        filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    elif field == 'GOODS-S':
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

    # loop over the position angles (that are in separate extensions in the fits files)
    # and combine them
    # these empty lists here are to keep track of which position angles were rejected and combined
    # this is useful later when the average LSF is contructed based on the position angles that
    # were used in the combination
    rejected_pa = []
    combined_pa = []

    for count in range(spec_extens):
         
        flam = spec_hdu[count + 1].data['FLUX']
        ferr = spec_hdu[count + 1].data['FERROR']
        contam = spec_hdu[count + 1].data['CONTAM']
        lam_obs = spec_hdu[count + 1].data['LAMBDA']
        exten_pa = spec_hdu[count + 1].header['EXTNAME'].split('PA')[1]

        if count == 0.0:
            for x in range(len(lam_grid)):
                old_flam[x] = []
                old_flamerr[x] = []

        # Reject spectrum if it is more than 30% contaminated
        if np.sum(abs(ferr)) > 0.3 * np.sum(abs(flam)):
            print "Skipped extension #", count + 1, "in", specname , "because of excess contamination."
            rejected_pa.append(exten_pa) 
            continue
        else:
            old_flam, old_flamerr = add_position_angle(lam_obs, flam, ferr, old_flam, old_flamerr, lam_grid)
            combined_pa.append(exten_pa)
    
    for y in range(len(lam_grid)):
        if old_flam[y]:
            # fix the error formula over here
            # you could do a 100 bootstrap samples
            old_flamerr[y] = 1.253 * np.std(old_flam[y]) / np.sqrt(len(old_flam[y]))
            old_flam[y] = np.median(old_flam[y])
        else:
            # this shoudl not be set to 0
            # you DID NOT measure an exactly zero signal
            old_flam[y] = 0.0
            old_flamerr[y] = 0.0

    old_flam = np.asarray(old_flam)
    old_flamerr = np.asarray(old_flamerr)

    return lam_grid, old_flam, old_flamerr, rejected_pa, combined_pa

def plot_all_pa_and_combinedspec(allpearsids, allpearsfields):
    """
    This function will plot, for all the pears galaxies supplied, 
    the spectra from individual PAs and the combined spectrum 
    overlaid on top.

    Make sure that there is a one-to-one correspondence between the 
    id array and the field array supplied.
    """

    totalgalaxies = len(allpearsids)

    # Plot all PAs and plot the median
    for u in range(totalgalaxies):

        pears_index = allpearsids[u]
        field = allpearsfields[u]

        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        # Get the correct filename and the number of extensions
        if field == 'GOODS-N':
            filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
        elif field == 'GOODS-S':
            filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'
    
        fitsfile = fits.open(filename)
        fits_extens = fcj.get_total_extensions(fitsfile)

        for j in range(fits_extens):
            ax.plot(fitsfile[j+1].data['LAMBDA'], fitsfile[j+1].data['FLUX'], 'b')

        lam_obs, combined_spec, combined_spec_err = combine_all_position_angles_pears(pears_index, field)
        ax.plot(lam_obs, combined_spec, 'k')
        ax.fill_between(lam_obs, combined_spec + combined_spec_err, combined_spec - combined_spec_err, color='lightgray')

        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')

        plt.show()
        del fig, ax

    return None

if __name__ == '__main__':

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # Read pears + santini + 3dhst matched catalogs
    cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt',
                   dtype=None, names=True, skip_header=1)
    cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt',
                   dtype=None, names=True, skip_header=1)

    allcats = [cat_n, cat_s]

    catcount = 0
    for cat in allcats:

        if catcount == 0:
            fieldname = 'GOODS-N'
            print 'Starting with', len(cat), 'matched objects in', fieldname
        elif catcount == 1:
            fieldname = 'GOODS-S'
            print 'Starting with', len(cat), 'matched objects in', fieldname

        redshift_indices = np.where((cat['zphot'] >= 0.558) & (cat['zphot'] <= 1.317))[0]

        pears_id = cat['pearsid'][redshift_indices]
        photz = cat['zphot'][redshift_indices]

        print len(np.unique(pears_id)), "unique objects in", fieldname, "in redshift range"

        # create arrays for writing
        pears_id_write = []
        pearsfield = []
        rejected_pa_to_write = []
        combined_pa_to_write = []

        # Loop over all spectra 
        pears_unique_ids, pears_unique_ids_indices = np.unique(pears_id, return_index=True)
        i = 0
        for current_pears_index, count in zip(pears_unique_ids, pears_unique_ids_indices):

            lam_grid, old_flam, old_flamerr, rejected_pa, combined_pa = combine_all_position_angles_pears(current_pears_index, fieldname)

            pears_id_write.append(current_pears_index)
            pearsfield.append(fieldname)
            rejected_pa_to_write.append(rejected_pa)
            combined_pa_to_write.append(combined_pa)

        pears_id_write = np.asarray(pears_id_write)
        pearsfield = np.asarray(pearsfield)
        rejected_pa_to_write = np.asarray(rejected_pa_to_write)
        combined_pa_to_write = np.asarray(combined_pa_to_write)
        
        data = np.array(zip(pears_id_write, pearsfield, rejected_pa_to_write, combined_pa_to_write),\
                    dtype=[('pears_id_write', int), ('pearsfield', '|S7'), ('rejected_pa_to_write', list), ('combined_pa_to_write', list)])

        np.savetxt(massive_galaxies_dir + 'pears_combined_pa_' + fieldname + '.txt', data, fmt=['%d', '%s', list, list], delimiter=' ',\
                   header='Catalog for all galaxies that matched between 3DHST and PEARS in ' + fieldname + '. \n' +
                   'pears_id field redshift zphot_source ra dec dn4000 dn4000_err d4000 d4000_err')


    sys.exit(0)