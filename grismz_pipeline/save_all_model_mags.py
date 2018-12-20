from __future__ import division

import numpy as np
from numpy import nansum
import pysynphot
from scipy.interpolate import griddata
from astropy.io import fits

import os
import sys

home = os.getenv('HOME')
figs_dir = home + "/Desktop/FIGS/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

def compute_filter_mags(filt, model_comp_spec, model_lam_grid, total_models, z):

    # ------------------------------------ Now compute model filter magnitudes ------------------------------------ #
    all_filt_flam_model = np.zeros(total_models, dtype=np.float64)

    # Redshift the base models
    model_comp_spec_z = model_comp_spec / (1+z)
    model_lam_grid_z = model_lam_grid * (1+z)

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
        all_filt_flam_model[i] = filt_flam_model

    # transverse array to make shape consistent with others
    # I did it this way so that in the above for loop each filter is looped over only once
    # i.e. minimizing the number of times each filter is gridded on to the model grid
    #all_filt_flam_model_t = all_filt_flam_model.T

    print "Filter f_lam for models computed."

    return all_filt_flam_model

def save_filter_mags(filtername, all_model_mags):

    np.save(figs_dir + 'all_model_mags_' + filtername + '.npy', all_model_mags)

    return None

def main():

    # Read in models with emission lines adn put in numpy array
    total_models = 34542
    total_emission_lines_to_add = 12

    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_comp_spec_withlines = np.zeros((total_models, len(model_lam_grid) + total_emission_lines_to_add), dtype=np.float64)

    bc03_all_spec_hdulist_withlines = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample_withlines.fits')
    model_lam_grid_withlines = np.load(figs_dir + 'model_lam_grid_withlines.npy')
    for q in range(total_models):
        model_comp_spec_withlines[q] = bc03_all_spec_hdulist_withlines[q+1].data

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

    all_filter_names = ['u', 'f435w', 'f606w', 'f775w', 'f850lp', \
    'f125w', 'f140w', 'f160w', 'irac1', 'irac2', 'irac3', 'irac4']

    # Loop over all redshifts and filters and compute magnitudes
    zrange = np.arange(0.1, 2.51, 0.01)

    filter_count = 0
    for filt in all_filters:
        all_model_mags_filt = np.zeros((len(zrange), total_models), dtype=np.float32)
        filtername = all_filter_names[filter_count]

        for i in range(len(zrange)):
            redshift = zrange[i]
            print "Filter:", filtername, "       Redshift:", redshift

            # compute the mags
            all_model_mags_filt[i] = \
            compute_filter_mags(filt, model_comp_spec_withlines, model_lam_grid_withlines, total_models, redshift)

        # save the mags
        save_filter_mags(filtername, all_model_mags_filt)
        print "Computation done and saved for:", filtername

        filter_count += 1

    return None

if __name__ == '__main__':
    main()