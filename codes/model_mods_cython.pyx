from __future__ import division

import numpy as np
import numpy.ma as ma

from astropy.convolution import convolve_fft

import os
import sys

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

def do_model_modifications(np.ndarray[] object_lam_obs, np.ndarray[] model_lam_grid, \
	np.ndarray[] model_comp_spec, np.ndarray[] resampling_lam_grid, \
	int total_models, np.ndarray[] lsf, double z):

    # Before fitting
    # 0. get lsf and models (supplied as arguments to this function)
    # 1. redshift the models
    # 2. convolve the models with the lsf
    # 3. resample the models

    # Cython type declarations for the variables
    cdef int k
    cdef int i
    cdef int lam_step_low
    cdef int lam_step_high

    cdef int resampling_lam_grid_length
    cdef int lsf_length

    # hardcoded lengths
    # Can len() be redefined as a C function to be faster?
    resampling_lam_grid_length = len(resampling_lam_grid)
    lsf_length = len(lsf)

    # create empty array in which final modified models will be stored
    model_comp_spec_modified = np.empty((total_models, resampling_lam_grid_length), dtype=DTYPE)

    # redshift lambda grid for model 
    # this is the lambda grid at the model's native resolution
    model_lam_grid_z = model_lam_grid * (1 + z)

    # redshift flux
    model_comp_spec = model_comp_spec / (1 + z)

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
    line_mask = np.zeros(resampling_lam_grid_length)

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

        # using a broader lsf just to see if that can do better
        interppoints = np.linspace(0, lsf_length, int(lsf_length*10))
        # just making the lsf sampling grid longer # i.e. sampled at more points 
        broad_lsf = np.interp(interppoints, xp=np.arange(lsf_length), fp=lsf)
        temp_broadlsf_model = convolve_fft(model_comp_spec[k], broad_lsf)

        # resample to object resolution
        cdef np.ndarray[] resampled_flam_broadlsf = np.zeros(resampling_lam_grid_length, dtype=DTYPE)

        for i in range(resampling_lam_grid_length):

            if i == 0:
                lam_step_high = resampling_lam_grid[i+1] - resampling_lam_grid[i]
                lam_step_low = lam_step_high
            elif i == resampling_lam_grid_length - 1:
                lam_step_low = resampling_lam_grid[i] - resampling_lam_grid[i-1]
                lam_step_high = lam_step_low
            else:
                lam_step_high = resampling_lam_grid[i+1] - resampling_lam_grid[i]
                lam_step_low = resampling_lam_grid[i] - resampling_lam_grid[i-1]

            new_ind = np.where((model_lam_grid_z >= resampling_lam_grid[i] - lam_step_low) & \
                (model_lam_grid_z < resampling_lam_grid[i] + lam_step_high))[0]

            resampled_flam_broadlsf[i] = np.mean(temp_broadlsf_model[new_ind])

        # Now mask the flux at these wavelengths using the mask generated before the for loop
        resampled_flam_broadlsf = ma.array(resampled_flam_broadlsf, mask=line_mask)

        model_comp_spec_modified[k] = resampled_flam_broadlsf

    return model_comp_spec_modified
