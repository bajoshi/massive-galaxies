from __future__ import division

from scipy.signal import fftconvolve
import numpy as np
#import numpy.ma as ma

cimport numpy as np
#cimport cython

#from astropy.convolution import convolve_fft
import matplotlib.pyplot as plt

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

def do_model_modifications(np.ndarray[DTYPE_t, ndim=1] model_lam_grid, \
    np.ndarray[DTYPE_t, ndim=2] model_comp_spec, np.ndarray[DTYPE_t, ndim=1] resampling_lam_grid, \
    int total_models, np.ndarray[DTYPE_t, ndim=1] lsf, float z):

    # Before fitting
    # 0. get lsf and models (supplied as arguments to this function)
    # 1. redshift the models
    # 2. convolve the models with the lsf
    # 3. resample the models

    # Cython type declarations for the variables
    # hardcoded lengths
    # Can len() be redefined as a C function to be faster?
    cdef int resampling_lam_grid_length = len(resampling_lam_grid)
    cdef int lsf_length = len(lsf)

    # assert types
    assert model_lam_grid.dtype == DTYPE and resampling_lam_grid.dtype == DTYPE
    assert model_comp_spec.dtype == DTYPE and lsf.dtype == DTYPE
    assert type(total_models) is int
    assert type(z) is float
    #print type(z)
    #if type(z) is DTYPE:
    #    print "All okay here."
    ##assert type(z) is DTYPE
    #z = np.float64(z)
    #print type(z)

    # create empty array in which final modified models will be stored
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_modified = \
    np.empty((total_models, resampling_lam_grid_length), dtype=DTYPE)

    # redshift lambda grid for model 
    # this is the lambda grid at the model's native resolution
    #cdef np.float64_t redshift_factor = 1 + z
    cdef np.ndarray[DTYPE_t, ndim=1] model_lam_grid_z = model_lam_grid * (1 + z) #redshift_factor

    # redshift flux
    model_comp_spec = model_comp_spec / (1 + z) #redshift_factor

    # more type definitions
    cdef int k
    cdef int i
    cdef double lam_step_low
    cdef double lam_step_high
    cdef np.ndarray[DTYPE_t, ndim=1] interppoints
    cdef np.ndarray[DTYPE_t, ndim=1] broad_lsf
    cdef np.ndarray[DTYPE_t, ndim=1] temp_broadlsf_model
    cdef np.ndarray[long, ndim=1] new_ind
    cdef np.ndarray[DTYPE_t, ndim=1] resampled_flam_broadlsf

    for k in range(total_models):

        # using a broader lsf just to see if that can do better
        interppoints = np.linspace(start=0, stop=lsf_length, num=lsf_length*5, dtype=DTYPE)
        # just making the lsf sampling grid longer # i.e. sampled at more points 
        broad_lsf = np.interp(interppoints, xp=np.arange(lsf_length), fp=lsf)
        temp_broadlsf_model = fftconvolve(model_comp_spec[k], broad_lsf)

        # resample to object resolution
        resampled_flam_broadlsf = np.zeros(resampling_lam_grid_length, dtype=DTYPE)

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
        #model_comp_spec_modified[k] = ma.array(resampled_flam_broadlsf, mask=line_mask)
        model_comp_spec_modified[k] = resampled_flam_broadlsf

    return model_comp_spec_modified
