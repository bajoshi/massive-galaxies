"""
THis is hte cythonized version of the model f_lambda
computation. This will only work on firstlight because
it needs cython to work.
"""
from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from numpy import nansum
from scipy.interpolate import griddata
import scipy.integrate as spint

import os
import sys
import time
import datetime

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

figs_data_dir = '/Users/baj/Desktop/FIGS/'
cluster_spz_scripts = '/Users/baj/Desktop/FIGS/massive-galaxies/cluster_codes/'

cdef float speed_of_light_kms = 299792.458  # km per s

# -------- Define cosmology -------- # 
# Planck 2018
cdef float H0 = 67.4  # km/s/Mpc
cdef float omega_m0 = 0.315
cdef float omega_r0 = 8.24e-5
cdef float omega_lam0 = 1.0 - omega_m0

cdef proper_distance(float H0, float omega_m0, float omega_r0, float omega_lam0, float ae):
    """
    This function will integrate 1/(a*a*H)
    between scale factor at emission to scale factor of 1.0.
    """
    p = lambda a: 1/(a*a*H0*np.sqrt((omega_m0/a**3) + (omega_r0/a**4) + omega_lam0 + ((1 - omega_m0 - omega_r0 - omega_lam0)/a**2)))
    return spint.quadrature(p, ae, 1.0)

cdef get_lum_dist_cy(float redshift):
    """
    Returns luminosity distance in megaparsecs for a given redshift.
    """

    # Type definitions
    cdef float scale_fac_to_z
    cdef float dp
    cdef float dl

    # Get the luminosity distance to the given redshift
    # Get proper distance and multiply by (1+z)
    scale_fac_to_z = 1 / (1+redshift)
    dp = proper_distance(H0, omega_m0, omega_r0, omega_lam0, scale_fac_to_z)[0]  # returns answer in Mpc/c
    dl = dp * speed_of_light_kms * (1+redshift)  # dl now in Mpc

    return dl

def compute_filter_flam(np.ndarray[DTYPE_t, ndim=1] filt_wav, np.ndarray[DTYPE_t, ndim=1] filt_trans, \
    str filtername, float start, \
    np.ndarray[DTYPE_t, ndim=2] model_comp_spec, np.ndarray[DTYPE_t, ndim=1] model_lam_grid, \
    int total_models, np.ndarray[DTYPE_t, ndim=1] zrange):

    print "\n", "Working on filter:", filtername

    # Type definitions
    cdef int i
    cdef int j

    cdef np.ndarray[DTYPE_t, ndim=2] filt_flam_model
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_z
    cdef np.ndarray[DTYPE_t, ndim=1] model_lam_grid_z
    cdef np.ndarray[DTYPE_t, ndim=1] filt_interp

    cdef float z
    cdef float dl
    cdef float pi = np.pi
    cdef float num
    cdef float den

    # Create array to save final flam values
    filt_flam_model = np.zeros(shape=(len(zrange), total_models), dtype=np.float64)

    for j in range(len(zrange)):
    
        z = zrange[j]
        print "At z:", z
    
        # ------------------------------------ Now compute model filter magnitudes ------------------------------------ #
        # Redshift the base models
        dl = get_lum_dist_cy(z)  # in Mpc
        dl = dl * 3.086e24  # convert Mpc to cm
        model_comp_spec_z = model_comp_spec / (4 * pi * dl * dl * (1+z))
        model_lam_grid_z = model_lam_grid * (1+z)
    
        # first interpolate the transmission curve to the model lam grid
        filt_interp = griddata(points=filt_wav, values=filt_trans, xi=model_lam_grid_z, method='linear')
    
        # multiply model spectrum to filter curve
        for i in xrange(total_models):
    
            num = nansum(model_comp_spec_z[i] * filt_interp)
            den = nansum(filt_interp)
    
            filt_flam_model[j, i] = num / den
    
        # transverse array to make shape consistent with others
        # I did it this way so that in the above for loop each filter is looped over only once
        # i.e. minimizing the number of times each filter is gridded on to the model grid
        #filt_flam_model_t = filt_flam_model.T

    # save the model flux densities
    np.save(figs_data_dir + 'all_model_flam_' + filtername + '.npy', filt_flam_model)
    print "Computation done and saved for:", filtername,
    print "Total time taken:", time.time() - start

    del filt_flam_model

    return None

