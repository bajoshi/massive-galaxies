# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from scipy.constants import *
import scipy.integrate as spint

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

def generate_redshift_age_lookup_table():

    redshift_step = 0.005
    redshift_arr = np.arange(0, 1100, redshift_step)
    age_of_universe = np.zeros(len(redshift_arr))

    mpc, H_0, omega_m0, omega_r0, omega_lam0, year = get_cosmology_params()

    count = 0 
    for z in redshift_arr:

        ao = 1 / (1+z)
        age_of_universe[count] = time_after_big_bang(H_0, omega_m0, omega_r0, omega_lam0, ao)[0] * mpc / year
        count += 1

    redshift_age_arr = np.vstack((redshift_arr, age_of_universe))
    np.save(massive_galaxies_dir + "lookuptable_redshift_ages", redshift_age_arr)
    # np.save saves it in a binary format with the extension .npy 
    # This is done for faster I/O

    return None

def get_cosmology_params():

    # ========== CONSTANTS ========== # DO NOT CHANGE!
    # mpc = 1e6 * parsec/1e3 # Mpc to km conversion
    # Cosmological constants
    # The 8.24e-5 value for relativistic matter density is from carroll and ostlie. 
    # They say it's from WMAP but I need to find it.
    # Planck 2018
    H_0 = 67.4  # km/s/Mpc
    omega_m0 = 0.315
    omega_r0 = 8.24e-5
    omega_lam0 = 1.0 - omega_m0

    return H_0, omega_m0, omega_r0, omega_lam0

# calculates age at a given scale factor based on FRW metric
# needs to be multiplied by Mpc/(seconds_per_year) to convert to time to years
def time_after_big_bang(H_0, omega_m0, omega_r0, omega_lam0, ao):
    f = lambda a: 1/(a*H_0*np.sqrt((omega_m0/a**3) + (omega_r0/a**4) + omega_lam0 + ((1 - omega_m0 - omega_r0 - omega_lam0)/a**2))) # this is the function 1/(aH)
    return spint.quadrature(f, 0.0, ao)

# calculates proper distance at a given scale factor based on FRW metric
# needs to be multiplied by c in km/s to convert to distance in Mpc
def proper_distance(H_0, omega_m0, omega_r0, omega_lam0, ae):
    p = lambda a: 1/(a*a*H_0*np.sqrt((omega_m0/a**3) + (omega_r0/a**4) + omega_lam0 + ((1 - omega_m0 - omega_r0 - omega_lam0)/a**2))) # this is the function 1/(a*a*H)
    return spint.quadrature(p, ae, 1.0)

if __name__ == '__main__':
    
    redshift = np.arange(0,10,0.1)
    dp = np.zeros(len(redshift))
    
    z_test_arr = [0.5, 0.75, 1, 2, 12]
    
    # Couldn't really get these formulae to work the one time I tried to check
    """
    # Comoving Volumes
    # these formulae will give the comoving volume in
    dh = 1e-3*c/H_0
    e = lambda z: 1/(np.sqrt(omega_m0*(1+z)**3 + omega_k*(1+z)**2 + omega_lam0)) # this is actually 1/E(z)
    dc = spint.quadrature(e, 0.0, redshift)[0]
    if omega_k > 0:
        dm = dh*(1/np.sqrt(omega_k))*np.sinh(dc*np.sqrt(omega_k)/dh)
    if omega_k == 0:
        dm = dc
    if omega_k < 0:
        dm = dh*(1/np.sqrt(abs(omega_k)))*np.sin(dc*np.sqrt(abs(omega_k))/dh)
    if omega_k > 0:
        comoving_vol = (4*pi*dh**3/(2*omega_k))*((dm/dh) * np.sqrt(1 + (omega_k*dm**2/dh**2)) - (1/abs(omega_k)) * np.arcsinh(np.sqrt(abs(omega_k))*dm/dh))
    if omega_k == 0:
        comoving_vol = (4*pi*dm**3)
    if omega_k < 0:
        comoving_vol = (4*pi*dh**3/(2*omega_k))*((dm/dh) * np.sqrt(1 + (omega_k*dm**2/dh**2)) - (1/abs(omega_k)) * np.arcsin(np.sqrt(abs(omega_k))*dm/dh))
    """
    
    for count in range(len(redshift)):
        # the last parameter is the supplied redshift converted to scale factor when the light was emitted
        # the proper distance is given in Mpc
        dp[count] = 1e-3*c*proper_distance(H_0, omega_m0, omega_r0, omega_lam0, 1/(1+redshift[count]))[0] # distance in Mpc
    
    #time = time_after_big_bang(H_0, omega_m0, omega_r0, omega_lam0)[0] * mpc / year # time in yr
    
    #comoving_vol = (4*pi*dp**3/3)*1e-9 # comoving volume in Gpc^3 # works only for a universe with no curvature
    #lum_dist = (1 + redshift)*dp
    #ang_size_dist = dp/(1 + redshift)
    
    #print "%e" % time
    #print dp
    #print ang_size_dist
    #print lum_dist
    #print comoving_vol
    
    """
    # this for loop is for confirming the four values asked for in the email
    for count in z_test_arr:
        print count
        dp = 1e-3*c*proper_distance(H_0, omega_m0, omega_r0, omega_lam0, 1/(1+count))[0]
        print dp
        print dp/(1 + count) # ang_size_dist
        print (1 + count)*dp # lum_dist
        print (4*pi*dp**3/3)*1e-9 # comoving volume
        print " "
    """
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)#, title = '$Proper\ distance\ vs\ redshift$')
    ax.set_xlabel('$Redshift,\ z$', fontsize=14)
    ax.set_ylabel('$Proper\ distance,\ d_p\ (Mpc)$', fontsize=14)
    ax.plot(redshift, dp, 'o', label='$Proper\ Distance\ \Omega_{\Lambda}=0.714\ \Omega_m=0.286\ \Omega_r=8.24e-5\ H_0=69.6$', color='black', markersize=2, markeredgecolor='black')
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1.2, length=4.5, which='major')
    ax.legend(loc=0, numpoints=1, prop={'size':11})
    
    
    # this plot gives you the age at the redshift label already plotted
    ax2 = ax.twiny()
    redshifts_for_labels = np.arange(0, 11, 2) # 6 values
    scale_factors = 1/(1 + redshifts_for_labels)
    age_labels = np.zeros(len(redshifts_for_labels))
    for count in range(len(age_labels)):
        age_labels[count] = np.around(time_after_big_bang(H_0, omega_m0, omega_r0, omega_lam0, scale_factors[count])[0] * mpc / (year * 1e9) , decimals = 2) # Age in Gyr
    
    ax2.set_xticks(redshifts_for_labels)
    ax2.set_xticklabels(age_labels)
    ax2.set_xlabel('$Age (Gyr)$', fontsize=14)
    ax2.minorticks_on()
    ax2.tick_params('both', width=1, length=3, which='minor')
    ax2.tick_params('both', width=1.2, length=4.5, which='major')
    
    fig.savefig('Proper_distance_vs_redshift', dpi=300)
    plt.show()
    """