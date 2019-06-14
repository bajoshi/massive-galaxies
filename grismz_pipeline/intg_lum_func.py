from __future__ import division

import numpy as np
from scipy.special import gamma, gammaincc  # gamma and upper incomplete gamma functions

import os
import sys

import matplotlib.pyplot as plt

def integrate_lum_func(M_star, phi_star, alpha, lower_lim, upper_lim):
    """
    This function will integrate the supplied LF, 
    i.e., phi_star, alpha, and M_star are given,
    between some range of magnitudes. It also 
    needs the upper and lower limits of integration 
    to be supplied.

    Integrals of the Schechter luminosity function
    involve incomplete Gamma functions.
    """

    lower_lim_conv = np.power(10, 0.4*(M_star - lower_lim))
    upper_lim_conv = np.power(10, 0.4*(M_star - upper_lim))

    print lower_lim, upper_lim
    print lower_lim_conv, upper_lim_conv

    print gammaincc(alpha+1, lower_lim_conv)
    print gammaincc(alpha+1, upper_lim_conv)
    print gamma(alpha + 1)

    intg = (gammaincc(alpha+1, lower_lim_conv) - gammaincc(alpha+1, upper_lim_conv)) * gamma(alpha+1)
    # The additional factor of gamma(alpha+1) being multiplied 
    # above is because the scipy incomplete gamma functions are
    # normalized by that factor. Look up:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaincc.html
    intg = -1 * phi_star * intg

    return intg

def get_comoving_volume(redshift):
    """
    Given the redshift, this function computes the 
    proper distance to the redshift and the comoving 
    volume within that redshift.
    """

    return comov_vol

def schechter_lf(M_star, phi_star, alpha, M):
    expo = np.exp(-1 * np.power(10, 0.4 * (M_star - M)))
    lf = 0.4 * np.log(10) * phi_star * np.power(10, 0.4 * (alpha+1) * (M_star - M)) * expo
    return lf

def run_test_case():

    print "Running test case. Read comments in the test",
    print "case function to understand what is being shown."

    # Test case for r-band in z range: 0.1 < z < 0.82
    # from Dahlen et al. 2005, ApJ, 631, 126
    # See their Table 5
    M_star = -22.38
    alpha = -1.3
    phi_star = 28.2 * 1e-4  # in units of: (Mpc/h_{70})^{-3} mag^{-1}

    # Now construct the LF based on the above parameters
    mag_arr = np.arange(-25.0, -16.9, 0.1)
    sch_lf_arr = schechter_lf(M_star, phi_star, alpha, mag_arr)

    # Try plotting to check
    # Compare this to the best fitting (solid line)
    # LF shown in Figure 10 (top left panel) of Dahlen et al. 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    title_string1 = r'$\rm Example\ LF:\ best\ fitting\ LF\ for\ {\it r}-band$' 
    title_string2 = r'$\rm in\ 0.1 \leq z \leq 0.82\ from\ Dahlen\ et\ al.\ 2005$'
    ax.set_title(title_string1 + '\n' + title_string2)
    ax.set_xlabel(r'$\rm M_R\ -\ 5\, log(h_{70})$', fontsize=14)
    ax.set_ylabel(r'$\rm \phi\ [(Mpc/h_{70})^{-3}\, mag^{-1}]$', fontsize=14)

    ax.plot(mag_arr, sch_lf_arr)

    ax.set_yscale('log')
    ax.set_xlim(-25.0, -17.0)
    ax.set_ylim(1e-6, 1e-1)
    ax.minorticks_on()

    plt.show()
    """

    # Now do the integral between two magnitudes
    print "Testing the LF integral for apparent magnitude limited sample within",
    print "appropriate redshift range (i.e., wherever the LF is valid)."
    # Set apparent magnitude limit and Hubble constant first
    app_mag_lim = 24.0
    H0 = 72.0  # this has to be in km/s/Mpc
    speed_of_light_kms = 3e5  # this has to be in km/s

    # Create redshift array # Careful! LF changes dependent on redshift
    redshift_arr = np.linspace(0.1, 0.82, 100)

    # Now convert to absolute magnitudes
    # i.e., you will integrate the LF from this abs mag to all brighter values
    abs_mag_faint_lim_arr = app_mag_lim + 5 * np.log10(H0) - 5 * np.log10(speed_of_light_kms * redshift_arr) - 25.0

    # Now do the integration
    # Set the bright end limit before calling the integrating function.
    # This is just some absurdly bright limit to make 
    # sure you don't miss any galaxies in the integral.
    # I'm setting it to the apparent magnitude of the Sun.
    upper_lim = -27.0
    for i in range(len(abs_mag_faint_lim_arr)):
        print "\n"
        print "At redshift:", redshift_arr[i]
        print "Integrating between absolute magnitude range:", abs_mag_faint_lim_arr[i], "to", upper_lim

        res = integrate_lum_func(M_star, phi_star, alpha, abs_mag_faint_lim_arr[i], upper_lim)
        print res
        if i == 3:
            break

    return None

def main():

    run_test_case()

    # Define the luminosity function
    # This is dependent on the redshift range

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)