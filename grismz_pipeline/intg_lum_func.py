"""
This code is to be used for getting number counts 
by integrating the luminosity function. The main
function here is -- integrate_num_mag(...).
"""
from __future__ import division

import numpy as np
from scipy.special import gamma, gammaincc  # gamma and upper incomplete gamma functions
import scipy.integrate as integrate
from scipy.interpolate import griddata

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

sys.path.append(massive_galaxies_dir)
import cosmology_calculator as cc

# Get cosmology and set speed of light
# This needs to be visible to every function
H0, omega_m0, omega_r0, omega_lam0 = cc.get_cosmology_params()
print "Flat Universe assumed. Cosmology assumed is:"
print "Hubble constant [km/s/Mpc]:", H0
print "Omega Matter:", omega_m0
print "Omega Lambda", omega_lam0
print "Omega Radiation:", omega_r0
speed_of_light_kms = 299792.458  # in km/s
print "Speed of Light [km/s]:", speed_of_light_kms

pears_coverage = 119  # in sq. arcmin
pears_coverage_sq_deg = pears_coverage / 3600
onesqdeg_to_steradian = 3.05e-4
solid_angle = pears_coverage_sq_deg * onesqdeg_to_steradian  # in steradians

def inc_gamma_integrand(x, alpha):
    integrand = np.power(x, alpha) * np.exp(-1 * x)
    return integrand

def integrate_lum_func(M_star, phi_star, alpha, lower_lim, upper_lim):
    """
    This function will integrate the supplied LF, 
    i.e., phi_star, alpha, and M_star are given,
    between some range of magnitudes. It also 
    needs the upper and lower limits of integration 
    to be supplied.

    Integrals of the Schechter luminosity function
    involve incomplete Gamma functions.
    See more detail below on why it doesn't work in
    practice.
    """

    lower_lim_conv = np.power(10, 0.4*(M_star - lower_lim))
    upper_lim_conv = np.power(10, 0.4*(M_star - upper_lim))

    print "Converted limits of integration (lower and upper):",
    print lower_lim_conv, upper_lim_conv

    # This is commented out here because 
    """
    intg = (gammaincc(alpha+1, lower_lim_conv) - gammaincc(alpha+1, upper_lim_conv)) * gamma(alpha+1)
    # The additional factor of gamma(alpha+1) being multiplied 
    # above is because the scipy incomplete gamma functions are
    # normalized by that factor. Look up:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaincc.html
    intg = -1 * phi_star * intg
    """

    # Using the quadrature integration from scipy to numerically integrate
    # The format to call quad is the following:
    # First argument should be a callable Python function
    # The next two arguments are the lower and upper limits of integration, respectively.
    # The next argument is the tuple of arguments to be passed to the callable integrand.
    # It returns the result of integration and the integration error.
    I = integrate.quad(inc_gamma_integrand, lower_lim_conv, upper_lim_conv, args=(alpha))
    intg_val = I[0] * phi_star
    intg_err = I[1] * phi_star

    return intg_val, intg_err

def get_comoving_volume(z1, z2):
    """
    Given the redshift, this function computes the 
    comoving volume between the two given redshifts.
    """

    H_0, omega_m0, omega_r0, omega_lam0 = cc.get_cosmology_params()

    # Convert redshift to scale factor because that is what 
    # cc.proper_distance() expects.
    ae1 = 1 / (1 + z1)  # i.e., scale factor at emission  
    ae2 = 1 / (1 + z2)
    dp1_res, dp1_err = cc.proper_distance(H_0, omega_m0, omega_r0, omega_lam0, ae1)
    dp2_res, dp2_err = cc.proper_distance(H_0, omega_m0, omega_r0, omega_lam0, ae2)

    dp1 = 3e5 * dp1_res
    dp2 = 3e5 * dp2_res
    # The proper distance returned by cc.proper_distance() needs to be 
    # multiplied by the speed of light in km/s to convert to Mpc.

    comov_vol = (4/3) * np.pi * (dp2**3 - dp1**3)

    return comov_vol

def schechter_lf(M_star, phi_star, alpha, M):
    expo = np.exp(-1 * np.power(10, 0.4 * (M_star - M)))
    lf = 0.4 * np.log(10) * phi_star * np.power(10, 0.4 * (alpha+1) * (M_star - M)) * expo
    return lf

def get_test_lf():

    # Test case for r-band in z range: 0.1 < z < 0.82
    # from Dahlen et al. 2005, ApJ, 631, 126
    # See their Table 5
    M_star = -22.38
    alpha = -1.3
    phi_star = 28.2 * 1e-4  # in units of: (Mpc/h_{70})^{-3} mag^{-1}

    return M_star, alpha, phi_star

def run_test_case():

    print "Running test case. Read comments in the test",
    print "case function to understand what is being shown."

    M_star, alpha, phi_star = get_test_lf()

    # Now construct the LF based on the above parameters
    mag_arr = np.arange(-25.0, -16.9, 0.1)
    sch_lf_arr = schechter_lf(M_star, phi_star, alpha, mag_arr)

    # Try plotting to check
    # Compare this to the best fitting (solid line)
    # LF shown in Figure 10 (top left panel) of Dahlen et al. 
    plot_lf = False
    if plot_lf:
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
    else:
        print "LF figure plot skipped."

    # Now do the integral between two magnitudes
    print "Testing the LF integral for apparent magnitude limited sample within",
    print "appropriate redshift range (i.e., wherever the LF is valid)."
    # Set apparent magnitude limit and Hubble constant first
    app_mag_lim = 24.0
    H0 = 70.0  # this has to be in km/s/Mpc  # I set this to 70 because Dahlen et al. used this value

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
    total_num_dens_within_z_range = 0
    upper_lim = -27.0
    for i in range(len(abs_mag_faint_lim_arr)):
        print "\n"
        print "At redshift:", redshift_arr[i]
        print "Integrating between absolute magnitude range:", abs_mag_faint_lim_arr[i], "to", upper_lim

        num_per_comov_vol, intg_err = integrate_lum_func(M_star, phi_star, alpha, abs_mag_faint_lim_arr[i], upper_lim)
        print "Number density per comoving volume at the redshift stated above [Mpc^{-3}]:", num_per_comov_vol
        total_num_dens_within_z_range += num_per_comov_vol

    print "\n", "Within the redshift interval:", redshift_arr[0], "< z <", redshift_arr[-1]
    print "Total number density is (in Mpc^{-3}):", total_num_dens_within_z_range

    print "Total comoving volume (in Mpc^3) within the redshift interval is:", 
    print "{:.3e}".format(get_comoving_volume(redshift_arr[0], redshift_arr[-1]))

    return None

def get_test_sed():
    """
    Will return an SED that is 2 Gyr old with tau=63 Gyr
    and no dust and solar metallicity. Returned spectrum
    is in L_lambda units.

    You can change these SED parameters within the function.
    """
    # Set the required model here 
    age = 4e9  # in years
    tau = 0.1  # in Gyr
    tauv = 0.0  # needs tau_v not A_v
    metallicity = 0.02  # total metal fraction
    
    # ------------------------------- Get correct directories ------------------------------- #
    figs_data_dir = '/Volumes/Bhavins_backup/bc03_models_npy_spectra/'
    threedhst_datadir = "/Volumes/Bhavins_backup/3dhst_data/"
    cspout = "/Volumes/Bhavins_backup/bc03_models_npy_spectra/cspout_2016updated_galaxev/"
    # This is if working on the laptop. 
    # Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(figs_data_dir):
        figs_data_dir = figs_dir  # this path only exists on firstlight
        threedhst_datadir = home + "/Desktop/3dhst_data/"  # this path only exists on firstlight
        cspout = home + '/Documents/galaxev_bc03_2016update/bc03/src/cspout_2016updated_galaxev/'
        if not os.path.isdir(figs_data_dir):
            print "Model files not found. Exiting..."
            sys.exit(0)

    # ------------------------------ Get models ------------------------------ #
    # read in entire model set
    # To see how these arrays were created check the code:
    # $HOME/Desktop/test-codes/shared_memory_multiprocessing/shmem_parallel_proc.py
    # This part will fail if the arrays dont already exist.
    total_models = 37761 # get_total_extensions(bc03_all_spec_hdulist)

    log_age_arr = np.load(figs_data_dir + 'log_age_arr.npy', mmap_mode='r')
    metal_arr = np.load(figs_data_dir + 'metal_arr.npy', mmap_mode='r')
    nlyc_arr = np.load(figs_data_dir + 'nlyc_arr.npy', mmap_mode='r')
    tau_gyr_arr = np.load(figs_data_dir + 'tau_gyr_arr.npy', mmap_mode='r')
    tauv_arr = np.load(figs_data_dir + 'tauv_arr.npy', mmap_mode='r')
    ub_col_arr = np.load(figs_data_dir + 'ub_col_arr.npy', mmap_mode='r')
    bv_col_arr = np.load(figs_data_dir + 'bv_col_arr.npy', mmap_mode='r')
    vj_col_arr = np.load(figs_data_dir + 'vj_col_arr.npy', mmap_mode='r')
    ms_arr = np.load(figs_data_dir + 'ms_arr.npy', mmap_mode='r')
    mgal_arr = np.load(figs_data_dir + 'mgal_arr.npy', mmap_mode='r')

    model_lam_grid_withlines_mmap = np.load(figs_data_dir + 'model_lam_grid_withlines.npy', mmap_mode='r')
    model_comp_spec_withlines_mmap = np.load(figs_data_dir + 'model_comp_spec_withlines.npy', mmap_mode='r')

    # Now find the model you need
    # First find the closest values to the user supplied values
    closest_age_idx = np.argmin(abs(log_age_arr - np.log10(age)))
    closest_tau_idx = np.argmin(abs(tau_gyr_arr - tau))
    closest_tauv_idx = np.argmin(abs(tauv_arr - tauv))
    closest_metal_idx = np.argmin(abs(metal_arr - metallicity))

    closest_age = log_age_arr[closest_age_idx]
    closest_tau = tau_gyr_arr[closest_tau_idx]
    closest_tauv = tauv_arr[closest_tauv_idx]
    closest_metallicity = metal_arr[closest_metal_idx]

    """
    print "\n", "Returning test SED with the following parameters:"
    print "Age [Gyr]:", 10**closest_age / 1e9
    print "Tau [Gyr]:", closest_tau
    print "Tau_V:", closest_tauv
    print "Metallicity [total fraction of metals]:", closest_metallicity
    """

    # Get the model index
    model_idx = np.where((log_age_arr == closest_age) & (tau_gyr_arr == closest_tau) & \
        (tauv_arr == closest_tauv) & (metal_arr == closest_metallicity))[0]
    model_idx = int(model_idx)

    # Now assign the llam array
    llam = model_comp_spec_withlines_mmap[model_idx]

    return llam, model_lam_grid_withlines_mmap

def get_lum_dist(redshift):
    """
    Returns luminosity distance in megaparsecs for a given redshift.
    """

    # Get the luminosity distance to the given redshift
    # Get proper distance and multiply by (1+z)
    scale_fac_to_z = 1 / (1+redshift)
    dp = cc.proper_distance(H0, omega_m0, omega_r0, omega_lam0, scale_fac_to_z)[0]  # returns answer in Mpc/c
    dl = dp * speed_of_light_kms * (1+redshift)  # dl now in Mpc

    return dl

def get_kcorr(sed_llam, sed_lam, redshift, filt_curve):
    """
    Returns the K-correction ONLY due to redshift,
    i.e., within the same filter.
    It needs to be supplied with the object SED
    (l_lambda and lambda), redshift, and with the 
    filter curve.
    """

    z_fac = 1 / (1+redshift)

    # Redshift the spectrum
    lam_obs = sed_lam * (1+redshift)
    llam_obs = sed_llam / (1+redshift)

    wav_filt_idx = np.where((sed_lam >= filt_curve['wav'][0]) & (sed_lam <= filt_curve['wav'][-1]))

    # Make sure the filter curve and the SED are 
    # on the same wavelength grid.
    filt_curve_interp_obs = griddata(points=filt_curve['wav'], values=filt_curve['trans'], \
        xi=lam_obs[wav_filt_idx], method='linear', fill_value=0.0)
    filt_curve_interp_rf = griddata(points=filt_curve['wav'], values=filt_curve['trans'], \
        xi=sed_lam[wav_filt_idx], method='linear', fill_value=0.0)
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(filt_curve['wav'], filt_curve['trans'], color='k')
    ax.plot(lam_obs[wav_filt_idx], filt_curve_interp_obs, color='b')
    ax.plot(sed_lam[wav_filt_idx], filt_curve_interp_rf, color='r')
    plt.show()
    sys.exit(0)
    """

    # Define integrand for observed space integral
    y_obs = llam_obs[wav_filt_idx] * lam_obs[wav_filt_idx] * filt_curve_interp_obs

    # Define integrand for rest-frame space integral
    y_rf = sed_llam[wav_filt_idx] * sed_lam[wav_filt_idx] * filt_curve_interp_rf

    obs_intg = z_fac * integrate.simps(y=y_obs, x=lam_obs[wav_filt_idx])
    em_intg = integrate.simps(y=y_rf, x=sed_lam[wav_filt_idx])

    kcorr = -2.5 * np.log10(obs_intg / em_intg)

    return kcorr

def get_volume_element(redshift):

    dl = get_lum_dist(redshift)  # returns answer in Mpc

    # Get the volume element per redshift
    dVdz_num = 4 * np.pi * speed_of_light_kms * dl**2
    dVdz_den = H0 * (1+redshift)**2

    dVdz = dVdz_num / dVdz_den

    return dVdz

def integrate_num_mag(z1, z2, M_star, alpha, phi_star):
    """
    """

    # Beginning with equation 9 in Gardner 1998, PASP, 110, 291
    # and moving backwards.
    app_mag = 24.0
    # Generate redshift array to integrate over
    total_samples = 1000
    z_arr = np.linspace(z1, z2, total_samples)
    nmz = np.zeros(total_samples)
    abs_mag_arr = np.zeros(total_samples)

    dz = (z2 - z1) / total_samples

    for i in range(total_samples):
        z = z_arr[i]
        print "\n", "At z:", z
    
        # First get the volume element at redshift z
        dVdz = get_volume_element(z)  # in Mpc^3
        print "Volume element [Mpc^3]:", dVdz
    
        # Now get LF (i.e, number density at abs mag M) counts 
        # dependent on the redshfit and the apparent magnitude limit.
        # You need to convert the apparent mag to absolute before 
        # passing the abs mag to the schechter_lf function.
        # You will also need the K-correction
        dl = get_lum_dist(z)  # in Mpc
    
        # Also get the SED 
        sed_llam, sed_lam = get_test_sed()
    
        # Read in the i-band filter
        f775w_filt_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f775w_filt_curve.txt', \
            dtype=None, names=['wav', 'trans']) 
    
        # Now get the K-correction
        # This function needs the SED, z, and the filter curve
        kcorr = get_kcorr(sed_llam, sed_lam, z, f775w_filt_curve)
        print "K-correction:", kcorr
    
        abs_mag = app_mag - kcorr - 5 * np.log10(dl * 1e6/10)
        print "Absolute Magnitude:", abs_mag
        abs_mag_arr[i] = abs_mag
    
        lf_value = schechter_lf(M_star, phi_star, alpha, abs_mag)
        print "LF value []:", lf_value
    
        # Now put them together
        nmz[i] = solid_angle * dVdz * lf_value * dz / (4 * np.pi)
        print "Total number counts per steradian, i.e., N(<M,z)[sr^-1]:", nmz[i]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(abs_mag_arr, nmz)
    plt.show()
    sys.exit(0)

    return nmz

def main():

    # To run the older test function
    # run_test_case()

    # Define the luminosity function
    # This is dependent on the redshift range
    M_star, alpha, phi_star = get_test_lf()

    # Now do the integral per magnitude bin
    zlow = 0.6
    zhigh = 1.2
    integrate_num_mag(zlow, zhigh, M_star, alpha, phi_star)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)