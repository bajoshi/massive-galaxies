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
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"

sys.path.append(massive_galaxies_dir)
sys.path.append(massive_galaxies_dir + "grismz_pipeline/")
import cosmology_calculator as cc
import mocksim_results as mr

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
speed_of_light_ang = speed_of_light_kms * 1e3 * 1e10  # In angstroms per second

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
    age = 3e9  # in years
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

    log_age_arr = np.load(figs_data_dir + 'log_age_arr_chab.npy', mmap_mode='r')
    metal_arr = np.load(figs_data_dir + 'metal_arr_chab.npy', mmap_mode='r')
    nlyc_arr = np.load(figs_data_dir + 'nlyc_arr_chab.npy', mmap_mode='r')
    tau_gyr_arr = np.load(figs_data_dir + 'tau_gyr_arr_chab.npy', mmap_mode='r')
    tauv_arr = np.load(figs_data_dir + 'tauv_arr_chab.npy', mmap_mode='r')
    ub_col_arr = np.load(figs_data_dir + 'ub_col_arr_chab.npy', mmap_mode='r')
    bv_col_arr = np.load(figs_data_dir + 'bv_col_arr_chab.npy', mmap_mode='r')
    vj_col_arr = np.load(figs_data_dir + 'vj_col_arr_chab.npy', mmap_mode='r')
    ms_arr = np.load(figs_data_dir + 'ms_arr_chab.npy', mmap_mode='r')
    mgal_arr = np.load(figs_data_dir + 'mgal_arr_chab.npy', mmap_mode='r')

    model_lam_grid_withlines_mmap = np.load(figs_data_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')
    model_comp_spec_withlines_mmap = np.load(figs_data_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')

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
    lam_km = model_lam_grid_withlines_mmap / 1e13  # was in angstroms, need it in kilometers to get to nu

    # Convert to L_nu before returning
    lnu = llam * model_lam_grid_withlines_mmap**2 / speed_of_light_ang
    nu = speed_of_light_kms / lam_km  # in Hz
    
    """
    # Plot test SED
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(model_lam_grid_withlines_mmap, lnu, color='k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.invert_xaxis()
    plt.show()
    """

    return lnu, nu, llam, model_lam_grid_withlines_mmap

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

def get_kcorr(sed_lnu, sed_nu, redshift, filt_curve_Q, filt_curve_R):
    """
    Returns the K-correction due to redshift and 
    for going between two filters. 
    It needs to be supplied with the object SED
    (L_nu and nu), redshift, and with the 
    rest frame and obs filter curves. 

    This function uses the K-correction formula
    given in eq 8 of Hogg et al. 2002.
    """

    # Redshift the spectrum
    nu_obs = sed_nu / (1+redshift)
    lnu_obs = sed_lnu * (1+redshift)

    # Convert filter wavlengths to frequency
    filt_curve_R_nu = speed_of_light_ang / filt_curve_R['wav']
    filt_curve_Q_nu = speed_of_light_ang / filt_curve_Q['wav']

    # Find indices where filter and spectra frequencies match
    R_nu_filt_idx = np.where((nu_obs <= filt_curve_R_nu[0]) & (nu_obs >= filt_curve_R_nu[-1]))
    Q_nu_filt_idx = np.where((sed_nu <= filt_curve_Q_nu[0]) & (sed_nu >= filt_curve_Q_nu[-1]))

    # Make sure the filter curve and the SED are 
    # on the same wavelength grid.
    # Filter R is in obs frame
    # Filter Q is in rest frame
    filt_curve_R_interp_obs = griddata(points=filt_curve_R_nu, values=filt_curve_R['trans'], \
        xi=nu_obs[R_nu_filt_idx], method='linear', fill_value=0.0)
    filt_curve_Q_interp_rf = griddata(points=filt_curve_Q_nu, values=filt_curve_Q['trans'], \
        xi=sed_nu[Q_nu_filt_idx], method='linear', fill_value=0.0)

    # Define standard for AB magnitdues
    # i.e., 3631 Janskys in L_nu units
    standard = 3631 * 1e-23  # in erg/s/Hz

    # Define integrands
    y1 = lnu_obs[R_nu_filt_idx] * filt_curve_R_interp_obs / nu_obs[R_nu_filt_idx]
    y2 = standard * filt_curve_Q_interp_rf / sed_nu[Q_nu_filt_idx]
    y3 = standard * filt_curve_R_interp_obs / nu_obs[R_nu_filt_idx]
    y4 = sed_lnu[Q_nu_filt_idx] * filt_curve_Q_interp_rf / sed_nu[Q_nu_filt_idx]

    # Now get the integrals required within the K-correction formula
    integral1 = integrate.simps(y=y1, x=nu_obs[R_nu_filt_idx])
    integral2 = integrate.simps(y=y2, x=sed_nu[Q_nu_filt_idx])
    integral3 = integrate.simps(y=y3, x=nu_obs[R_nu_filt_idx])
    integral4 = integrate.simps(y=y4, x=sed_nu[Q_nu_filt_idx])

    # Compute K-correction
    kcorr_qr = -2.5 * np.log10((1+redshift) * integral1 * integral2 / (integral3 * integral4))
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(filt_curve['wav'], filt_curve['trans'], color='k')
    ax.plot(lam_obs[wav_filt_idx], filt_curve_interp_obs, color='b')
    ax.plot(sed_lam[wav_filt_idx], filt_curve_interp_rf, color='r')
    plt.show()
    sys.exit(0)
    """

    # Older equations for single filter 
    # These wont work anymore. They are all in L_lambda units.
    """
    # Define integrand for observed space integral
    y_obs = llam_obs[wav_filt_idx] * lam_obs[wav_filt_idx] * filt_curve_interp_obs
    # Define integrand for rest-frame space integral
    y_rf = sed_llam[wav_filt_idx] * sed_lam[wav_filt_idx] * filt_curve_interp_rf

    obs_intg = integrate.simps(y=y_obs, x=lam_obs[wav_filt_idx]) / (1+redshift)
    em_intg = integrate.simps(y=y_rf, x=sed_lam[wav_filt_idx])

    kcorr = -2.5 * np.log10(obs_intg / em_intg)
    """

    return kcorr_qr

def get_volume_element(redshift):

    dl = get_lum_dist(redshift)  # returns answer in Mpc

    # Get the volume element per redshift
    dVdz_num = 4 * np.pi * speed_of_light_kms * dl**2
    Ez = np.sqrt(omega_lam0 + omega_m0 * (1+redshift)**3)
    dVdz_den = H0 * (1+redshift)**2 * Ez

    dVdz = dVdz_num / dVdz_den

    return dVdz

def integrate_num_mag(z1, z2, M_star, alpha, phi_star, app_mag, lf_band, num_counts_band, silent=True):
    """
    """

    # Beginning with equation 9 in Gardner 1998, PASP, 110, 291
    # and moving backwards.
    # Generate redshift array to integrate over
    total_samples = 200  # should be some large number but your final integral should not change 
    z_arr = np.linspace(z1, z2, total_samples)
    nmz = np.zeros(total_samples)
    abs_mag_arr = np.zeros(total_samples)

    # Also get the SED 
    sed_lnu, sed_nu = get_test_sed()

    for i in range(total_samples):
        z = z_arr[i]
        if not silent:
            print "\n", "At z:", z
    
        # First get the volume element at redshift z
        dVdz = get_volume_element(z)  # in Mpc^3
        if not silent:
            print "Volume element [Mpc^3]:", dVdz
    
        # Now get LF (i.e, number density at abs mag M) counts 
        # dependent on the redshfit and the apparent magnitude limit.
        # You need to convert the apparent mag to absolute before 
        # passing the abs mag to the schechter_lf function.
        # You will also need the K-correction
        dl = get_lum_dist(z)  # in Mpc
    
        # Now get the K-correction
        # This function needs the SED, z, and the filter curve
        kcorr = get_kcorr(sed_lnu, sed_nu, z, lf_band, num_counts_band)
        if not silent:
            print "K-correction:", kcorr
    
        abs_mag = app_mag - kcorr - 5 * np.log10(dl * 1e6/10)
        if not silent:
            print "Absolute Magnitude:", abs_mag
        abs_mag_arr[i] = abs_mag
    
        lf_value = schechter_lf(M_star, phi_star, alpha, abs_mag)
        if not silent:
            print "LF value [# per comoving Mpc^3 between M (abs mag) to M+dM]:", lf_value
    
        # Now put them together
        nmz[i] = onesqdeg_to_steradian * dVdz * lf_value / (4 * np.pi)
        # since you want the number per square degree multiply the integral 
        # by the solid angle corresponding to 1 sq. degree.
        if not silent:
            print "Total number counts per sq degree, i.e., N(<M,z)[deg^-2]:", nmz[i]

    total_num = integrate.simps(y=nmz, x=z_arr)
    print app_mag, "                       ", total_num

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(abs_mag_arr, nmz)
    plt.show()
    sys.exit(0)
    """

    return total_num

def main():

    # To run the older test function
    # run_test_case()

    # Define the luminosity function
    # This is assumed independent of the redshift range i.e, no LF evolution
    # From GAMA survey: Kelvin et al. 2014, MNRAS
    # Although I say J, H, K here, these really are Y, J, and H.
    # I took YJH LF params from the KElvin paper.
    # From Table 5, LF for ellipticals.
    # ------------ i-band ------------ # 
    M_star_i = -22.28  # mag
    alpha_i = -0.77
    phi_star_i = 1.04e-3  # per mag per Mpc^3

    # ------------ denoted j-band but this is Y from Kelvin2014 ------------ # 
    M_star_j = -22.27  # mag
    alpha_j = -0.82
    phi_star_j = 0.86e-3  # per mag per Mpc^3

    # ------------ denoted h-band but this is J from Kelvin2014------------ # 
    M_star_h = -22.73  # mag
    alpha_h = -0.77
    phi_star_h = 0.91e-3  # per mag per Mpc^3

    # ------------ denoted k-band but this is H from Kelvin2014 ------------ # 
    M_star_k = -22.98  # mag
    alpha_k = -0.80
    phi_star_k = 0.91e-3  # per mag per Mpc^3

    # Set up redshift ranges
    zlow_pears = 0.600
    zhigh_pears = 1.235
    print "Running over redshift range for PEARS:", zlow_pears, "to", zhigh_pears 

    # ------------------------ # 
    zlow_wfirst_J = 1.67
    zhigh_wfirst_J = 2.3
    print "Running over redshift range for WFIRST J-band:", zlow_wfirst_J, "to", zhigh_wfirst_J 

    zlow_wfirst_H = 2.3
    zhigh_wfirst_H = 2.9
    print "Running over redshift range for WFIRST H-band:", zlow_wfirst_H, "to", zhigh_wfirst_H

    zlow_wfirst_K = 2.9
    zhigh_wfirst_K = 3.45
    print "Running over redshift range for WFIRST K-band:", zlow_wfirst_K, "to", zhigh_wfirst_K

    # ------------------------ #
    zlow_euclid_J = 1.45
    zhigh_euclid_J = 2.3
    print "Running over redshift range for Euclid J-band:", zlow_euclid_J, "to", zhigh_euclid_J 

    zlow_euclid_H = 2.3
    zhigh_euclid_H = 2.9
    print "Running over redshift range for Euclid H-band:", zlow_euclid_H, "to", zhigh_euclid_H 

    zlow_euclid_K = 2.9
    zhigh_euclid_K = 3.35
    print "Running over redshift range for Euclid K-band:", zlow_euclid_K, "to", zhigh_euclid_K 

    # ------------ Now do the integral per magnitude bin ------------ #
    app_mag_lim_arr = np.arange(18.0, 28.5, 0.5)
    print "Running over apparent magnitude limit array:"
    print app_mag_lim_arr

    print "Apparent magnitude limit", "             ", "Total number in redshift range:"
    print "----------------------------------------------------------------------------"

    total_num_dens_in_z_arr_pears = np.zeros(len(app_mag_lim_arr))

    total_num_dens_in_z_arr_wfirst_J = np.zeros(len(app_mag_lim_arr))
    total_num_dens_in_z_arr_wfirst_H = np.zeros(len(app_mag_lim_arr))
    total_num_dens_in_z_arr_wfirst_K = np.zeros(len(app_mag_lim_arr))

    total_num_dens_in_z_arr_euclid_J = np.zeros(len(app_mag_lim_arr))
    total_num_dens_in_z_arr_euclid_H = np.zeros(len(app_mag_lim_arr))
    total_num_dens_in_z_arr_euclid_K = np.zeros(len(app_mag_lim_arr))

    # ----------------------------- i band for PEARS ----------------------------- #
    # Since the measured PEARS numbers are in i-band
    # Read in the two filter curves needed
    # filter in which LF is measured
    lf_band = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f775w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    # filter in which number counts are being predicted
    num_counts_band = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f775w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])

    for i in range(len(app_mag_lim_arr)):
        app_mag_lim = app_mag_lim_arr[i]
        total_num_dens_in_z_arr_pears[i] = \
        integrate_num_mag(zlow_pears, zhigh_pears, M_star_i, alpha_i, phi_star_i, app_mag_lim, lf_band, num_counts_band)

    print "Cumulative number counts for PEARS:"
    cumulative_num_counts_pears = np.cumsum(total_num_dens_in_z_arr_pears)
    print cumulative_num_counts_pears

    # Ratio of observed to predicted number counts for PEARS
    #pears_num_gal_in_z_range = 790  # within z range of spz paper
    # This includes ALL of PEARS i.e., even the incomplete magnitude bins
    # What needs to be done is find the number of PEARS galaxies 
    # out to 26 mag (because that is the completeness limit) 
    # WITHOUT ANY OTHER cuts, and divide that by the predicted 
    # number of galaxies out to 26 app mag.
    pears_num_gal_in_z_range = 790  # within z range AND within 26 mag
    idx26 = np.where(app_mag_lim_arr == 26.0)[0]
    pears_pred_num_out_to_26 = float(cumulative_num_counts_pears[idx26])
    r = (pears_num_gal_in_z_range / pears_pred_num_out_to_26) * 3600/pears_coverage
    # make sure to also take the areal coverage into account
    print "Total PEARS predicted number counts within 0.600 <= z <= 1.235 over 1 sq deg:", pears_pred_num_out_to_26 
    print "Actual to predicted ratio:", r 

    # ----------------------------- J, H, and K bands for WFIRST and Euclid ----------------------------- #
    # Read in the two filter curves needed
    # filter in which LF is measured
    lf_band_J = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/UKIRT_UKIDSS.Y.dat', dtype=None, names=['wav', 'trans'])
    # filter in which number counts are being predicted
    num_counts_band_J = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/F105W_IR_throughput.csv', delimiter=',', \
        usecols=(1,2), dtype=[np.float, np.float], names=['wav', 'trans'], skip_header=1)

    lf_band_H = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/UKIRT_UKIDSS.J.dat', dtype=None, names=['wav', 'trans'])
    num_counts_band_H = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f160w_filt_curve.txt', dtype=None, names=['wav', 'trans'])

    lf_band_K = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/UKIRT_UKIDSS.H.dat', dtype=None, names=['wav', 'trans'])
    num_counts_band_K = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f160w_filt_curve.txt', dtype=None, names=['wav', 'trans'])

    # Convert LF band wavelengths from microns to angstroms
    # The k-correction code assumes filter wavelengths are in angstroms
    #lf_band_J['wav'] = lf_band_J['wav'] * 1e4
    #lf_band_H['wav'] = lf_band_H['wav'] * 1e4
    #lf_band_K['wav'] = lf_band_K['wav'] * 1e4

    for i in range(len(app_mag_lim_arr)):
        app_mag_lim = app_mag_lim_arr[i]
        # ------------ J ------------ #
        total_num_dens_in_z_arr_wfirst_J[i] = \
        integrate_num_mag(zlow_wfirst_J, zhigh_wfirst_J, M_star_j, alpha_j, phi_star_j, app_mag_lim, lf_band_J, num_counts_band_J)
        total_num_dens_in_z_arr_euclid_J[i] = \
        integrate_num_mag(zlow_euclid_J, zhigh_euclid_J, M_star_j, alpha_j, phi_star_j, app_mag_lim, lf_band_J, num_counts_band_J)
        # ------------ H ------------ #
        total_num_dens_in_z_arr_wfirst_H[i] = \
        integrate_num_mag(zlow_wfirst_H, zhigh_wfirst_H, M_star_h, alpha_h, phi_star_h, app_mag_lim, lf_band_H, num_counts_band_H)
        total_num_dens_in_z_arr_euclid_H[i] = \
        integrate_num_mag(zlow_euclid_H, zhigh_euclid_H, M_star_h, alpha_h, phi_star_h, app_mag_lim, lf_band_H, num_counts_band_H)
        # ------------ K ------------ #
        total_num_dens_in_z_arr_wfirst_K[i] = \
        integrate_num_mag(zlow_wfirst_K, zhigh_wfirst_K, M_star_k, alpha_k, phi_star_k, app_mag_lim, lf_band_K, num_counts_band_K)
        total_num_dens_in_z_arr_euclid_K[i] = \
        integrate_num_mag(zlow_euclid_K, zhigh_euclid_K, M_star_k, alpha_k, phi_star_k, app_mag_lim, lf_band_K, num_counts_band_K)

    # Get index for app mag = 24.0
    appmag24_idx = np.where(app_mag_lim_arr == 24.0)[0]

    # define other fractions and multiply cumulative sums by them
    f_d4000 = 0.72
    f_acc = 0.36

    # ------------ J ------------ #
    print "Cumulative number counts for WFIRST J-band out to 24 mag:"
    cumulative_num_counts_wfirst_J = np.cumsum(total_num_dens_in_z_arr_wfirst_J)
    cumulative_num_counts_wfirst_J = cumulative_num_counts_wfirst_J * r * f_d4000 * f_acc
    print cumulative_num_counts_wfirst_J[appmag24_idx]
    # ------------ H ------------ #
    print "Cumulative number counts for WFIRST H-band out to 24 mag:"
    cumulative_num_counts_wfirst_H = np.cumsum(total_num_dens_in_z_arr_wfirst_H)
    cumulative_num_counts_wfirst_H = cumulative_num_counts_wfirst_H * r * f_d4000 * f_acc
    print cumulative_num_counts_wfirst_H[appmag24_idx]
    # ------------ K ------------ #
    print "Cumulative number counts for WFIRST K-band out to 24 mag:"
    cumulative_num_counts_wfirst_K = np.cumsum(total_num_dens_in_z_arr_wfirst_K)
    cumulative_num_counts_wfirst_K = cumulative_num_counts_wfirst_K * r * f_d4000 * f_acc
    print cumulative_num_counts_wfirst_K[appmag24_idx]

    # ------------ J ------------ #
    print "Cumulative number counts for Euclid J-band out to 24 mag:"
    cumulative_num_counts_euclid_J = np.cumsum(total_num_dens_in_z_arr_euclid_J)
    cumulative_num_counts_euclid_J = cumulative_num_counts_euclid_J * r * f_d4000 * f_acc
    print cumulative_num_counts_euclid_J[appmag24_idx]
    # ------------ H ------------ #
    print "Cumulative number counts for Euclid H-band out to 24 mag:"
    cumulative_num_counts_euclid_H = np.cumsum(total_num_dens_in_z_arr_euclid_H)
    cumulative_num_counts_euclid_H = cumulative_num_counts_euclid_H * r * f_d4000 * f_acc
    print cumulative_num_counts_euclid_H[appmag24_idx]
    # ------------ K ------------ #
    print "Cumulative number counts for Euclid K-band out to 24 mag:"
    cumulative_num_counts_euclid_K = np.cumsum(total_num_dens_in_z_arr_euclid_K)
    cumulative_num_counts_euclid_K = cumulative_num_counts_euclid_K * r * f_d4000 * f_acc
    print cumulative_num_counts_euclid_K[appmag24_idx]

    # ------------------------------------ 3-panel plot ------------------------------------ #
    # Plot number counts
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(10,28)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:, :8])
    ax2 = fig.add_subplot(gs[:, 10:18])
    ax3 = fig.add_subplot(gs[:, 20:])

    # Axes labels
    ax1.set_xlabel(r'$\rm m^{F105W}_{AB}$', fontsize=14)
    ax2.set_xlabel(r'$\rm m^{F125W}_{AB}$', fontsize=14)
    ax3.set_xlabel(r'$\rm m^{F160W}_{AB}$', fontsize=14)
    ax1.set_ylabel(r'$\rm N(<m)\ [deg^{-2}]$', fontsize=14)

    # Actual plotting
    ax1.scatter(app_mag_lim_arr, cumulative_num_counts_wfirst_J, marker='o', color='k', s=26, facecolor='None', label="WFIRST")
    ax1.scatter(app_mag_lim_arr, cumulative_num_counts_euclid_J, marker='^', color='k', s=26, facecolor='None', label="Euclid")

    ax2.scatter(app_mag_lim_arr, cumulative_num_counts_wfirst_H, marker='o', color='k', s=26, facecolor='None', label="WFIRST")
    ax2.scatter(app_mag_lim_arr, cumulative_num_counts_euclid_H, marker='^', color='k', s=26, facecolor='None', label="Euclid")

    ax3.scatter(app_mag_lim_arr, cumulative_num_counts_wfirst_K, marker='o', color='k', s=26, facecolor='None', label="WFIRST")
    ax3.scatter(app_mag_lim_arr, cumulative_num_counts_euclid_K, marker='^', color='k', s=26, facecolor='None', label="Euclid")

    # Scale, ticks and other stuff
    # First set up iterables
    all_axes = [ax1, ax2, ax3]
    
    zlow_wfirst_list = [zlow_wfirst_J, zlow_wfirst_H, zlow_wfirst_K]
    zhigh_wfirst_list = [zhigh_wfirst_J, zhigh_wfirst_H, zhigh_wfirst_K]

    zlow_euclid_list = [zlow_euclid_J, zlow_euclid_H, zlow_euclid_K]
    zhigh_euclid_list = [zhigh_euclid_J, zhigh_euclid_H, zhigh_euclid_K]

    M_star_list = [M_star_j, M_star_h, M_star_k]
    phi_star_list = [phi_star_j, phi_star_h, phi_star_k]
    alpha_list = [alpha_j, alpha_h, alpha_k]

    for u in range(3):

        ax = all_axes[u]

        zlow_wfirst = zlow_wfirst_list[u]
        zhigh_wfirst = zhigh_wfirst_list[u]
        zlow_euclid = zlow_euclid_list[u]
        zhigh_euclid = zhigh_euclid_list[u]

        M_star = M_star_list[u]
        phi_star = phi_star_list[u]
        alpha = alpha_list[u]

        ax.set_yscale('log')
        #ax.minorticks_on()
        ax.legend(loc=4, fontsize=13, frameon=False, handletextpad=-0.2)

        # Text on plot
        ax.text(0.02, 0.98, "Predicted number counts " + "\n" + \
            str(zlow_wfirst) + r"$\,\leq z_{\rm WFIRST} \leq\,$" + str(zhigh_wfirst), \
            verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=12)
    
        ax.text(0.02, 0.87, str(zlow_euclid) + r"$\,\leq z_{\rm Euclid} \leq\,$" + str(zhigh_euclid), \
            verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=12)

        ax.text(0.52, 0.6, r"$\rm M^* = $" + r"$\,$" + str(M_star), \
            verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=12)
        ax.text(0.52, 0.54, r"$\rm \phi^*\, [Mpc^{-3}\, mag^{-1}]$" + "\n" \
            + r"$ = $" + r"$\,$" + mr.convert_to_sci_not(phi_star), \
            verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=12)
        ax.text(0.52, 0.42, r"$\rm \alpha = $" + r"$\,$" + str(alpha), \
            verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=12)

        # Limits
        ax.set_xlim(16.0, 28.5)
        ax.set_ylim(1e-6, 1e6)

        # Ticks at every 0.5
        ax.xaxis.set_ticks(np.arange(16.0, 29.0, 0.5))

    # Add twin abs mag axis
    #ax2 = ax.twiny()

    # Convert apparent to absolute magnitudes
    #abs_mag_lim_arr = 

    fig.savefig(massive_figures_dir + "predicted_num_counts_wfirst_euclid.pdf", \
        dpi=300, bbox_inches='tight')
    #plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)