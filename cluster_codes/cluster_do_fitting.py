from __future__ import division

import numpy as np
import numpy.ma as ma
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import griddata, interp1d
from scipy.signal import fftconvolve
from scipy.integrate import simps
import scipy.integrate as spint

import time

pears_datadir = "/home/bajoshi/pears_spectra/"
spz_outdir = "/home/bajoshi/spz_out/"

# Only for testing with firstlight
# Comment this out before copying code to Agave
# Uncomment above directory paths which are correct for Agave
#spz_outdir = '/Users/baj/Desktop/FIGS/massive-galaxies/cluster_results/'
#pears_datadir = '/Users/baj/Documents/PEARS/data_spectra_only/'

speed_of_light = 299792458e10  # angsroms per second
speed_of_light_kms = 299792.458  # km per s

# -------- Define cosmology -------- # 
H_0 = 69.6
omega_m0 = 0.286
omega_r0 = 8.24e-5
omega_lam0 = 0.714

def proper_distance(H_0, omega_m0, omega_r0, omega_lam0, ae):
    """
    This function will integrate 1/(a*a*H)
    between scale factor at emission to scale factor of 1.0.
    """
    p = lambda a: 1/(a*a*H_0*np.sqrt((omega_m0/a**3) + (omega_r0/a**4) + omega_lam0 + ((1 - omega_m0 - omega_r0 - omega_lam0)/a**2)))
    return spint.quadrature(p, ae, 1.0)

def get_lum_dist(redshift):
    """
    Returns luminosity distance in megaparsecs for a given redshift.
    """

    # Get the luminosity distance to the given redshift
    # Get proper distance and multiply by (1+z)
    scale_fac_to_z = 1 / (1+redshift)
    dp = proper_distance(H0, omega_m0, omega_r0, omega_lam0, scale_fac_to_z)[0]  # returns answer in Mpc/c
    dl = dp * speed_of_light_kms * (1+redshift)  # dl now in Mpc

    return dl

def get_avg_dlam(lam):

    dlam = 0
    for i in range(len(lam) - 1):
        dlam += lam[i+1] - lam[i]

    avg_dlam = dlam / (len(lam) - 1)

    return avg_dlam

def get_net_sig(*args):
    """
    This function simply needs either the fits extension that 
    contains the spectrum for which netsig is to be computed
    or 
    the counts and the errors on the counts as separate arrays.

    It should be able to figure out what you gave but if you
    give it the separate arrays then make sure that the counts 
    arrays comes before the counts error array.

    DO NOT give it any additional arguments or it will fail.
    """

    if len(args) == 1:
        fitsdata = args[0]
        count_arr = fitsdata['COUNT']
        error_arr = fitsdata['ERROR']
    elif len(args) == 2:
        count_arr = args[0]
        error_arr = args[1]

    # Make sure that the arrays are not empty to begin with
    if not count_arr.size:
        print "Returning -99.0 for NetSig due to empty signal and/or noise array for this object (or some PA for this object)."
        return -99.0
    if not error_arr.size:
        print "Returning -99.0 for NetSig due to empty signal and/or noise array for this object (or some PA for this object)."
        return -99.0

    # Also check that the error array does not have ALL zeros
    if np.all(error_arr == 0.0):
        #print "Returning -99.0 for NetSig due to noise array containing all 0 for this object (or some PA for this object)."
        return -99.0

    try:
        signal_sum = 0
        noise_sum = 0
        totalsum = 0
        cumsum = []
        
        sn = count_arr/error_arr
        # mask NaNs in this array to deal with the division by errors than are 0
        mask = ~np.isfinite(sn)
        sn = ma.array(sn, mask=mask)
        sn_sorted = np.sort(sn)
        sn_sorted_reversed = sn_sorted[::-1]
        reverse_mask = ma.getmask(sn_sorted_reversed)
        # I need the reverse mask for checking since I'm reversing the sn sorted array
        # and I need to only compute the netsig using unmasked elements.
        # This is because I need to check that the reverse sorted array will not have 
        # a blank element when I use the where function later causing the rest of the 
        # code block to mess up. Therefore, the mask I'm checking i.e. the reverse_mask
        # and the sn_sorted_reversed array need to have the same order.
        sort_arg = np.argsort(sn)
        sort_arg_rev = sort_arg[::-1]

        i = 0
        for _count_ in sort_arg_rev:
            # if it a masked element then don't do anything
            if reverse_mask[i]:
                i += 1
                continue
            else:
                signal_sum += count_arr[_count_]
                noise_sum += error_arr[_count_]**2
                totalsum = signal_sum/np.sqrt(noise_sum)
                #print reverse_mask[i], sn_sorted_reversed[i], _count_, signal_sum, totalsum
                # Above Line useful for debugging. Do not remove. Just uncomment.
                cumsum.append(totalsum)
                i += 1

        cumsum = np.asarray(cumsum)
        if not cumsum.size:
            print "Exiting due to empty cumsum array. More debugging needed."
            print "Cumulative sum array:", cumsum
            sys.exit(0)
        netsig = np.nanmax(cumsum)
        
        return netsig
            
    except ZeroDivisionError:
        logging.warning("Division by zero! The net sig here cannot be trusted. Setting Net Sig to -99.")
        print "Exiting. This error should not have come up anymore."
        sys.exit(0)
        return -99.0

def get_data(pears_index, field, check_contam=True):
    """
    Using code from fileprep in this function; not including 
    everything because it does a few things that I don't 
    actually need; also there is no contamination checking
    """

    # read in spectrum file
    data_path = pears_datadir
    # Get the correct filename and the number of extensions
    if field == 'GOODS-N':
        filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    elif field == 'GOODS-S':
        filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

    fitsfile = fits.open(filename)
    n_ext = fitsfile[0].header['NEXTEND']

    # Loop over all extensions and get the best PA
    # Get highest netsig to find the spectrum to be added
    if n_ext > 1:
        netsiglist = []
        palist = []
        for count in range(n_ext):
            #print "At PA", fitsfile[count+1].header['POSANG']  # Line useful for debugging. Do not remove. Just uncomment.
            fitsdata = fitsfile[count+1].data
            netsig = get_net_sig(fitsdata)
            netsiglist.append(netsig)
            palist.append(fitsfile[count+1].header['POSANG'])
            #print "At PA", fitsfile[count+1].header['POSANG'], "with NetSig", netsig  
            # Above line also useful for debugging. Do not remove. Just uncomment.
        netsiglist = np.array(netsiglist)
        maxnetsigarg = np.argmax(netsiglist)
        netsig_chosen = np.max(netsiglist)
        spec_toadd = fitsfile[maxnetsigarg+1].data
        pa_chosen = fitsfile[maxnetsigarg+1].header['POSANG']
    elif n_ext == 1:
        spec_toadd = fitsfile[1].data
        pa_chosen = fitsfile[1].header['POSANG']
        netsig_chosen = get_net_sig(fitsfile[1].data)
        
    # Now get the spectrum to be added
    lam_obs = spec_toadd['LAMBDA']
    flam_obs = spec_toadd['FLUX']
    ferr_obs = spec_toadd['FERROR']
    contam = spec_toadd['CONTAM']

    """
    In the next few lines within this function, I'm using a flag called
    return_code. This flag is used to tell the next part of the code,
    which is using the output from this function, if this function thinks 
    it returned anything useful. 1 = Useful. 0 = Not useful.
    """
 
    # Check that contamination level is not too high
    if check_contam:
        if np.nansum(contam) > 0.33 * np.nansum(flam_obs):
            print pears_index, " in ", field, " has an too high a level of contamination.",
            print "Contam =", np.nansum(contam) / np.nansum(flam_obs), " * F_lam. This galaxy will be skipped."
            return_code = 0
            fitsfile.close()
            return lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code

    # Check that input wavelength array is not empty
    if not lam_obs.size:
        print pears_index, " in ", field, " has an empty wav array. Returning empty array..."
        return_code = 0
        fitsfile.close()
        return lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code

    # Now chop off the ends and only look at the observed spectrum from 6000A to 9500A
    arg6500 = np.argmin(abs(lam_obs - 6000))
    arg9000 = np.argmin(abs(lam_obs - 9500))
        
    lam_obs = lam_obs[arg6500:arg9000]
    flam_obs = flam_obs[arg6500:arg9000]
    ferr_obs = ferr_obs[arg6500:arg9000]
    contam = contam[arg6500:arg9000]

    # subtract contamination if all okay
    flam_obs -= contam
    
    return_code = 1
    fitsfile.close()
    return lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code

def get_filt_zp(filtname):

    filtname_arr = np.array(['F435W', 'F606W', 'F775W', 'F850LP', 'F125W', 'F140W', 'F160W'])

    # Corresponding lists containing AB and ST zeropoints
    # The first 4 filters i.e. 'F435W', 'F606W', 'F775W', 'F850LP'
    # are ACS/WFC filters and the correct ZP calculation is yet to be done for these.
    # ACS zeropoints are time-dependent and therefore the zeropoint calculator has to be used.
    # Check: http://www.stsci.edu/hst/acs/analysis/zeropoints
    # For WFC3/IR: http://www.stsci.edu/hst/wfc3/analysis/ir_phot_zpt
    # This page: http://www.stsci.edu/hst/acs/analysis/zeropoints/old_page/localZeropoints
    # gives the old ACS zeropoints.
    # I'm going to use the old zeropoints for now.
    # While these are outdated, I have no way of knowing the dates of every single GOODS observation 
    # (actually I could figure out the dates from the archive) and then using the zeropoint associated 
    # with that date and somehow combining all the photometric data to get some average zeropoint?

    # ------ After talking to Seth about this: Since we only really care about the difference
    # between the STmag and ABmag zeropoints it won't matter very much that we are using the 
    # older zeropoints. The STmag and the ABmag zeropoints should chagne by the same amount. 
    # This should therefore be okay.
    filt_zp_st_list = [25.16823, 26.67444, 26.41699, 25.95456, 28.0203, 28.479, 28.1875]
    filt_zp_ab_list = [25.68392, 26.50512, 25.67849, 24.86663, 26.2303, 26.4524, 25.9463]

    filt_idx = np.where(filtname_arr == filtname)[0][0]

    filt_zp_st = filt_zp_st_list[filt_idx]
    filt_zp_ab = filt_zp_ab_list[filt_idx]

    return filt_zp_st, filt_zp_ab

def get_flam(filtname, cat_flux):
    """ Convert everything to f_lambda units """

    filt_zp_st, filt_zp_ab = get_filt_zp(filtname)

    cat_flux = float(cat_flux)  # because it should be a single float
    abmag = 25.0 - 2.5*np.log10(cat_flux)  # 3DHST fluxes are normalized to an ABmag ZP of 25.0
    stmag = filt_zp_st - filt_zp_ab + abmag

    flam = 10**(-1 * (stmag + 21.1) / 2.5)

    return flam

def get_flam_nonhst(filtname, cat_flux, vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam):

    """
    # Get filter response curves 
    # Both with respect to wavelength and frequency
    filt_resp, filt_nu, filt_lam = get_nonhst_filt_response(filtname)

    # INterpolate the Vega spectrum to the same grid as the filter curve in nu space
    vega_spec_fnu_interp = griddata(points=vega_nu, values=vega_spec_fnu, xi=filt_nu, method='linear')

    # Now find the AB magnitude of Vega in the filter
    vega_abmag_in_filt = -2.5 * np.log10(np.sum(vega_spec_fnu_interp * filt_resp) / np.sum(filt_resp)) - 48.6

    # Get AB mag of object
    cat_flux = float(cat_flux)  # because it should be a single float
    abmag = 25.0 - 2.5*np.log10(cat_flux)

    # Get Vega magnitude of object
    vegamag = abmag - vega_abmag_in_filt
    print abmag, vegamag, vega_abmag_in_filt

    # Convert vegamag to flam
    # INterpolate the Vega spectrum to the same grid as the filter curve in lambda space
    vega_spec_flam_interp = griddata(points=vega_lam, values=vega_spec_flam, xi=filt_lam, method='linear')
    vega_flam = np.sum(vega_spec_flam_interp * filt_resp)  # multiply by delta_lam?

    flam = vega_flam * 10**(-1 * vegamag / 2.5)

    # --------------- vegamag from fnu -------------- # 
    fnu = 10**(-1 * (abmag + 48.6) / 2.5)
    print fnu
    vegamag_from_fnu = -2.5 * np.log10(fnu / np.sum(vega_spec_fnu_interp * filt_resp))
    print vegamag_from_fnu
    """

    # Using just the stupid way of doing this for now
    cat_flux = float(cat_flux)  # because it should be a single float
    abmag = 25.0 - 2.5*np.log10(cat_flux)
    fnu = 10**(-1 * (abmag + 48.6) / 2.5)

    filtname_arr = np.array(['kpno_mosaic_u', 'irac1', 'irac2', 'irac3', 'irac4'])
    filt_idx = int(np.where(filtname_arr == filtname)[0])
    pivot_wavelengths = np.array([3582.0, 35500.0, 44930.0, 57310.0, 78720.0])  # in angstroms
    lp = pivot_wavelengths[filt_idx]

    flam = fnu * speed_of_light / lp**2

    return flam

def get_covmat(spec_wav, spec_flux, spec_ferr, lsf_covar_len, silent=True):

    galaxy_len_fac = lsf_covar_len
    # galaxy_len_fac includes the effect in correlation due to the 
    # galaxy morphology, i.e., for larger galaxies, flux data points 
    # need to be farther apart to be uncorrelated.
    base_fac = 0
    # base_fac includes the correlation effect due to the overlap 
    # between flux observed at adjacent spectral elements.
    # i.e., this amount of correlation in hte noise will 
    # exist even for a point source
    kern_len_fac = base_fac + galaxy_len_fac

    # Get number of spectral elements and define covariance mat
    N = len(spec_wav)
    covmat = np.identity(N)

    # Now populate the elements of the matrix
    len_fac = -1 / (2 * kern_len_fac**2)
    theta_0 = max(spec_ferr)**2
    for i in range(N):
        for j in range(N):

            if i == j:
                covmat[i,j] = 1.0/spec_ferr[i]**2
            else:
                covmat[i,j] = (1.0/theta_0) * np.exp(len_fac * (spec_wav[i] - spec_wav[j])**2)

    # Set everything below a certain lower limit to exactly zero
    inv_idx = np.where(covmat <= 1e-4 * theta_0)
    covmat[inv_idx] = 0.0

    return covmat

def get_alpha_chi2_covmat(total_models, flam_obs, model_spec_in_objlamgrid, covmat):

    # Now use the matrix computation to get chi2
    N = len(flam_obs)
    out_prod = np.outer(flam_obs, model_spec_in_objlamgrid.T.ravel())
    out_prod = out_prod.reshape(N, N, total_models)

    num_vec = np.sum(np.sum(out_prod * covmat[:, :, None], axis=0), axis=0)
    den_vec = np.zeros(total_models)
    alpha_vec = np.zeros(total_models)
    chi2_vec = np.zeros(total_models)
    for i in range(total_models):  # Get rid of this for loop as well, if you can
        den_vec[i] = np.sum(np.outer(model_spec_in_objlamgrid[i], model_spec_in_objlamgrid[i]) * covmat, axis=None)
        alpha_vec[i] = num_vec[i]/den_vec[i]
        col_vector = flam_obs - alpha_vec[i] * model_spec_in_objlamgrid[i]
        chi2_vec[i] = np.matmul(col_vector, np.matmul(covmat, col_vector))

    return alpha_vec, chi2_vec

def redshift_and_resample(model_comp_spec_lsfconv, z, total_models, model_lam_grid, resampling_lam_grid, resampling_lam_grid_length):

    # --------------- Redshift model --------------- #
    redshift_factor = 1.0 + z
    model_lam_grid_z = model_lam_grid * redshift_factor
    dl = get_lum_dist(z)  # in Mpc
    dl = dl * 3.086e24  # convert Mpc to cm
    model_comp_spec_redshifted = model_comp_spec_lsfconv / (4 * np.pi * dl * dl * redshift_factor)

    # --------------- Do resampling --------------- #
    # Define array to save modified models
    model_comp_spec_modified = np.zeros((total_models, resampling_lam_grid_length), dtype=np.float64)

    ### Zeroth element
    lam_step = resampling_lam_grid[1] - resampling_lam_grid[0]
    idx = np.where((model_lam_grid_z >= resampling_lam_grid[0] - lam_step) & (model_lam_grid_z < resampling_lam_grid[0] + lam_step))[0]
    model_comp_spec_modified[:, 0] = np.mean(model_comp_spec_redshifted[:, idx], axis=1)

    ### all elements in between
    for i in range(1, resampling_lam_grid_length - 1):
        idx = np.where((model_lam_grid_z >= resampling_lam_grid[i-1]) & (model_lam_grid_z < resampling_lam_grid[i+1]))[0]
        model_comp_spec_modified[:, i] = np.mean(model_comp_spec_redshifted[:, idx], axis=1)

    ### Last element
    lam_step = resampling_lam_grid[-1] - resampling_lam_grid[-2]
    idx = np.where((model_lam_grid_z >= resampling_lam_grid[-1] - lam_step) & (model_lam_grid_z < resampling_lam_grid[-1] + lam_step))[0]
    model_comp_spec_modified[:, -1] = np.mean(model_comp_spec_redshifted[:, idx], axis=1)

    return model_comp_spec_modified

def get_chi2(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs, \
    covmat, all_filt_flam_model, model_comp_spec_mod, model_resampling_lam_grid, \
    total_models, lsf_covar_len, use_broadband=True):

    # chop the model to be consistent with the objects lam grid
    model_lam_grid_indx_low = np.argmin(np.absolute(model_resampling_lam_grid - grism_lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(np.absolute(model_resampling_lam_grid - grism_lam_obs[-1]))
    model_spec_in_objlamgrid = model_comp_spec_mod[:, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # make sure that the arrays are the same length
    if int(model_spec_in_objlamgrid.shape[1]) != len(grism_lam_obs):
        print "Arrays of unequal length. Must be fixed before moving forward. Exiting..."
        print "Model spectrum array shape:", model_spec_in_objlamgrid.shape
        print "Object spectrum length:", len(grism_lam_obs)
        sys.exit(0)

    if use_broadband:
        # For both data and model, combine grism+photometry into one spectrum.
        # The chopping above has to be done before combining the grism+photometry
        # because todo the insertion correctly the model and grism wavelength
        # grids have to match.

        # Convert the model array to a python list of lists
        # This has to be done because np.insert() returns a new changed array
        # with the new value inserted but I cannot assign it back to the old
        # array because that changes the shape. This works for the grism arrays
        # since I'm simply using variable names to point to them but since the
        # model array is 2D I'm using indexing and that causes the np.insert()
        # statement to throw an error.
        model_spec_in_objlamgrid_list = []
        for j in range(total_models):
            model_spec_in_objlamgrid_list.append(model_spec_in_objlamgrid[j].tolist())

        count = 0
        combined_lam_obs = grism_lam_obs
        combined_flam_obs = grism_flam_obs
        combined_ferr_obs = grism_ferr_obs
        for phot_wav in phot_lam_obs:

            if phot_wav < combined_lam_obs[0]:
                lam_obs_idx_to_insert = 0

            elif phot_wav > combined_lam_obs[-1]:
                lam_obs_idx_to_insert = len(combined_lam_obs)

            else:
                lam_obs_idx_to_insert = np.where(combined_lam_obs > phot_wav)[0][0]

            # For grism
            combined_lam_obs = np.insert(combined_lam_obs, lam_obs_idx_to_insert, phot_wav)
            combined_flam_obs = np.insert(combined_flam_obs, lam_obs_idx_to_insert, phot_flam_obs[count])
            combined_ferr_obs = np.insert(combined_ferr_obs, lam_obs_idx_to_insert, phot_ferr_obs[count])

            # For model
            for i in range(total_models):
                model_spec_in_objlamgrid_list[i] = \
                np.insert(model_spec_in_objlamgrid_list[i], lam_obs_idx_to_insert, all_filt_flam_model[i, count])

            count += 1

        # Convert back to numpy array
        model_spec_in_objlamgrid = np.asarray(model_spec_in_objlamgrid_list)

        # Get covariance matrix
        covmat = get_covmat(combined_lam_obs, combined_flam_obs, combined_ferr_obs, lsf_covar_len)
        alpha_, chi2_ = get_alpha_chi2_covmat(total_models, combined_flam_obs, model_spec_in_objlamgrid, covmat)
        print "Min chi2 for redshift:", min(chi2_)

    else:
        # Get covariance matrix
        covmat = get_covmat(grism_lam_obs, grism_flam_obs, grism_ferr_obs, lsf_covar_len)
        alpha_, chi2_ = get_alpha_chi2_covmat(total_models, grism_flam_obs, model_spec_in_objlamgrid, covmat)
        print "Min chi2 for redshift:", min(chi2_)

    return chi2_, alpha_

def get_chi2_alpha_at_z(z, grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs, covmat, \
    model_lam_grid, model_comp_spec_lsfconv, all_model_flam, z_model_arr, phot_fin_idx, \
    resampling_lam_grid, resampling_lam_grid_length, total_models, start_time, lsf_covar_len, ub):

    # ------------ Get photomtery for model by convolving with filters ------------- #
    z_idx = np.where(z_model_arr == z)[0]

    # and because for some reason (in some cases probably 
    # due to floating point roundoff) it does not find matches 
    # in the model redshift array, I need this check here.
    if not z_idx.size:
        z_idx = np.argmin(abs(z_model_arr - z))

    all_filt_flam_model = all_model_flam[:, z_idx, :]
    all_filt_flam_model = all_filt_flam_model[phot_fin_idx, :]
    all_filt_flam_model = all_filt_flam_model.reshape(len(phot_fin_idx), total_models)

    all_filt_flam_model_t = all_filt_flam_model.T

    # ------------- Now do the modifications for the grism data and get a chi2 using both grism and photometry ------------- #
    # first modify the models at the current redshift to be able to compare with data
    model_comp_spec_modified = \
    redshift_and_resample(model_comp_spec_lsfconv, z, total_models, model_lam_grid, resampling_lam_grid, resampling_lam_grid_length)
    print "Model mods done at current z:", z
    print "Total time taken up to now --", time.time() - start_time, "seconds."

    # ------------- Now do the chi2 computation ------------- #
    if ub:
        chi2_temp, alpha_temp = get_chi2(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs,\
            covmat, all_filt_flam_model_t, model_comp_spec_modified, resampling_lam_grid, \
            total_models, lsf_covar_len, use_broadband=True)
    else:
        chi2_temp, alpha_temp = get_chi2(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs,\
            covmat, all_filt_flam_model_t, model_comp_spec_modified, resampling_lam_grid, \
            total_models, lsf_covar_len, use_broadband=False)

    return chi2_temp, alpha_temp

def get_pz(chi2_map, z_arr_to_check):

    # Convert chi2 to likelihood
    likelihood = np.exp(-1 * chi2_map / 2)

    # Normalize likelihood function
    norm_likelihood = likelihood / np.sum(likelihood)

    # Get p(z)
    pz = np.zeros(len(z_arr_to_check))

    for i in range(len(z_arr_to_check)):
        pz[i] = np.sum(norm_likelihood[i])

    return pz

def do_fitting(grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs, covmat, \
    lsf, resampling_lam_grid, resampling_lam_grid_length, all_model_flam, phot_fin_idx, \
    model_lam_grid, total_models, model_comp_spec, start_time, obj_id, obj_field, specz, photoz, \
    log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr, \
    use_broadband=True, single_galaxy=False, for_loop_method='sequential'):

    """
    All models are redshifted to each of the redshifts in the list defined below,
    z_arr_to_check. Then the model modifications are done at that redshift.

    For each iteration through the redshift list it computes a chi2 for each model.
    """

    savedir_spz = spz_outdir  # Required to save p(z) curve and z_arr
    savedir_grismz = spz_outdir  # Required to save p(z) curve and z_arr

    # Set up redshift grid to check
    z_arr_to_check = np.arange(0.3, 1.5, 0.01)

    # The model mags were computed on a finer redshift grid
    # So make sure to get the z_idx correct
    z_model_arr = np.arange(0.0, 6.0, 0.005)

    ####### ------------------------------------ Main loop through redshfit array ------------------------------------ #######
    # Loop over all redshifts to check
    # set up chi2 and alpha arrays
    chi2 = np.empty((len(z_arr_to_check), total_models))
    alpha = np.empty((len(z_arr_to_check), total_models))

    # First do the convolution with the LSF
    model_comp_spec_lsfconv = np.zeros(model_comp_spec.shape)
    for i in range(total_models):
        model_comp_spec_lsfconv[i] = fftconvolve(model_comp_spec[i], lsf, mode = 'same')

    print "Convolution done.",
    print "Total time taken up to now --", time.time() - start_time, "seconds."

    # To get the covariance length, fit the LSF with a gaussian
    # and then the cov length is simply the best fit std dev.
    lsf_length = len(lsf)
    gauss_init = models.Gaussian1D(amplitude=np.max(lsf), mean=lsf_length/2, stddev=lsf_length/4)
    fit_gauss = fitting.LevMarLSQFitter()
    x_arr = np.arange(lsf_length)
    g = fit_gauss(gauss_init, x_arr, lsf)
    # get fit std.dev.
    lsf_std =  g.parameters[2]
    lsf_covar_len = 3*lsf_std
    print "Grism covariance length based on fitting Gaussian to LSF is:", lsf_covar_len
    # i.e., if any pair of spectral elements are 3-sigma away
    # from each other then their data is uncorrelated. 

    count = 0
    for z in z_arr_to_check:
        chi2[count], alpha[count] = get_chi2_alpha_at_z(z, \
            grism_flam_obs, grism_ferr_obs, grism_lam_obs, phot_flam_obs, phot_ferr_obs, phot_lam_obs, covmat, \
            model_lam_grid, model_comp_spec_lsfconv, all_model_flam, z_model_arr, phot_fin_idx, \
            resampling_lam_grid, resampling_lam_grid_length, total_models, start_time, lsf_covar_len, use_broadband)

        count += 1

    ####### -------------------------------------- Min chi2 and best fit params -------------------------------------- #######
    # Sort through the chi2 and make sure that the age is physically meaningful
    sortargs = np.argsort(chi2, axis=None)  # i.e. it will use the flattened array to sort

    for k in range(len(chi2.ravel())):

        # Find the minimum chi2
        min_idx = sortargs[k]
        min_idx_2d = np.unravel_index(min_idx, chi2.shape)

        # Get the best fit model parameters
        # first get the index for the best fit
        model_idx = int(min_idx_2d[1])

        age = log_age_arr[model_idx] # float(bc03_all_spec_hdulist[model_idx + 1].header['LOGAGE'])

        current_z = z_arr_to_check[min_idx_2d[0]]
        age_at_z = cosmo.age(current_z).value * 1e9  # in yr

        # Colors and stellar mass
        ub_col = ub_col_arr[model_idx] 
        bv_col = bv_col_arr[model_idx] 
        vj_col = vj_col_arr[model_idx] 
        template_ms = ms_arr[model_idx]

        tau = tau_gyr_arr[model_idx]
        tauv = tauv_arr[model_idx]

        # now check if the age is meaningful
        # This condition is essentially saying that the model age has to be at least 
        # 100 Myr younger than the age of the Universe at the given redshift and at 
        # the same time it needs to be at least 10 Myr in absolute terms
        if (age < np.log10(age_at_z - 1e8)) and (age > 9 + np.log10(0.01)):
            # If the age is meaningful then you don't need to do anything
            # more. Just break out of the loop. the best fit parameters have
            # already been assigned to variables. This assignment is done before 
            # the if statement to make sure that there are best fit parameters 
            # even if the loop is broken out of in the first iteration.
            break

    print "Minimum chi2 from sorted indices which also agrees with the age of the Universe:", "{:.4}".format(chi2[min_idx_2d])
    print "Minimum chi2 from np.min():", "{:.4}".format(np.min(chi2))
    z_grism = z_arr_to_check[min_idx_2d[0]]

    print "Current best fit log(age [yr]):", "{:.4}".format(age)
    print "Current best fit Tau [Gyr]:", "{:.4}".format(tau)
    print "Current best fit Tau_V:", tauv

    ############# -------------------------- Errors on z and other derived params ----------------------------- #############
    min_chi2 = chi2[min_idx_2d]
    # See Andrae+ 2010;arXiv:1012.3754. The number of d.o.f. for non-linear models 
    # is not well defined and reduced chi2 should really not be used.
    # Seth's comment: My model is actually linear. Its just a factor 
    # times a set of fixed points. And this is linear, because each
    # model is simply a function of lambda, which is fixed for a given 
    # model. So every model only has one single free parameter which is
    # alpha i.e. the vertical scaling factor; that's true since alpha is 
    # the only one I'm actually solving for to get a min chi2. I'm not 
    # varying the other parameters - age, tau, av, metallicity, or 
    # z_grism - within a given model. Therefore, I can safely use the 
    # methods described in Andrae+ 2010 for linear models.

    # Also now that we're using the covariance matrix approach
    # we should use the correct dof since the effective degrees
    # is freedom is smaller. 
    grism_dof = len(grism_lam_obs) / lsf_covar_len 
    if use_broadband:
        dof = grism_dof + len(phot_lam_obs) - 1  # i.e., total effective independent data points minus the single fitting parameter
    else:
        dof = grism_dof - 1  # i.e., total effective independent data points minus the single fitting parameter

    chi2_red = chi2 / dof
    chi2_red_error = np.sqrt(2/dof)
    min_chi2_red = min_chi2 / dof
    chi2_red_2didx = np.where((chi2_red >= min_chi2_red - chi2_red_error) & (chi2_red <= min_chi2_red + chi2_red_error))

    # use first dimension indices to get error on grism-z
    z_grism_range = z_arr_to_check[chi2_red_2didx[0]]

    low_z_lim = np.min(z_grism_range)
    upper_z_lim = np.max(z_grism_range)
    print "Min z_grism within 1-sigma error:", low_z_lim
    print "Max z_grism within 1-sigma error:", upper_z_lim

    # Save p(z), chi2 map, and redshift grid
    if use_broadband:
        pz = get_pz(chi2/dof, z_arr_to_check)
        np.save(savedir_spz + obj_field + '_' + str(obj_id) + '_spz_z_arr.npy', z_arr_to_check)
        np.save(savedir_spz + obj_field + '_' + str(obj_id) + '_spz_pz.npy', pz)
    else:
        pz = get_pz(chi2/dof, z_arr_to_check)
        np.save(savedir_grismz + obj_field + '_' + str(obj_id) + '_zg_z_arr.npy', z_arr_to_check)
        np.save(savedir_grismz + obj_field + '_' + str(obj_id) + '_zg_pz.npy', pz)

    z_wt = np.sum(z_arr_to_check * pz)
    print "Weighted z:", "{:.3}".format(z_wt)
    print "Grism redshift:", z_grism
    print "Ground-based spectroscopic redshift [-99.0 if it does not exist]:", specz
    print "Photometric redshift:", photoz

    bestalpha = alpha[min_idx_2d]
    print "Vertical scaling factor for best fit model:", bestalpha

    return z_grism, z_wt, low_z_lim, upper_z_lim, min_chi2_red, bestalpha, model_idx, age, tau, (tauv/1.086)

def get_chi2_alpha_at_z_photoz_lookup(z, all_filt_flam_model, phot_flam_obs, phot_ferr_obs):

    all_filt_flam_model_t = all_filt_flam_model.T

    # ------------------------------------ Now do the chi2 computation ------------------------------------ #
    # compute alpha and chi2
    alpha_ = np.sum(phot_flam_obs * all_filt_flam_model_t / (phot_ferr_obs**2), axis=1) / np.sum(all_filt_flam_model_t**2 / phot_ferr_obs**2, axis=1)
    chi2_ = np.sum(((phot_flam_obs - (alpha_ * all_filt_flam_model).T) / phot_ferr_obs)**2, axis=1)

    return chi2_, alpha_

def do_photoz_fitting_lookup(phot_flam_obs, phot_ferr_obs, phot_lam_obs, \
    model_lam_grid, total_models, model_comp_spec, start_time,\
    obj_id, obj_field, all_model_flam, phot_fin_idx, specz, savedir, \
    log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr):
    """
    All models are redshifted to each of the redshifts in the list defined below,
    z_arr_to_check. Then the model modifications are done at that redshift.

    For each iteration through the redshift list it computes a chi2 for each model.
    """

    # Set up redshift grid to check
    z_arr_to_check = np.arange(0.3, 1.5, 0.01)

    # The model mags were computed on a finer redshift grid
    # So make sure to get the z_idx correct
    z_model_arr = np.arange(0.0, 6.0, 0.005)

    ####### ------------------------------------ Main loop through redshfit array ------------------------------------ #######
    # Loop over all redshifts to check
    # set up chi2 and alpha arrays
    chi2 = np.empty((len(z_arr_to_check), total_models))
    alpha = np.empty((len(z_arr_to_check), total_models))

    count = 0
    for z in z_arr_to_check:

        z_idx = np.where(z_model_arr == z)[0]

        # and because for some reason it does not find matches 
        # in the model redshift array, I need this check here.
        if not z_idx.size:
            z_idx = np.argmin(abs(z_model_arr - z))

        all_filt_flam_model = all_model_flam[:, z_idx, :]
        all_filt_flam_model = all_filt_flam_model[phot_fin_idx, :]
        all_filt_flam_model = all_filt_flam_model.reshape(len(phot_fin_idx), total_models)

        chi2[count], alpha[count] = get_chi2_alpha_at_z_photoz_lookup(z, all_filt_flam_model, phot_flam_obs, phot_ferr_obs)

        count += 1

    # Check for all NaNs in chi2 array
    # For now skipping all galaxies that have any NaNs in them.
    if len(np.where(np.isfinite(chi2.ravel()))[0]) != len(chi2.ravel()):
        print "Chi2 has NaNs. Skiiping galaxy for now."
        return -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0

    ####### -------------------------------------- Min chi2 and best fit params -------------------------------------- #######
    # Sort through the chi2 and make sure that the age is physically meaningful
    sortargs = np.argsort(chi2, axis=None)  # i.e. it will use the flattened array to sort

    for k in xrange(len(chi2.ravel())):

        # Find the minimum chi2
        min_idx = sortargs[k]
        min_idx_2d = np.unravel_index(min_idx, chi2.shape)

        # Get the best fit model parameters
        # first get the index for the best fit
        model_idx = int(min_idx_2d[1])

        age = log_age_arr[model_idx]

        current_z = z_arr_to_check[min_idx_2d[0]]
        age_at_z = cosmo.age(current_z).value * 1e9  # in yr

        # Colors and stellar mass
        ub_col = ub_col_arr[model_idx] 
        bv_col = bv_col_arr[model_idx] 
        vj_col = vj_col_arr[model_idx] 
        template_ms = ms_arr[model_idx]

        tau = tau_gyr_arr[model_idx]
        tauv = tauv_arr[model_idx]

        # now check if the age is meaningful
        # This condition is essentially saying that the model age has to be at least 
        # 100 Myr younger than the age of the Universe at the given redshift and at 
        # the same time it needs to be at least 10 Myr in absolute terms
        if (age < np.log10(age_at_z - 1e8)) and (age > 9 + np.log10(0.01)):
            # If the age is meaningful then you don't need to do anything
            # more. Just break out of the loop. the best fit parameters have
            # already been assigned to variables. This assignment is done before 
            # the if statement to make sure that there are best fit parameters 
            # even if the loop is broken out of in the first iteration.
            break

    zp_minchi2 = z_arr_to_check[min_idx_2d[0]]

    print "Current best fit log(age [yr]):", "{:.4}".format(age)
    print "Current best fit Tau [Gyr]:", "{:.4}".format(tau)
    print "Current best fit Tau_V:", tauv

    ############# -------------------------- Errors on z and other derived params ----------------------------- #############
    min_chi2 = chi2[min_idx_2d]
    # See Andrae+ 2010;arXiv:1012.3754. The number of d.o.f. for non-linear models 
    # is not well defined and reduced chi2 should really not be used.
    # Seth's comment: My model is actually linear. Its just a factor 
    # times a set of fixed points. And this is linear, because each
    # model is simply a function of lambda, which is fixed for a given 
    # model. So every model only has one single free parameter which is
    # alpha i.e. the vertical scaling factor; that's true since alpha is 
    # the only one I'm actually solving for to get a min chi2. I'm not 
    # varying the other parameters - age, tau, av, metallicity, or 
    # z - within a given model. Therefore, I can safely use the 
    # methods described in Andrae+ 2010 for linear models.
    dof = len(phot_lam_obs) - 1  # i.e. total data points minus the single fitting parameter

    chi2_red = chi2 / dof
    chi2_red_error = np.sqrt(2/dof)
    min_chi2_red = min_chi2 / dof
    chi2_red_2didx = np.where((chi2_red >= min_chi2_red - chi2_red_error) & (chi2_red <= min_chi2_red + chi2_red_error))
    print "Minimum chi2 (reduced):", "{:.4}".format(min_chi2_red)

    # use first dimension indices to get error on zphot
    z_range = z_arr_to_check[chi2_red_2didx[0]]

    low_z_lim = np.min(z_range)
    upper_z_lim = np.max(z_range)
    print "Min z within 1-sigma error:", low_z_lim
    print "Max z within 1-sigma error:", upper_z_lim

    # Save pz and z_arr 
    np.save(savedir + obj_field + '_' + str(obj_id) + '_photoz_z_arr.npy', z_arr_to_check)
    pz = get_pz(chi2/dof, z_arr_to_check)
    # Save p(z)
    np.save(savedir + obj_field + '_' + str(obj_id) + '_photoz_pz.npy', pz)

    zp = np.sum(z_arr_to_check * pz)
    print "Ground-based spectroscopic redshift [-99.0 if it does not exist]:", specz
    #print "Previous photometric redshift from 3DHST:", photoz
    print "Photometric redshift from min chi2 from this code:", "{:.3f}".format(zp_minchi2)
    print "Photometric redshift (weighted) from this code:", "{:.3f}".format(zp)

    # Stellar mass
    bestalpha = alpha[min_idx_2d]
    print "Min idx 2d:", min_idx_2d
    print "Alpha for best-fit model:", bestalpha

    ms = template_ms / bestalpha
    print "Template mass [normalized to 1 sol]:", template_ms
    print "Stellar mass for galaxy [M_sol]:", "{:.2e}".format(ms)

    # Rest frame f_lambda values
    zbest_idx = np.argmin(abs(z_model_arr - zp))
    print "Rest-frame f_lambda values:", all_model_flam[:, zbest_idx, min_idx_2d[1]]
    print "Rest-frame U-B color:", ub_col
    print "Rest-frame B-V color:", bv_col
    print "Rest-frame U-V color:", ub_col - bv_col
    print "Rest-frame V-J color:", vj_col

    return zp_minchi2, zp, low_z_lim, upper_z_lim, min_chi2_red, bestalpha, model_idx, age, tau, (tauv/1.086)

