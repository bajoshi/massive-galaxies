from __future__ import division

import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo
from joblib import Parallel, delayed
from scipy.interpolate import griddata, interp1d
from scipy.signal import fftconvolve
from scipy.integrate import simps

pears_datadir = "/home/bajoshi/pears_spectra/"

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
                #print reverse_mask[i], sn_sorted_reversed[i], _count_, signal_sum, totalsum  # Line useful for debugging. Do not remove. Just uncomment.
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

