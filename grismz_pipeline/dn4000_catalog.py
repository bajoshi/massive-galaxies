from __future__ import division

import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel

import sys
import os
import glob

import matplotlib.pyplot as plt
# modify rc Params
import matplotlib as mpl
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.sans-serif"] = ["Computer Modern Sans"]
#mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = [r'\usepackage{amsmath}']
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
pears_spectra_dir = home + "/Documents/PEARS/data_spectra_only/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
newcodes_dir = home + "/Desktop/FIGS/new_codes/"

"""
I think this code should also have the same contamination tolerances that grid_coadd has.
"""

def get_dn4000(lam, spec, spec_err):
    """
        Make sure the supplied lambda is in angstroms and the spectrum is in f_lambda -- IN THE REST FRAME!!!
    """

    arg3850 = np.argmin(abs(lam - 3850))
    arg3950 = np.argmin(abs(lam - 3950))
    arg4000 = np.argmin(abs(lam - 4000))
    arg4100 = np.argmin(abs(lam - 4100))

    fnu_plus = spec[arg4000:arg4100+1] * lam[arg4000:arg4100+1]**2 / 2.99792458e10
    fnu_minus = spec[arg3850:arg3950+1] * lam[arg3850:arg3950+1]**2 / 2.99792458e10

    dn4000 = np.trapz(fnu_plus, x=lam[arg4000:arg4100+1]) / np.trapz(fnu_minus, x=lam[arg3850:arg3950+1])

    delta_lam = 100
    spec_nu_err = spec_err * lam**2 / 2.99792458e10
    flux_nu_err_sqr = spec_nu_err**2

    if ((len(flux_nu_err_sqr[arg4000:arg4100+1])-1) >= 1) and ((len(flux_nu_err_sqr[arg3850:arg3950+1])-1) >= 1):
        sum_up_err = np.sqrt((delta_lam**2 / (2*(len(flux_nu_err_sqr[arg4000:arg4100+1])-1))) * \
            (4*sum(flux_nu_err_sqr[arg4000+1:arg4100+1]) + flux_nu_err_sqr[arg4000] + flux_nu_err_sqr[arg4100]))
        sum_low_err = np.sqrt((delta_lam**2 / (2*(len(flux_nu_err_sqr[arg3850:arg3950+1])-1))) * \
            (4*sum(flux_nu_err_sqr[arg3850+1:arg3950+1]) + flux_nu_err_sqr[arg3850] + flux_nu_err_sqr[arg3950]))
    elif ((len(flux_nu_err_sqr[arg4000:arg4100+1])-1) == 0) and ((len(flux_nu_err_sqr[arg3850:arg3950+1])-1) == 0):
        sum_up_err = np.sqrt(flux_nu_err_sqr[arg4000:arg4100+1])
        sum_low_err = np.sqrt(flux_nu_err_sqr[arg3850:arg3950+1])
    else:
        return np.nan, np.nan

    sum_low = np.trapz(fnu_minus, x=lam[arg3850:arg3950+1])
    sum_up = np.trapz(fnu_plus, x=lam[arg4000:arg4100+1])
    dn4000_err = (1/sum_low**2) * np.sqrt(sum_up_err**2 * sum_low**2 + sum_up**2 * sum_low_err**2)
    
    return dn4000, dn4000_err

def get_d4000(lam, spec, spec_err, interpolate_flag=True, makeplot=False):
    """
        Make sure the supplied lambda is in angstroms and the spectrum is in f_lambda -- IN THE REST FRAME!!!
    """

    speed_of_light = 2.99792458e10  # in cm/s

    """
    This interpolation method needs to be improved so that it checks first whether the wavelength 
    that it is trying to interpolate to already exists. Once this fix is done the interpolate flag
    in the function argument can be removed.
    """

    if interpolate_flag:
        # Get flux (f_nu) at the exact endpoints of the D4000 bandpasses
        # Convert to f_nu first
        fnu = spec * lam**2 / speed_of_light  # spec is in f_lam units
        fnu_err = spec_err * lam**2 / speed_of_light

        # First interpolate and figure out the flux at 
        # the exact endpoints of the D4000 bandpasses
        from scipy import interpolate

        # Now get new lambda and flux arrays with the exact
        # required lambda and flux values inserted
        # Find indices for insertion 
        insert_idx_3750 = np.where(lam > 3750.0)[0][0]
        insert_idx_3950 = np.where(lam > 3950.0)[0][0]
        insert_idx_4050 = np.where(lam > 4050.0)[0][0]
        # if there aren't any flux measurements above 4250 or the
        # closest measurement is below 4250 
        id_up = np.where(lam > 4250.0)[0]
        if not id_up.size:
            # If this is the case then insert the interpolated 
            # flux measurement just after the last element
            # numpy insert will insert the value before the provided index
            # This is not a typical case
            insert_idx_4250 = len(lam)
        else:
            insert_idx_4250 = np.where(lam > 4250.0)[0][0]

        # Make sure that the spectrum you have does not already have any flux
        # measurements at the exact wavelengths you are trying to insert
        checkexact_arg3750 = np.where(lam == 3750.0)[0]
        checkexact_arg3950 = np.where(lam == 3950.0)[0]
        checkexact_arg4050 = np.where(lam == 4050.0)[0]
        checkexact_arg4250 = np.where(lam == 4250.0)[0]

        # if there are no flux measurements at the exact wavelengths
        # This is the typical case
        if (not checkexact_arg3750.size) and (not checkexact_arg3950.size) \
        and (not checkexact_arg4050.size) and (not checkexact_arg4250.size):
            insert_idx = np.array([insert_idx_3750, insert_idx_3950, insert_idx_4050, insert_idx_4250])
            insert_wav = np.array([3750.0, 3950.0, 4050.0, 4250.0])

        else:
            # i.e. if there are one or more flux measurements at the exact D4000 bandpass wavelengths
            allcheck_args = [checkexact_arg3750, checkexact_arg3950, checkexact_arg4050, checkexact_arg4250]
            allinsert_idx = [insert_idx_3750, insert_idx_3950, insert_idx_4050, insert_idx_4250]
            allinsert_wav = [3750.0, 3950.0, 4050.0, 4250.0]
            insert_idx = []
            insert_wav = []
            for k in range(len(allcheck_args)):
                if not allcheck_args[k].size:
                    insert_idx.append(allinsert_idx[k])
                    insert_wav.append(allinsert_wav[k])

            insert_idx = np.asarray(insert_idx)
            insert_wav = np.asarray(insert_wav)

        # now do all the insertion
        lam_new = np.insert(lam, insert_idx, insert_wav)

        # now find the interpolating function
        f = interpolate.interp1d(lam, fnu, fill_value='extrapolate')
        fnu_insert_vals = f(insert_wav)
        fnu_new = np.insert(fnu, insert_idx, fnu_insert_vals)

        # Also find the error at the exact points
        # I'm basically just interpolating the error array as well
        fe = interpolate.interp1d(lam, fnu_err, fill_value='extrapolate')
        fnu_err_insert_vals = fe(insert_wav)
        fnu_err_new = np.insert(fnu_err, insert_idx, fnu_err_insert_vals)

        # plot to check
        if makeplot:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(lam_new, fnu_new, '.-', color='peru')
            ax.plot(lam, f_nu, '.-', color='steelblue')

            plt.show()

    else:
        # i.e. don't do interpolation if the user is sure that 
        # flux measurements at the exact endpoints of the D4000
        # bandpasses already exist
        lam_new = lam
        fnu_new = spec * lam**2 / speed_of_light
        fnu_err_new = spec_err * lam**2 / speed_of_light

    # -------------- D4000 computation ------------- # 
    arg3750 = np.where(lam_new == 3750.0)[0]
    arg3950 = np.where(lam_new == 3950.0)[0]
    arg4050 = np.where(lam_new == 4050.0)[0]
    arg4250 = np.where(lam_new == 4250.0)[0]

    # make sure these indices are scalars
    arg3750 = np.asscalar(arg3750)
    arg3950 = np.asscalar(arg3950)
    arg4050 = np.asscalar(arg4050)
    arg4250 = np.asscalar(arg4250)

    fnu_plus = fnu_new[arg4050:arg4250+1]
    fnu_minus = fnu_new[arg3750:arg3950+1]

    d4000 = np.trapz(fnu_plus, x=lam_new[arg4050:arg4250+1]) / np.trapz(fnu_minus, x=lam_new[arg3750:arg3950+1])

    delta_lam = 100
    flux_nu_err_sqr = fnu_err_new**2

    if ((len(flux_nu_err_sqr[arg4050:arg4250+1])-1) >= 1) and ((len(flux_nu_err_sqr[arg3750:arg3950+1])-1) >= 1):

        sum_up_err = np.sqrt((delta_lam**2 / (2*(len(flux_nu_err_sqr[arg4050:arg4250+1])-1))) * \
            (4*sum(flux_nu_err_sqr[arg4050+1:arg4250+1]) + flux_nu_err_sqr[arg4050] + flux_nu_err_sqr[arg4250]))
        sum_low_err = np.sqrt((delta_lam**2 / (2*(len(flux_nu_err_sqr[arg3750:arg3950+1])-1))) * \
            (4*sum(flux_nu_err_sqr[arg3750+1:arg3950+1]) + flux_nu_err_sqr[arg3750] + flux_nu_err_sqr[arg3950]))

    elif ((len(flux_nu_err_sqr[arg4050:arg4250+1])-1) == 0) and ((len(flux_nu_err_sqr[arg3750:arg3950+1])-1) == 0):

        sum_up_err = np.sqrt(flux_nu_err_sqr[arg4050:arg4250+1])
        sum_low_err = np.sqrt(flux_nu_err_sqr[arg3750:arg3950+1])

    else:
        return np.nan, np.nan

    sum_low = np.trapz(fnu_minus, x=lam_new[arg3750:arg3950+1])
    sum_up = np.trapz(fnu_plus, x=lam_new[arg4050:arg4250+1])
    d4000_err = (1/sum_low**2) * np.sqrt(sum_up_err**2 * sum_low**2 + sum_up**2 * sum_low_err**2)
    
    return d4000, d4000_err

def refine_redshift_old():
    """
    #This function will measure the 4000 break index assuming first that 
    #the supplied redshift is correct. Then it will shift the spectrum by
    #+- 200 A in the rest frame and measure the 4000 break index each time
    #it shifts. It will assume that the point at which it gets the maximum 
    #value for Dn4000 or D4000 is the correct shift and recalculate the 
    #redshift at that point and return this new redshift.
    #It will return the old redshift if it does not find a new maxima
    #for Dn4000 or D4000.

    # Find the average difference between the elements in the lambda array
    dlam = 0
    for i in range(len(lam) - 1):
        dlam += lam[i+1] - lam[i]

    avg_dlam = dlam / (len(lam) - 1)

    # Shift the lambda array
    #although it says shift spec in the comment next to the line of code
    #I'm just using the resultant shift in the spectrum 
    #to more easily identify the operation
    #keep in mind that the shift in spectrum is opposite 
    #to the shift in the lambda array
    #i.e. if I add a constant to all the elements in the 
    #lambda array then that will shift the spectrum to the blue....and vice versa

    #Still need to take care of the case where the break is close to the edge of the
    #wavelength coverage. This will result in the function being able to shift
    #the spectrum by unequal amounts on either side (or shift only on one side).

    if use_index == 'narrow':
        dn4000_arr = np.zeros()
        for k in range():

            # shift_spec_blue
            for i in range(len(lam) - 1):
                lam[i] = lam[i+1]
            lam[-1] = lam[-1] + avg_dlam
        
            # shift_spec_red
            for i in np.arange(len(lam),1,-1):
                lam[i] = lam[i-1]
            lam[0] = lam[0] - avg_dlam 

            dn4000_arr.append(get_dn4000(lam, spec, spec_err))

    elif use_index == 'normal':
        # shift_spec_blue
        for i in range(len(lam) - 1):
            lam[i] = lam[i+1]
        lam[-1] = lam[-1] + avg_dlam
        
        # shift_spec_red
        for i in np.arange(len(lam),1,-1):
            lam[i] = lam[i-1]
        lam[0] = lam[0] - avg_dlam 

        d4000_arr.append(get_d4000(lam, spec, spec_err))  
    """  
    print("Deprecated function!")
    print("This function is no longer used. Use refine_redshift() to get the refined refined redshifts.")
    print("Exiting...")
    sys.exit(0)

    return None
    
def refine_redshift(pearsid, z_old, fname, use_index='narrow'):

    z_pot_arr = np.arange(0.55, 1.3, 0.01)  # pot stands for potential

    dn4000_pot_arr = np.zeros(len(z_pot_arr))
    dn4000_err_pot_arr = np.zeros(len(z_pot_arr))
    d4000_pot_arr = np.zeros(len(z_pot_arr))
    d4000_err_pot_arr = np.zeros(len(z_pot_arr))

    count = 0
    for z in z_pot_arr:

        lam_em, flam_em, ferr, specname = gd.fileprep(pearsid, z, fname)

        dn4000_pot_arr[count], dn4000_err_pot_arr[count] = get_dn4000(lam_em, flam_em, ferr)
        d4000_pot_arr[count], d4000_err_pot_arr[count] = get_d4000(lam_em, flam_em, ferr)

        count += 1

    print(np.argmax(dn4000_pot_arr), np.argmax(d4000_pot_arr))

    z_arg = np.argmax(dn4000_pot_arr)
    z_new = z_pot_arr[z_arg]

    # for plotting (i.e. testing) purposes
    fig, ax = fcj.makefig(r"$\lambda$", r"$f_\lambda$")
    lam_em, flam_em, ferr, specname = gd.fileprep(pearsid, z_old, fname)
    fig, ax = plotspectrum(lam_em, flam_em, fig, ax)
    lam_em, flam_em, ferr, specname = gd.fileprep(pearsid, z_new, fname)
    fig, ax = plotspectrum(lam_em, flam_em, fig, ax, col='b')
    plt.show()

    del fig, ax

    fig, ax = fcj.makefig("z", r"$\mathrm{D_n(4000)}$")
    fig, ax = plotdn4000(z_pot_arr, dn4000_pot_arr, fig, ax, z_old, z_new)
    plt.show()

    del fig, ax

    """
    In these spectrum plots, you will notice that the flux level has also been shifted in the
    plot of the spectrum with the newer redshift. This might be surprising at first 
    glance given the expectation that a new redshfit should only shift the spectrum
    left or right while keeping the flux the same.
    A few seconds of thought (or just reading this comment) will point to the realization 
    that gd.fileprep() will also unredshift the flux by multiplying the observed flux 
    by (1+z) which is the reason for the shfit in the flux level for the spectrum with
    the new redshift.
    """

    return z_new

def plotspectrum(lam_em, flam_em, ferr_em, pearsid, pearsfield, d4000_temp, d4000_err_temp, netsig):

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # smooth before plotting
    gauss1d = Gaussian1DKernel(stddev=0.9)
    
    # Convolve data
    flam_em = convolve(flam_em, gauss1d)
    ferr_em = convolve(ferr_em, gauss1d)

    ax.plot(lam_em, flam_em, ls='-', color='b')
    ax.fill_between(lam_em, flam_em + ferr_em, flam_em - ferr_em, color='lightskyblue')

    ax.axvline(x=4000)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

    # add text to plot
    ax.text(0.6, 0.23, str(pearsid), \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=12)
    ax.text(0.6, 0.18, str(pearsfield), \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=12)
    ax.text(0.6, 0.13, r'$\mathrm{D(4000)\,=\,}$' + str("{:.3}".format(d4000_temp)) + r'$\pm$' + str("{:.3}".format(d4000_err_temp)),\
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=12)
    ax.text(0.6, 0.08, r'$\mathrm{NetSig\,=\,}$' + str("{:.3}".format(netsig)), \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=12)

    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    return None

def plotdn4000(z_pot_arr, dn4000_pot_arr, fig, ax, z_old, z_new):

    ax.plot(z_pot_arr, dn4000_pot_arr, 'o', markersize=3, color='b')
    ind = np.where(z_pot_arr == z_new)[0]
    ax.plot(z_pot_arr[ind], dn4000_pot_arr[ind], 'o', markersize=3, color='r')

    return fig, ax

def get_figs_dn4000(field, threed_cat, field_match, field_spc):

    # get stellar masses and the redshift indices for the galaxies    
    if field == 'gn1':
        gn1_mat = field_match
        stellarmass_gn1, redshift_gn1, redshift_type_gn1, use_phot_gn1, figsid_gn1, figsra_gn1, figsdec_gn1 = \
        getfigs.get_stellar_masses_redshifts(gn1_mat, 'gn1', threed_cat)
        redshift_gn1_indices = np.where((redshift_gn1 >= 1.2) & (redshift_gn1 <= 1.8))[0]
        spec = field_spc
        tot_range = len(figsid_gn1[redshift_gn1_indices])
        allids = figsid_gn1[redshift_gn1_indices]
        allredshifts = redshift_gn1[redshift_gn1_indices]
        allra = figsra_gn1[redshift_gn1_indices]
        alldec = figsdec_gn1[redshift_gn1_indices]

    elif field == 'gn2':
        gn2_mat = field_match
        stellarmass_gn2, redshift_gn2, redshift_type_gn2, use_phot_gn2, figsid_gn2, figsra_gn2, figsdec_gn2 = \
        getfigs.get_stellar_masses_redshifts(gn2_mat, 'gn2', threed_cat)
        redshift_gn2_indices = np.where((redshift_gn2 >= 1.2) & (redshift_gn2 <= 1.8))[0]
        spec = field_spc
        tot_range = len(figsid_gn2[redshift_gn2_indices])
        allids = figsid_gn2[redshift_gn2_indices]
        allredshifts = redshift_gn2[redshift_gn2_indices]
        allra = figsra_gn2[redshift_gn2_indices]
        alldec = figsdec_gn2[redshift_gn2_indices]

    elif field == 'gs1':
        gs1_mat = field_match
        stellarmass_gs1, redshift_gs1, redshift_type_gs1, use_phot_gs1, figsid_gs1, figsra_gs1, figsdec_gs1 = \
        getfigs.get_stellar_masses_redshifts(gs1_mat, 'gs1', threed_cat)
        redshift_gs1_indices = np.where((redshift_gs1 >= 1.2) & (redshift_gs1 <= 1.8))[0]
        spec = field_spc
        tot_range = len(figsid_gs1[redshift_gs1_indices])
        allids = figsid_gs1[redshift_gs1_indices]
        allredshifts = redshift_gs1[redshift_gs1_indices]
        allra = figsra_gs1[redshift_gs1_indices]
        alldec = figsdec_gs1[redshift_gs1_indices]

    dn4000_arr = []
    dn4000_err_arr = []
    d4000_arr = []
    d4000_err_arr = []
    figs_ra_arr = []
    figs_dec_arr = []
    redshift_arr = []
    figs_id_arr = []
    
    count_valid = 0
    for i in range(tot_range):

        figsid = allids[i]

        if field == 'gn1':
            figsid = figsid - 300000
        elif field == 'gn2':
            figsid = figsid - 400000
        elif field == 'gs1':
            figsid = figsid - 100000

        redshift = allredshifts[i]

        # Get observed quantities
        try:
            lam_obs      = spec["BEAM_%sA" % (figsid)].data["LAMBDA"]     # Wavelength (A) 
            avg_flux     = spec["BEAM_%sA" % (figsid)].data["AVG_FLUX"]   # Flux (erg/s/cm^2/A)
            avg_ferr     = spec["BEAM_%sA" % (figsid)].data["STD_FLUX"]   # Flux error (erg/s/cm^2/A)
            avg_wht_flux = spec["BEAM_%sA" % (figsid)].data["AVG_WFLUX"]  # Weighted Flux (erg/s/cm^2/A)
            avg_wht_ferr = spec["BEAM_%sA" % (figsid)].data["STD_WFLUX"]  # Weighted Flux error (erg/s/cm^2/A)

            # get the deredshifted quantitites first
            # both the fluxes above already have the contamination estimate subtracted from them
            flam_obs = avg_wht_flux
            ferr = avg_wht_ferr
                
            # First chop off the ends and only look at the observed spectrum from 8500A to 11500A
            arg8500 = np.argmin(abs(lam_obs - 8500))
            arg11500 = np.argmin(abs(lam_obs - 11500))
                
            lam_obs = lam_obs[arg8500:arg11500]
            flam_obs = flam_obs[arg8500:arg11500]
            ferr = ferr[arg8500:arg11500]
                
            # Now unredshift the spectrum
            lam_em = lam_obs / (1 + redshift)
            flam_em = flam_obs * (1 + redshift)

            # Get both break indices
            dn4000_temp, dn4000_err_temp = get_dn4000(lam_em, flam_em, ferr)
            d4000_temp, d4000_err_temp = get_d4000(lam_em, flam_em, ferr)

            # fill in arrays to be written out
            dn4000_arr.append(dn4000_temp)
            dn4000_err_arr.append(dn4000_err_temp)
            d4000_arr.append(d4000_temp)
            d4000_err_arr.append(d4000_err_temp)
            figs_id_arr.append(figsid)
            redshift_arr.append(redshift)
            figs_ra_arr.append(allra[i])
            figs_dec_arr.append(alldec[i])

            count_valid += 1

        except KeyError as e:
            continue

    print(count_valid, "galaxies included in dn4000 catalog for", field)

    dn4000_arr = np.asarray(dn4000_arr)
    dn4000_err_arr = np.asarray(dn4000_err_arr)
    d4000_arr = np.asarray(d4000_arr)
    d4000_err_arr = np.asarray(d4000_err_arr)
    figs_ra_arr = np.asarray(figs_ra_arr)
    figs_dec_arr = np.asarray(figs_dec_arr)
    redshift_arr = np.asarray(redshift_arr)
    figs_id_arr = np.asarray(figs_id_arr)

    data = np.array(zip(figs_id_arr, redshift_arr, figs_ra_arr, figs_dec_arr, dn4000_arr, dn4000_err_arr, d4000_arr, d4000_err_arr),\
                dtype=[('figs_id', int), ('photz', float), ('figs_ra', float), ('figs_dec', float), \
                ('dn4000_arr', float), ('dn4000_err_arr', float), ('d4000_arr', float), ('d4000_err_arr', float)])
    np.savetxt(stacking_analysis_dir + 'figs_' + field + '_4000break_catalog.txt', data, \
        fmt=['%d', '%.3f', '%.6f', '%.6f', '%.4f', '%.4f', '%.4f', '%.4f'], delimiter=' ',\
        header='Catalog for all galaxies that matched between 3DHST and FIGS within 1.2<z<1.8 for ' + field + '. \n' +\
        'figs_id redshift ra dec dn4000 dn4000_err d4000 d4000_err')

    return None

if __name__ == '__main__':

    # Using topcat to match
    topcat = False
    if topcat:
        cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_match_3dhst_topcat.txt', dtype=None, names=True)
        cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_match_3dhst_topcat.txt', dtype=None, names=True)
    else:
        cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', dtype=None, names=True, skip_header=1)
        cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', dtype=None, names=True, skip_header=1)

    allcats = [cat_n, cat_s]

    catcount = 0
    for cat in allcats:

        if catcount == 0:
            fieldname = 'GOODS-N'
            print('Starting with', len(cat), 'matched objects in', fieldname)
        elif catcount == 1:
            fieldname = 'GOODS-S'
            print('Starting with', len(cat), 'matched objects in', fieldname)

        if topcat:
            redshift_indices = np.where((cat['z_peak'] >= 0.6) & (cat['z_peak'] <= 1.235))[0]
            pears_id = cat['col1'][redshift_indices]  # col1 is pears id in the topcat saved catalog
            photz = cat['z_peak'][redshift_indices]
        else:
            redshift_indices = np.where((cat['zphot'] >= 0.6) & (cat['zphot'] <= 1.235))[0]
            pears_id = cat['pearsid'][redshift_indices]
            photz = cat['zphot'][redshift_indices]

        print(len(np.unique(pears_id)), "unique objects within", fieldname, "in redshift range")

        # create lists to write final data in and 
        # loop over all galaxies
        pears_id_write = []
        photz_write = []
        dn4000_arr = []
        dn4000_err_arr = []
        d4000_arr = []
        d4000_err_arr = []
        pears_ra = []
        pears_dec = []
        pearsfield = []
        redshift_source = []

        high_netsig_sample = 0

        # Loop over all spectra 
        pears_unique_ids, pears_unique_ids_indices = np.unique(pears_id, return_index=True)
        i = 0
        for current_pears_index, count in zip(pears_unique_ids, pears_unique_ids_indices):

            # Next two lines useful for debugging only a single object. Do not remove. Just uncomment.
            #if current_pears_index != 120306:
            #    continue

            redshift = photz[count]
            print("At object", current_pears_index, "in", fieldname)  # Line useful for debugging. Do not remove. Just uncomment.
            # get data and then d4000
            lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_pears_index, fieldname)

            if return_code == 0:
                print('Got return code 0. Skipping galaxy.', current_pears_index, fieldname)
                continue

            lam_em = lam_obs / (1 + redshift)
            flam_em = flam_obs * (1 + redshift)
            ferr_em = ferr_obs * (1 + redshift)

            #if (lam_em[0] > 3780) or (lam_em[-1] < 4220):
            #    # old limits (lam_em[0] > 3780) or (lam_em[-1] < 4220):
            #    # based on d4000 instead of dn4000
            #    # i've pushed the limits a little inward (> 50 A i.e. approx two spec measuremnt points), to be conservative, 
            #    # so that if there isn't an flux measurement at the exact end point wavelengths
            #    # but there is one nearby then the galaxy isn't skipped
            #    # skipping galaxy because there are too few or no flux measurements at the required wavelengths
            #    print("Skipping", current_pears_index, "in", fieldname,\
            #     "due to too few or no flux measurements at the required wavelengths. The end points of the wavelength array are",\
            #     lam_em[0], lam_em[-1])
            #    continue

            arg3750 = np.argmin(abs(lam_em - 3750))
            arg3950 = np.argmin(abs(lam_em - 3950))
            arg4050 = np.argmin(abs(lam_em - 4050))
            arg4250 = np.argmin(abs(lam_em - 4250))

            if (len(lam_em[arg3750:arg3950+1]) == 0) or (len(lam_em[arg4050:arg4250+1]) == 0):
                # based on d4000 instead of dn4000
                print("Skipping", current_pears_index, "in", fieldname,\
                 "due to no flux measurements at the required wavelengths. The end points of the wavelength array are",\
                 lam_em[0], lam_em[-1])
                continue

            if topcat:
                pears_ra.append(float(cat['col2'][redshift_indices][count]))  # col2 is the pears ra
                pears_dec.append(float(cat['col3'][redshift_indices][count]))  # col3 is the pears dec
                redshift_source.append('3DHST')
                # because the catalogs matched using topcat only matched to 3DHST. Didn't use Santini at all.
            else:
                pears_ra.append(float(cat['pearsra'][redshift_indices][count]))
                pears_dec.append(float(cat['pearsdec'][redshift_indices][count]))
                redshift_source.append(cat['source'][redshift_indices][count])

            dn4000_temp, dn4000_err_temp = get_dn4000(lam_em, flam_em, ferr_em)
            d4000_temp, d4000_err_temp = get_d4000(lam_em, flam_em, ferr_em)
            pearsfield.append(fieldname)
            pears_id_write.append(current_pears_index)
            photz_write.append(redshift)

            dn4000_arr.append(dn4000_temp)
            dn4000_err_arr.append(dn4000_err_temp)
            d4000_arr.append(d4000_temp)
            d4000_err_arr.append(d4000_err_temp)

            # Plotting. Skip the plot is NetSig is too low
            if netsig_chosen < 100:
                continue
            else:
                high_netsig_sample += 1
                #plotspectrum(lam_em, flam_em, ferr, current_pears_index, fieldname, d4000_temp, d4000_err_temp, netsig_chosen)

            i += 1

        # convert lists to numpy arrays for writing with savetxt
        pears_id_write = np.asarray(pears_id_write)
        photz_write = np.asarray(photz_write)
        dn4000_arr = np.asarray(dn4000_arr)
        dn4000_err_arr = np.asarray(dn4000_err_arr)
        d4000_arr = np.asarray(d4000_arr)
        d4000_err_arr = np.asarray(d4000_err_arr)
        pears_ra = np.asarray(pears_ra)
        pears_dec = np.asarray(pears_dec)
        pearsfield = np.asarray(pearsfield, dtype='|S7')
        redshift_source = np.asarray(redshift_source, dtype='|S7')

        data = np.array(zip(pears_id_write, pearsfield, photz_write, redshift_source, pears_ra, \
            pears_dec, dn4000_arr, dn4000_err_arr, d4000_arr, d4000_err_arr),\
            dtype=[('pears_id_write', int), ('pearsfield', '|S7'), \
            ('photz_write', float), ('redshift_source', '|S7'), ('pears_ra', float),\
            ('pears_dec', float), ('dn4000_arr', float), ('dn4000_err_arr', float), ('d4000_arr', float), ('d4000_err_arr', float)])
        if fieldname == 'GOODS-N':
            np.savetxt(massive_galaxies_dir + 'pears_4000break_catalog_' + fieldname + '.txt', data,\
            fmt=['%d', '%s', '%.4f', '%s', '%.6f', '%.6f', '%.4f', '%.4f', '%.4f', '%.4f'], delimiter=' ',\
            header='Catalog for all galaxies that matched between 3DHST and PEARS in ' + fieldname + '. \n' + \
            'pearsid field redshift zphot_source ra dec dn4000 dn4000_err d4000 d4000_err')
        elif fieldname == 'GOODS-S':
            np.savetxt(massive_galaxies_dir + 'pears_4000break_catalog_' + fieldname + '.txt', data, \
                fmt=['%d', '%s', '%.4f', '%s', '%.6f', '%.6f', '%.4f', '%.4f', '%.4f', '%.4f'], delimiter=' ', \
                header='Catalog for all galaxies that matched between CANDELS and PEARS in GOODS-S.' + '\n' + \
                'If a galaxy did not match with CANDELS then a matching in 3DHST was attempted.' + '\n' + \
                'the \'zphot_source\' column indicates the source of the photometric redshift.' + '\n' + \
                'pearsid field redshift zphot_source ra dec dn4000 dn4000_err d4000 d4000_err')

        print(len(np.isfinite(dn4000_arr)), len(np.isfinite(dn4000_err_arr)), len(np.isfinite(d4000_arr)), \
        len(np.isfinite(d4000_err_arr)))

        catcount += 1

        print("Total", high_netsig_sample, "galaxies with high netsig in", fieldname)

    """
    # Read in FIGS spc files
    gn1 = fits.open(home + '/Desktop/FIGS/spc_files/GN1_G102_2.combSPC.fits')
    gn2 = fits.open(home + '/Desktop/FIGS/spc_files/GN2_G102_2.combSPC.fits')
    gs1 = fits.open(home + '/Desktop/FIGS/spc_files/GS1_G102_2.combSPC.fits')

    # read 3dhst photometry cat
    threed_cat = fits.open(newcodes_dir + '3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat.FITS')

    # read in matched figs and 3dhst files
    # I am ignoring GS2 for now.
    gn1_mat = np.genfromtxt(massive_galaxies_dir + 'gn1_threedhst_matches.txt', dtype=None, names=True, skip_header=1)
    gn2_mat = np.genfromtxt(massive_galaxies_dir + 'gn2_threedhst_matches.txt', dtype=None, names=True, skip_header=1)
    gs1_mat = np.genfromtxt(massive_galaxies_dir + 'gs1_threedhst_matches.txt', dtype=None, names=True, skip_header=1)

    get_figs_dn4000('gn1', threed_cat, gn1_mat, gn1)
    get_figs_dn4000('gn2', threed_cat, gn2_mat, gn2)
    get_figs_dn4000('gs1', threed_cat, gs1_mat, gs1)
    """

    sys.exit(0)
