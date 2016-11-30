"""
    Plot Dn4000 vs stellar mass for all galaxies that matched between 3DHST and PEARS.
"""
import sys, os, glob
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

def get_dn4000(lam, spec):
    """
        Make sure the supplied lambda is in angstroms and the spectrum is in f_lambda.
    """

    arg3850 = np.argmin(abs(lam - 3850))
    arg3950 = np.argmin(abs(lam - 3950))
    arg4000 = np.argmin(abs(lam - 4000))
    arg4100 = np.argmin(abs(lam - 4100))

    fnu_plus = spec[arg4000:arg4100+1] * lam[arg4000:arg4100+1]**2 / 2.99792458e10
    fnu_minus = spec[arg3850:arg3950+1] * lam[arg3850:arg3950+1]**2 / 2.99792458e10

    dn4000 = np.trapz(fnu_plus, x=lam[arg4000:arg4100+1]) / np.trapz(fnu_minus, x=lam[arg3850:arg3950+1])
    
    return dn4000

def get_net_sig(fitsdata, filename):

    try:
        signal_sum = 0
        noise_sum = 0
        totalsum = 0
        cumsum = []
        
        if np.count_nonzero(fitsdata['ERROR']) != len(fitsdata['ERROR']):
            raise ZeroDivisionError
        
        sn = fitsdata['COUNT']/fitsdata['ERROR']
        sn_sorted = np.sort(sn)
        sn_sorted_reversed = sn_sorted[::-1]
    
        for _count_ in range(len(fitsdata)):
            idx = np.where(sn==sn_sorted_reversed[_count_])
            signal_sum += fitsdata['COUNT'][idx]
            noise_sum += fitsdata['ERROR'][idx]**2
            totalsum = signal_sum/np.sqrt(noise_sum)
            cumsum.append(totalsum)

        netsig = np.amax(cumsum)
        
        return netsig
            
    except ValueError, valerr_detail:
        #print filename
        print valerr_detail
        #print "The above spectrum will be given net sig of -99. Not sure of this error yet."
    except ZeroDivisionError, zeroerr_detail:
        #print filename
        #print zeroerr_detail
        #print "Division by zero! The net sig here cannot be trusted. Setting Net Sig to -99."
        return -99.0

def fileprep(pears_index, redshift):
    
    # Get the correct filename and the number of extensions
    filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    if not os.path.isfile(filename):
        filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

    fitsfile = fits.open(filename)
    n_ext = fitsfile[0].header['NEXTEND']

    specname = os.path.basename(filename)

    # Get highest netsig to find the spectrum to be added
    if n_ext > 1:
        netsiglist = []
        for count in range(n_ext):
            fitsdata = fitsfile[count+1].data
            netsig = get_net_sig(fitsdata, filename)
            netsiglist.append(netsig)
        netsiglist = np.array(netsiglist)
        maxnetsigarg = np.argmax(netsiglist)
        spec_toadd = fitsfile[maxnetsigarg+1].data
    elif n_ext == 1:
        spec_toadd = fitsfile[1].data
        
    # Now get the spectrum to be added
    lam_obs = spec_toadd['LAMBDA']
    flam_obs = spec_toadd['FLUX']
    ferr = spec_toadd['FERROR']
    contam = spec_toadd['CONTAM']
        
    # Subtract Contamination
    flam_obs = flam_obs - contam
        
    # First chop off the ends and only look at the observed spectrum from 6000A to 9500A
    arg6000 = np.argmin(abs(lam_obs - 6000))
    arg9500 = np.argmin(abs(lam_obs - 9500))
        
    lam_obs = lam_obs[arg6000:arg9500]
    flam_obs = flam_obs[arg6000:arg9500]
    ferr = ferr[arg6000:arg9500]
        
    # Now unredshift the spectrum
    lam_em = lam_obs / (1 + redshift)
    flam_em = flam_obs * (1 + redshift)
    # check the relations for unredshifting

    return lam_em, flam_em, ferr, specname

if __name__ == "__main__":

    light_speed = 2.99792458e10
    data_path = "/Users/baj/Documents/PEARS/data_spectra_only/"
    cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/color_stellarmass.txt',
                   dtype=None, names=True, skip_header=2)

    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    break_amp = np.zeros(len(pears_id))
    for i in range(len(pears_id)):
        redshift = photz[i]
        lam_em, flam_em, ferr, specname = fileprep(pears_id[i], redshift)

        break_amp[i] = get_dn4000(lam_em, flam_em)

    """
    high_dn4000 = np.where(break_amp > 2)[0]
    for j in range(len(high_dn4000)):

        filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
        if not os.path.isfile(filename):
            filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'
        fitsfile = fits.open(filename)
	print fitsfile[1].header
	sys.exit()
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\mathrm{log\left(\frac{M_s}{M_\odot}\right)}$', fontsize=14)
    ax.set_ylabel(r'$\mathrm{D_n}(4000)$', fontsize=14)
    ax.plot(stellarmass, break_amp, 'o', color='k', markeredgecolor='k', markersize=2)
    ax.set_ylim(-5,5)
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    fig.savefig('dn4000_mstar.png', dpi=300, bbox_inches='tight')
    #plt.show()
