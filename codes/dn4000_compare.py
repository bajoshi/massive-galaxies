"""
    Code to compare measured dn4000 with the supplied values from BC03.
"""
import sys, os, glob
import numpy as np
import pyfits as pf

import matplotlib.pyplot as plt

def dn4000_bc03(lam, spec):
    """
        Make sure the supplied lambda is in angstroms and the spectrum is in L_lambda (i.e. BC03 units).
    """

    arg3850 = np.argmin(abs(lam - 3850))
    arg3950 = np.argmin(abs(lam - 3950))
    arg4000 = np.argmin(abs(lam - 4000))
    arg4100 = np.argmin(abs(lam - 4100))

    fnu_plus = spec[arg4000:arg4100+1] * lam[arg4000:arg4100+1]**2 / 2.99792458e10
    fnu_minus = spec[arg3850:arg3950+1] * lam[arg3850:arg3950+1]**2 / 2.99792458e10

    dn4000 = np.trapz(fnu_plus, x=lam[arg4000:arg4100+1]) / np.trapz(fnu_minus, x=lam[arg3850:arg3950+1])
    #dn4000 = np.trapz(spec[arg4000:arg4100+1], x=lam[arg4000:arg4100+1]) / np.trapz(spec[arg3850:arg3950+1], x=lam[arg3850:arg3950+1])
    
    return dn4000

def plot_hist_diff(diff):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('$\Delta$', fontsize=14)
    ax.set_ylabel('$N$', fontsize=14)

    ax.hist(diff, 30, histtype='bar', align='mid', alpha=0.5)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')

    plt.show()

if __name__ == "__main__":

    metals = ["m22","m32","m42","m52","m62","m72"]

    for metallicity in metals:
        cspout = '/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/'
        metalfolder = metallicity + '/'
    
        for filename in glob.glob(cspout + metalfolder + '*.3color'):
            tau = float(filename.split('/')[-1].split('_')[5][3:])
            tauV = float(filename.split('/')[-1].split('_')[3][4:])
            
            col = np.genfromtxt(filename, dtype=None, names=['age', 'b4_vn'], skip_header=29, usecols=(0,2))
            
            fitsfile = filename.replace('.3color', '.fits')
            h = pf.open(fitsfile, memmap=False)
            ages = h[2].data
            lam = h[1].data
            
            diff = np.ones(len(ages) - 1) * 99
            
            for i in range(1,len(ages)): # starting with the second spectrum in the set
                modeldn4000 = col['b4_vn'][i-1]
                datadn4000 = dn4000_bc03(lam, h[i+3].data)
                diff[i-1] = modeldn4000 - datadn4000
                #print i, np.log10(ages[i]), modeldn4000, datadn4000

            print np.median(diff), np.mean(diff), np.std(diff)
            plot_hist_diff(diff)

            #sys.exit()

