from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys
import time
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'

def plotspectrum(lam, flam):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(lam, flam)

    plt.show()

    return None

if __name__ == '__main__':

    # Open spectra
    spec_hdu_old = fits.open(pears_datadir + 'h_pears_s_id65620.fits')
    spec_hdu_young = fits.open(pears_datadir + 'h_pears_n_id40498.fits')

    # assign arrays
    flam_old = spec_hdu_old[4].data['FLUX']
    ferr_old = spec_hdu_old[4].data['FERROR']
    contam_old = spec_hdu_old[4].data['CONTAM']
    lam_old = spec_hdu_old[4].data['LAMBDA']
    
    flam_young = spec_hdu_young[1].data['FLUX']
    ferr_young = spec_hdu_young[1].data['FERROR']
    contam_young = spec_hdu_young[1].data['CONTAM']
    lam_young = spec_hdu_young[1].data['LAMBDA']

    # Subtract contamination
    flam_old_consub = flam_old - contam_old
    flam_young_consub = flam_young - contam_young

    # Select only reliable data and plot. Select only data within 6000A to 9500A
    # old pop
    lam_low_idx = np.argmin(abs(lam_old - 6000))
    lam_high_idx = np.argmin(abs(lam_old - 9500))
    
    lam_old = lam_old[lam_low_idx:lam_high_idx+1]
    flam_old_consub = flam_old_consub[lam_low_idx:lam_high_idx+1]
    
    # young pop
    lam_low_idx = np.argmin(abs(lam_young - 6000))
    lam_high_idx = np.argmin(abs(lam_young - 9500))
    
    lam_young = lam_young[lam_low_idx:lam_high_idx+1]
    flam_young_consub = flam_young_consub[lam_low_idx:lam_high_idx+1]

    # Plot
    plotspectrum(lam_young, flam_young_consub)
    plotspectrum(lam_old, flam_old_consub)
    
    sys.exit(0)