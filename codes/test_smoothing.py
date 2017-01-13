from __future__ import division

import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd

if __name__ == '__main__':

    filename = '/Users/baj/Documents/PEARS/data_spectra_only/h_pears_n_id36105.fits'
    fitsfile = fits.open(filename)

    current_id = 36105
    current_field = 'GOODS-N'
    current_redshift = 0.936

    lam_em_sm, flam_em_sm, ferr_sm, specname = gd.fileprep(current_id, current_redshift, current_field, apply_smoothing=True, width=1.5, kernel_type='gauss')
    lam_em, flam_em, ferr, specname = gd.fileprep(current_id, current_redshift, current_field, apply_smoothing=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.errorbar(lam_em, flam_em, yerr=ferr, fmt='o-', color='b')
    ax.errorbar(lam_em_sm, flam_em_sm, yerr=ferr_sm, fmt='-', color='r')

    plt.show()