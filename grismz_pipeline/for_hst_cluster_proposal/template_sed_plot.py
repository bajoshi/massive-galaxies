from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

def plot_template_sed():

    return None

def main():

    # ---------------------------------- Read in the filters from Seth ---------------------------------- #
    f435w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/F435W_ACS.res', \
        dtype=None, names=['wav', 'trans'])
    f606w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/F606W_ACS.res', \
        dtype=None, names=['wav', 'trans'])
    f814w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/F814W_ACS.txt', \
        dtype=None, names=['wav', 'trans'])
    g141  = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/for_hst_cluster_proposal/g141l_tot.txt', \
        dtype=None, names=['wav', 'trans'])
    f140w = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f140w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])

    # Spitzer/IRAC channels
    irac1 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac1.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac2 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac2.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac3 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac3.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac4 = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac4.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)

    # IRAC wavelengths are in microns # convert to angstroms
    irac1['wav'] *= 1e4
    irac2['wav'] *= 1e4
    irac3['wav'] *= 1e4
    irac4['wav'] *= 1e4

    # ---------------------------------- Now get the templates ---------------------------------- #

    # ---------------------------------- Convolve templates with filters ---------------------------------- #

    # ---------------------------------- Plot ---------------------------------- #

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)