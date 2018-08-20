from __future__ import division

import numpy as np
import numpy.ma as ma
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo
from joblib import Parallel, delayed

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"
threedhst_datadir = home + "/Desktop/3dhst_data/"

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import grid_coadd as gd
import refine_redshifts_dn4000 as old_ref
import model_mods_cython_copytoedit as model_mods_cython
import dn4000_catalog as dc
import new_refine_grismz_gridsearch_parallel as ngp

def check_spec_plot(grism_spec, grism_lam_obs, grism_ferr_obs):

    return None

def get_filt_zp(filtname):

    filtname_arr = np.array(['F125W', 'F140W', 'F160W'])  # 'F435W', 'F606W', 'F775W', 'F850LP', 

    # Corresponding lists containing AB and ST zeropoints
    # The first 4 filters i.e. 'F435W', 'F606W', 'F775W', 'F850LP'
    # are ACS/WFC filters and the ZP calculation is yet to be done for these.
    # ACS zeropoints are time-dependent and therefore the zeropoint calculator has to be used.
    # Check: http://www.stsci.edu/hst/acs/analysis/zeropoints
    # For WFC3/IR: http://www.stsci.edu/hst/wfc3/analysis/ir_phot_zpt
    filt_zp_st_list = [28.0203, 28.479, 28.1875]
    filt_zp_ab_list = [26.2303, 26.4524, 25.9463]

    filt_idx = np.where(filtname_arr == filtname)[0]

    filt_zp_st = filt_zp_st_list[filt_idx]
    filt_zp_ab = filt_zp_ab_list[filt_idx]

    return filt_zp_st, filt_zp_ab

def get_flam(filtname, cat_flux):
    """ Convert everything to f_lambda units """

    filt_zp_st, filt_zp_ab = get_filt_zp(filtname)

    abmag = 25.0 - 2.5*np.log10(cat_flux)
    stmag = filt_zp_st - filt_zp_ab + abmag

    flam = 10**(-1 * (stmag + 21.1) / 2.5)

    return stmag

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Read in photometry and grism+photometry catalogs
    # GOODS-N from 3DHST
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshfits, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55), skip_header=3)

    threed_ra = goodsn_phot_cat_3dhst['ra']
    threed_dec = goodsn_phot_cat_3dhst['dec']

    # Read in grism data
    current_id = 106613
    current_field = 'GOODS-N'
    lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)

    # match and get photometry data
    # read in matched files for grism spectra info
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    # find grism obj ra,dec
    cat_idx = np.where(matched_cat_n['pearsid'] == current_id)[0]
    if cat_idx.size:
        zphot = float(matched_cat_n['zphot'][cat_idx])
        current_ra = float(matched_cat_n['pearsra'][cat_idx])
        current_dec = float(matched_cat_n['pearsdec'][cat_idx])

    ra_lim = 0.5/3600  # arcseconds in degrees
    dec_lim = 0.5/3600
    threed_phot_idx = np.where((threed_ra >= current_ra - ra_lim) & (threed_ra <= current_ra + ra_lim) & \
        (threed_dec >= current_dec - dec_lim) & (threed_dec <= current_dec + dec_lim))[0]

    print get_flam('F125W', goodsn_phot_cat_3dhst['f_F125W'][threed_phot_idx])
    print get_flam('F140W', goodsn_phot_cat_3dhst['f_F140W'][threed_phot_idx])
    print get_flam('F160W', goodsn_phot_cat_3dhst['f_F160W'][threed_phot_idx])

    # Plot to check
    #check_spec_plot(spec, lam_obs, ferr_obs, )

    sys.exit(0)