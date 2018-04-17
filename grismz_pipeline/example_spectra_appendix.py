from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(massive_galaxies_dir + 'grism_pipeline/')
sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import mag_hist as mh
import model_mods_cython_copytoedit as model_mods_cython
import new_refine_grismz_gridsearch_parallel as ngp
import dn4000_catalog as dc
import refine_redshifts_dn4000 as old_ref

if __name__ == '__main__':
	
	

	sys.exit(0)