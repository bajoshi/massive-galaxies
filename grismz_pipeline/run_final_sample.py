from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.integrate import simps

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"
threedhst_datadir = home + "/Desktop/3dhst_data/"
massive_figures_dir = figs_dir + 'massive-galaxies-figures/'
savedir = massive_figures_dir + 'single_galaxy_comparison/'  # Required to save p(z) curve

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import refine_redshifts_dn4000 as old_ref
import fullfitting_grism_broadband_emlines as ff
import photoz
import new_refine_grismz_gridsearch_parallel as ngp
import model_mods as mm
import dn4000_catalog as dc
import mocksim_results as mr

speed_of_light = 299792458e10  # angstroms per second

def main():
	
	return None

if __name__ == '__main__':
	main()
	sys.exit()