from __future__ import division

import numpy as np
from astropy.io import fits

import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import grid_coadd as gd
import mocksim_results as mr

if __name__ == '__main__':
	"""
	Check D4000 vs stellar mass
	"""
	
	# Get D4000 at new weighted z
	


	sys.exit(0)