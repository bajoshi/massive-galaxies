from __future__ import division

import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import griddata, interp1d
from scipy.integrate import simps

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
figs_dir = home + "/Desktop/FIGS/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

speed_of_light = 299792458e10  # angsroms per second

if __name__ == '__main__':
    main()
    sys.exit()