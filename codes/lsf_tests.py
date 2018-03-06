from __future__ import division

import numpy as np
from scipy.signal import fftconvolve

import os
import sys

home = os.getenv('HOME')

if __name__ == '__main__':
    
    galaxy_id = 78415
    field = 'GOODS-S'
    num_pa = 8

    # reading in avg lsf for now
    if field == 'GOODS-S':
        lsf = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_lsfs/south_lsfs/s' + str(galaxy_id) + '_avg_lsf.txt')
    elif field == 'GOODS-N':
        lsf = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_lsfs/north_lsfs/n' + str(galaxy_id) + '_avg_lsf.txt')

    # divide by total number of PA because avg LSF is not actually averaged
    lsf /= num_pa

    # do checks
    print "Total LSF length:", len(lsf)
    print "Maxima in LSF occurs at:", np.argmax(lsf)
    print "Total non-zero elements in LSF:", len(np.nonzero(lsf)[0])
    print "Non-zero element indices in LSF:", np.nonzero(lsf)

    # read in model spectra

    sys.exit(0)