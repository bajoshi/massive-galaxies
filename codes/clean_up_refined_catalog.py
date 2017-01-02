from __future__ import division

import numpy as np

import os
import sys

if __name__ == '__main__':
	
    # read catalog adn rename arrays for convenience
    pears_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-N.txt',\
     dtype=None, names=True, skip_header=1)
    pears_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-S.txt',\
     dtype=None, names=True, skip_header=1)

    