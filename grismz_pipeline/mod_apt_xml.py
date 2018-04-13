from __future__ import division

import numpy as np

import sys
import os

if __name__ == '__main__':
	
	# open file and get all lines in a list
	f = open(home + 'Desktop/FIGS/massive-galaxies/grismz_pipeline/10530_mod.apt', 'r')
	lines = f.readlines()
	f.close()

    # loop over lines and remove offending ones
    # create new file to write updated file to
    fn = open('10530_onlygrism_mod_test.txt', 'w')

    linecount = 0
    for line in lines[1129:]:
        if 'G800L' in line:
            for j in range(7,0,-1):
                fn.write(line[linecount-j] + '\n')
                break

    fn.close()

	sys.exit(0)