from __future__ import division

import numpy as np

import os
import sys

if __name__ == '__main__':

    # ------------------------------- Read cats ------------------------------- #
    n_cols_req = np.concatenate(([0], np.arange(6,27), np.arange(32,53)))
    ncat = np.genfromtxt('ugrism_n.cat', dtype=None, usecols=n_cols_req)

    s_cols_req = np.concatenate(([0], np.arange(6,27), np.arange(35,55)))
    scat = np.genfromtxt('ugrism_s.cat', dtype=None, usecols=s_cols_req)

    # ------------------------------- Resave ------------------------------- #
    allcats = [ncat, scat]
    fields = ['N', 'S']
    count = 0
    for cat in allcats:

        current_field = fields[count]

        # Get header
        """
        hdr = str(cat.dtype.names)
        # Remove unnecessary characters
        hdr = hdr.replace(',', ' ')
        hdr = hdr.replace('\'', '')
        hdr = hdr.replace('(', '')
        hdr = hdr.replace(')', '')
        """

        # Now rewrite file line by line
        with open('goods_' + current_field + '_eazy_grism_only.txt', 'w') as fh:

            #fh.write('#' + hdr + '\n')

            for i in range(len(cat)):
                string_to_write = str(cat[i])
                # Remove unnecessary characters again
                string_to_write = string_to_write.replace(',', ' ')
                string_to_write = string_to_write.replace('(', '')
                string_to_write = string_to_write.replace(')', '')

                fh.write(string_to_write + '\n')

        count += 1

    sys.exit(0)