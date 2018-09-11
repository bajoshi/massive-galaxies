from __future__ import division

import numpy as np

import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"

if __name__ == '__main__':

    # Read in Specz comparison catalogs
    specz_goodsn = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-N.txt', dtype=None, names=True)
    specz_goodss = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-S.txt', dtype=None, names=True)

    # read in matched files to get photo-z
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    # create empty lists to store final comparison arrays
    all_pz = []
    all_specz = []
    all_photoz = []
    all_zarr = []
    all_zwt = []

    # loop over all galaxies that the code has run through
    fl_count = 0
    for fl in sorted(glob.glob(massive_figures_dir + 'large_diff_specz_sample/*_pz.npy')):
        # the sorted() above just makes sure that I get galaxies in the right order

        # read in pz and zarr files
        pz = np.load(fl)
        z_arr = np.load(fl.replace('_pz', '_z_arr'))

        all_pz.append(pz)
        all_zarr.append(z_arr)

        # Get other info
        fl_name = os.path.basename(fl)
        split_arr = fl_name.split('.')[0].split('_')

        current_field = split_arr[0]
        current_id = int(split_arr[1])

        # Find specz and photoz by matching
        if current_field == 'GOODS-N':
            spec_cat = specz_goodsn
            cat = matched_cat_n

        elif current_field == 'GOODS-S':
            spec_cat = specz_goodss
            cat = matched_cat_s

        specz_idx = np.where(spec_cat['pearsid'] == current_id)[0]
        photoz_idx = np.where(cat['pearsid'] == current_id)[0]
        current_specz = float(spec_cat['specz'][specz_idx])
        current_photoz = float(cat['zphot'][photoz_idx])

        all_specz.append(current_specz)
        all_photoz.append(current_photoz)

        z_wt = np.sum(z_arr * pz)
        all_zwt.append(z_wt)

        fl_count += 1

    print "Total galaxies:", fl_count

    # Convert to numpy arrays
    all_pz = np.asarray(all_pz)
    all_specz = np.asarray(all_specz)
    all_photoz = np.asarray(all_photoz)
    all_zarr = np.asarray(all_zarr)
    all_zwt = np.asarray(all_zwt)

    # Get residuals
    resid_photoz = (all_specz - all_photoz) / (1+all_specz)
    resid_zweight = (all_specz - all_zwt) / (1+all_specz)

    # Plot 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(resid_photoz, 20, color='r', histtype='step', range=(-0.1, 0.1))
    ax.hist(resid_zweight, 20, color='g', histtype='step', range=(-0.1, 0.1))

    ax.axvline(x=0.0, ls='--', color='k')

    plt.show()

    sys.exit(0)
