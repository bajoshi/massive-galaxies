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
    all_zwt_new = []  # the new redshifts have their grism errors artificially increased by 2x
    all_zwt_old = []  # the old ones did not have any changes made to either the grism or the photometry errors

    check_old = False

    # loop over all galaxies that the code has run through
    fl_count = 0
    for fl in sorted(glob.glob(massive_figures_dir + 'large_diff_specz_sample/*_pz.npy')):
        # the sorted() above just makes sure that I get galaxies in the right order

        # read in pz and zarr files
        pz = np.load(fl)
        z_arr = np.load(fl.replace('_pz', '_z_arr'))

        # Get other info
        fl_name = os.path.basename(fl)
        split_arr = fl_name.split('.')[0].split('_')

        current_field = split_arr[0]
        current_id = int(split_arr[1])

        if check_old:
            # The galaxies that didn't finsih in the old run
            old_fl_path = fl.replace('large_diff_specz_sample', 'large_diff_specz_sample/run_with_1xphoterrors')
            if not os.path.isfile(old_fl_path):
                continue

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

        all_pz.append(pz)
        all_zarr.append(z_arr)
        all_specz.append(current_specz)
        all_photoz.append(current_photoz)

        z_wt = np.sum(z_arr * pz)
        all_zwt_new.append(z_wt)
        
        if check_old:
            # Get old z_wt
            # read in pz and zarr files
            old_fl_path = fl.replace('large_diff_specz_sample', 'large_diff_specz_sample/run_with_1xphoterrors')
            pz = np.load(old_fl_path)
            z_arr = np.load(old_fl_path.replace('_pz', '_z_arr'))

            z_wt_old = np.sum(z_arr * pz)
            all_zwt_old.append(z_wt_old)

            # Dont remove this print line. Useful for debugging.
            # print current_id, current_field, current_specz, current_photoz, "{:.4}".format(z_wt_old), "{:.4}".format(z_wt)

        fl_count += 1

    print "Total galaxies:", fl_count

    # Convert to numpy arrays
    all_pz = np.asarray(all_pz)
    all_specz = np.asarray(all_specz)
    all_photoz = np.asarray(all_photoz)
    all_zarr = np.asarray(all_zarr)
    all_zwt_new = np.asarray(all_zwt_new)

    if check_old:
        all_zwt_old = np.asarray(all_zwt_old)

    # Get residuals
    resid_photoz = (all_specz - all_photoz) / (1+all_specz)
    resid_zweight_new = (all_specz - all_zwt_new) / (1+all_specz)

    if check_old:
        resid_zweight_old = (all_specz - all_zwt_old) / (1+all_specz)
        sigma_nmad_zwt_old = 1.48 * np.median(abs(((all_specz - all_zwt_old) - np.median((all_specz - all_zwt_old))) / (1 + all_specz)))

    sigma_nmad_zwt_new = 1.48 * np.median(abs(((all_specz - all_zwt_new) - np.median((all_specz - all_zwt_new))) / (1 + all_specz)))
    sigma_nmad_photo = 1.48 * np.median(abs(((all_specz - all_photoz) - np.median((all_specz - all_photoz))) / (1 + all_specz)))

    print "Max residual photo-z:", "{:.3}".format(max(resid_photoz))
    print "Max residual weighted z (new):", "{:.3}".format(max(resid_zweight_new))
    if check_old:
        print "Max residual weighted z (old):", "{:.3}".format(max(resid_zweight_old))

    print "Mean, std. dev., and sigma_NMAD for residual photo-z:", "{:.4}".format(np.mean(resid_photoz)), "{:.4}".format(np.std(resid_photoz)), "{:.4}".format(sigma_nmad_photo)
    print "Mean, std. dev., and sigma_NMAD for residual weighted z (new):", "{:.4}".format(np.mean(resid_zweight_new)), "{:.4}".format(np.std(resid_zweight_new)), "{:.4}".format(sigma_nmad_zwt_new)
    if check_old:
        print "Mean, std. dev., and sigma_NMAD for residual weighted z (old):", "{:.4}".format(np.mean(resid_zweight_old)), "{:.4}".format(np.std(resid_zweight_old)), "{:.4}".format( sigma_nmad_zwt_old)

    # Plot 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(resid_photoz, 50, color='r', histtype='step', lw=2)#, range=(-0.15, 0.15))
    ax.hist(resid_zweight_new, 50, color='g', histtype='step', lw=2)#, range=(-0.15, 0.15))

    if check_old:
        ax.hist(resid_zweight_old, color='b', histtype='step', lw=2)#, range=(-0.1, 0.1))

    ax.axvline(x=0.0, ls='--', color='k')

    plt.show()

    sys.exit(0)
