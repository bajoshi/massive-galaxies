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
    all_zwt_1xerr = []
    all_zwt_2xgrismerr = []
    all_zwt_2xphoterr = []

    # loop over all galaxies that the code has run through
    fl_count = 0
    for fl in sorted(glob.glob(massive_figures_dir + 'large_diff_specz_sample/run_with_1xerrors/*_pz.npy')):
        # the sorted() above just makes sure that I get galaxies in the right order

        # read in pz and zarr files
        pz = np.load(fl)
        z_arr = np.load(fl.replace('_pz', '_z_arr'))

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

        if (current_id == 125963) and (current_field == 'GOODS-N'):  # got nan for this galaxy for some reason # skipping
            continue

        specz_idx = np.where(spec_cat['pearsid'] == current_id)[0]
        photoz_idx = np.where(cat['pearsid'] == current_id)[0]
        current_specz = float(spec_cat['specz'][specz_idx])
        current_photoz = float(cat['zphot'][photoz_idx])
        current_specz_qual = spec_cat['specz_qual'][specz_idx]

        if (current_specz_qual == 'D') or (current_specz_qual == '2'):
            continue

        all_pz.append(pz)
        all_zarr.append(z_arr)
        all_specz.append(current_specz)
        all_photoz.append(current_photoz)

        z_wt = np.sum(z_arr * pz)
        all_zwt_1xerr.append(z_wt)
        
        # Get weighted redshifts for other runs
        # define paths
        grismerr_2x_fl_path = fl.replace('large_diff_specz_sample/run_with_1xerrors', 'large_diff_specz_sample/run_with_2xgrismerrors')
        photerr_2x_fl_path = fl.replace('large_diff_specz_sample/run_with_1xerrors', 'large_diff_specz_sample/run_with_2xphoterrors')

        # ------- grism errors changed -------- #
        pz_2xgrismerr = np.load(grismerr_2x_fl_path)
        z_arr_2xgrismerr = np.load(grismerr_2x_fl_path.replace('_pz', '_z_arr'))

        z_wt_2xgrismerr = np.sum(z_arr_2xgrismerr * pz_2xgrismerr)
        all_zwt_2xgrismerr.append(z_wt_2xgrismerr)

        # ------- photometry errors changed -------- #
        pz_2xphoterr = np.load(photerr_2x_fl_path)
        z_arr_2xphoterr = np.load(photerr_2x_fl_path.replace('_pz', '_z_arr'))

        z_wt_2xphoterr = np.sum(z_arr_2xphoterr * pz_2xphoterr)
        all_zwt_2xphoterr.append(z_wt_2xphoterr)

        # Dont remove this print line. Useful for debugging.
        print current_id, current_field, current_specz, current_photoz, "{:.4}".format(z_wt), "{:.4}".format(z_wt_2xgrismerr), "{:.4}".format(z_wt_2xphoterr)

        fl_count += 1

    print "Total galaxies:", fl_count

    # Convert to numpy arrays
    all_pz = np.asarray(all_pz)
    all_specz = np.asarray(all_specz)
    all_photoz = np.asarray(all_photoz)
    all_zarr = np.asarray(all_zarr)
    all_zwt_1xerr = np.asarray(all_zwt_1xerr)
    all_zwt_2xgrismerr = np.asarray(all_zwt_2xgrismerr)
    all_zwt_2xphoterr = np.asarray(all_zwt_2xphoterr)

    # Get residuals
    resid_photoz = (all_specz - all_photoz) / (1+all_specz)
    resid_zweight_1xerr = (all_specz - all_zwt_1xerr) / (1+all_specz)
    resid_zweight_2xgrismerr = (all_specz - all_zwt_2xgrismerr) / (1+all_specz)
    resid_zweight_2xphoterr = (all_specz - all_zwt_2xphoterr) / (1+all_specz)

    # Get sigma_NMAD
    sigma_nmad_photo = 1.48 * np.median(abs(((all_specz - all_photoz) - np.median((all_specz - all_photoz))) / (1 + all_specz)))
    sigma_nmad_zwt_1xerr = 1.48 * np.median(abs(((all_specz - all_zwt_1xerr) - np.median((all_specz - all_zwt_1xerr))) / (1 + all_specz)))
    sigma_nmad_zwt_2xgrismerr = 1.48 * np.median(abs(((all_specz - all_zwt_2xgrismerr) - np.median((all_specz - all_zwt_2xgrismerr))) / (1 + all_specz)))
    sigma_nmad_zwt_2xphoterr = 1.48 * np.median(abs(((all_specz - all_zwt_2xphoterr) - np.median((all_specz - all_zwt_2xphoterr))) / (1 + all_specz)))

    print "Max residual photo-z:", "{:.3}".format(max(resid_photoz))
    print "Max residual weighted z (1xerr):", "{:.3}".format(max(resid_zweight_1xerr))
    print "Max residual weighted z (2xgrismerr):", "{:.3}".format(max(resid_zweight_2xgrismerr))
    print "Max residual weighted z (2xphoterr):", "{:.3}".format(max(resid_zweight_2xphoterr))

    print "Mean, std. dev., and sigma_NMAD for residual photo-z:", "{:.4}".format(np.mean(resid_photoz)), "{:.4}".format(np.std(resid_photoz)), "{:.4}".format(sigma_nmad_photo)
    print "Mean, std. dev., and sigma_NMAD for residual weighted z (1xerr):", "{:.4}".format(np.mean(resid_zweight_1xerr)), "{:.4}".format(np.std(resid_zweight_1xerr)), "{:.4}".format(sigma_nmad_zwt_1xerr)
    print "Mean, std. dev., and sigma_NMAD for residual weighted z (2xgrismerr):", "{:.4}".format(np.mean(resid_zweight_2xgrismerr)), "{:.4}".format(np.std(resid_zweight_2xgrismerr)), "{:.4}".format(sigma_nmad_zwt_2xgrismerr)
    print "Mean, std. dev., and sigma_NMAD for residual weighted z (2xphoterr):", "{:.4}".format(np.mean(resid_zweight_2xphoterr)), "{:.4}".format(np.std(resid_zweight_2xphoterr)), "{:.4}".format(sigma_nmad_zwt_2xphoterr)

    # Plot 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(resid_photoz, 50, color='r', histtype='step', lw=2, range=(-0.12, 0.12))
    ax.hist(resid_zweight_1xerr, 50, color='b', histtype='step', lw=2, range=(-0.12, 0.12))
    #ax.hist(resid_zweight_2xgrismerr, 50, color='g', histtype='step', lw=2, range=(-0.12, 0.12))
    #ax.hist(resid_zweight_2xphoterr, 50, color='pink', histtype='step', lw=2, range=(-0.12, 0.12))

    ax.axvline(x=0.0, ls='--', color='k')

    plt.show()

    sys.exit(0)
