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

def get_contam(pears_index, field):

    # read in spectrum file
    data_path = home + "/Documents/PEARS/data_spectra_only/"
    # Get the correct filename and the number of extensions
    if field == 'GOODS-N':
        filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    elif field == 'GOODS-S':
        filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

    fitsfile = fits.open(filename)
    n_ext = fitsfile[0].header['NEXTEND']

    # Loop over all extensions and get the best PA
    # Get highest netsig to find the spectrum to be added
    if n_ext > 1:
        netsiglist = []
        palist = []
        for count in range(n_ext):
            #print "At PA", fitsfile[count+1].header['POSANG']  # Line useful for debugging. Do not remove. Just uncomment.
            fitsdata = fitsfile[count+1].data
            netsig = gd.get_net_sig(fitsdata)
            netsiglist.append(netsig)
            palist.append(fitsfile[count+1].header['POSANG'])
            #print "At PA", fitsfile[count+1].header['POSANG'], "with NetSig", netsig  
            # Above line also useful for debugging. Do not remove. Just uncomment.
        netsiglist = np.array(netsiglist)
        maxnetsigarg = np.argmax(netsiglist)
        netsig_chosen = np.max(netsiglist)
        spec_toadd = fitsfile[maxnetsigarg+1].data
        pa_chosen = fitsfile[maxnetsigarg+1].header['POSANG']
    elif n_ext == 1:
        spec_toadd = fitsfile[1].data
        pa_chosen = fitsfile[1].header['POSANG']
        netsig_chosen = gd.get_net_sig(fitsfile[1].data)
        
    # Now get the spectrum to be added
    lam_obs = spec_toadd['LAMBDA']
    flam_obs = spec_toadd['FLUX']
    ferr_obs = spec_toadd['FERROR']
    contam = spec_toadd['CONTAM']

    return lam_obs, flam_obs, ferr_obs, contam

if __name__ == '__main__':

    # Read in Specz comparison catalogs
    specz_goodsn = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-N.txt', dtype=None, names=True)
    specz_goodss = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-S.txt', dtype=None, names=True)

    # read in matched files to get photo-z
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    # Read in results arrays
    id_arr = np.load(massive_figures_dir + 'large_diff_specz_sample/run_with_1xerrors/withemlines_id_list_gn.npy')
    field_arr = np.load(massive_figures_dir + 'large_diff_specz_sample/run_with_1xerrors/withemlines_field_list_gn.npy')
    d4000_arr = np.load(massive_figures_dir + 'large_diff_specz_sample/run_with_1xerrors/withemlines_d4000_list_gn.npy')

    # create empty lists to store final comparison arrays
    all_pz = []
    all_specz = []
    all_photoz = []
    all_zarr = []
    all_zwt_1xerr = []
    all_zwt_2xgrismerr = []
    all_zwt_2xphoterr = []
    all_d4000 = []
    all_contam_frac = []

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

        # Read in D4000
        results_idx = np.where((id_arr == current_id) & (field_arr == current_field))[0]
        current_d4000 = d4000_arr[results_idx]

        all_d4000.append(current_d4000)

        # Get contamination
        lam_obs, flam_obs, ferr_obs, contam = get_contam(current_id, current_field)

        all_contam_frac.append(np.nansum(contam) / np.nansum(flam_obs))

        # -------------------------------------
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
        #print current_id, current_d4000, current_field, current_specz, current_photoz, "{:.4}".format(z_wt), "{:.4}".format(z_wt_2xgrismerr), "{:.4}".format(z_wt_2xphoterr)

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
    all_d4000 = np.asarray(all_d4000)
    all_contam_frac = np.asarray(all_contam_frac)

    # -------------------------------------------- Quantify results -------------------------------------------- #
    # Cut on D4000
    d4000_low = 1.1
    d4000_high = 1.6
    d4000_idx = np.where((all_d4000 >= d4000_low) & (all_d4000 < d4000_high))[0]
    print "Galaxies in D4000 range:", len(d4000_idx)

    # apply cuts
    all_pz = all_pz[d4000_idx]
    all_specz = all_specz[d4000_idx]
    all_photoz = all_photoz[d4000_idx]
    all_zarr = all_zarr[d4000_idx]
    all_zwt_1xerr = all_zwt_1xerr[d4000_idx]
    all_zwt_2xgrismerr = all_zwt_2xgrismerr[d4000_idx]
    all_zwt_2xphoterr = all_zwt_2xphoterr[d4000_idx]
    all_d4000 = all_d4000[d4000_idx]
    all_contam_frac = all_contam_frac[d4000_idx]

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

    # -------------------------------------------- Plotting -------------------------------------------- #
    # ---------------------- Residual histogram ---------------------- #
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'Contamination Fraction')
    ax.set_ylabel(r'Abs(Residuals [SPZ])')

    ax.scatter(all_contam_frac, abs(resid_zweight_1xerr), s=5)

    ax.set_xlim(min(all_contam_frac), max(all_contam_frac))

    ax.axhline(y=0.0, color='k', ls='--')

    fig.savefig(massive_figures_dir + 'large_diff_specz_sample/resid_contam.png', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    # ---------------------- Residual histogram ---------------------- #
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(resid_photoz, 20, color='r', histtype='step', lw=2, label='Photo-z residuals', range=(-0.1, 0.1))
    ax.hist(resid_zweight_1xerr, 20, color='b', histtype='step', lw=2, label='SPZ residuals', range=(-0.1, 0.1))
    #ax.hist(resid_zweight_2xgrismerr, color='g', histtype='step', lw=2)#, range=(-0.12, 0.12))
    #ax.hist(resid_zweight_2xphoterr, color='orange', histtype='step', lw=2)#, range=(-0.12, 0.12))

    ax.axvline(x=0.0, ls='--', color='k')

    ax.set_xlabel(r'$\mathrm{Residuals}$')
    ax.set_ylabel(r'$\mathrm{N}$')

    ax.minorticks_on()
    ax.legend(loc='upper left')

    # Info text
    ax.text(0.05, 0.85, r'$' + str(d4000_low) + '\leq \mathrm{D4000} < ' + str(d4000_high) + '$', \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=14)

    ax.text(0.05, 0.8, r'$\mathrm{N = }$' + str(len(d4000_idx)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=14)


    ax.text(0.05, 0.71, r'$\mu_{\mathrm{photo-z}} = $' + mr.convert_to_sci_not(np.mean(resid_photoz)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=10)
    ax.text(0.05, 0.66, r'$\mathrm{\sigma^{NMAD}_{photo-z}} = $' + mr.convert_to_sci_not(sigma_nmad_photo), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=10)

    ax.text(0.05, 0.58, r'$\mu_{\mathrm{SPZ}} = $' + mr.convert_to_sci_not(np.mean(resid_zweight_1xerr)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=10)
    ax.text(0.05, 0.53, r'$\mathrm{\sigma^{NMAD}_{SPZ}} = $' + mr.convert_to_sci_not(sigma_nmad_zwt_1xerr), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=10)

    # Save figure
    fig.savefig(massive_figures_dir + \
        'large_diff_specz_sample/resid_hist_' + 'd4000_' + str(d4000_low) + 'to' + str(d4000_high) + '.png', \
        dpi=300, bbox_inches='tight')

    #plt.show()

    sys.exit(0)
