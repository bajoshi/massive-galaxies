from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys
import glob
import time
import datetime

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
import grid_coadd as gd
import dn4000_catalog as dc
import new_refine_grismz_gridsearch_parallel as ngp

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # ----------------------------------------- READ IN CATALOGS ----------------------------------------- #
    # read in matched files to get photo-z
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    # Read in Specz comparison catalogs
    specz_goodsn = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-N.txt', dtype=None, names=True)
    specz_goodss = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-S.txt', dtype=None, names=True)

    all_speccats = [specz_goodsn, specz_goodss]
    all_match_cats = [matched_cat_n, matched_cat_s]

    # save lists for comparing after code is done
    id_list = []
    field_list = []
    zgrism_list = []
    zspec_list = []
    zphot_list = []
    chi2_list = []
    netsig_list = []
    age_list = []
    tau_list = []
    av_list = []
    d4000_list = []
    d4000_err_list = []

    # start looping
    catcount = 0
    total_galaxies = 0
    skipped_galaxies = 0
    for cat in all_match_cats:

        for i in range(len(cat)):

            # --------------------------------------------- GET OBS DATA ------------------------------------------- #
            current_id = cat['pearsid'][i]

            if catcount == 0:
                current_field = 'GOODS-N'
                spec_cat = specz_goodsn
            elif catcount == 1: 
                current_field = 'GOODS-S'
                spec_cat = specz_goodss

            # Get specz if it exists as initial guess, otherwise get photoz
            specz_idx = np.where(spec_cat['pearsid'] == current_id)[0]

            if len(specz_idx) == 1:
                current_specz = float(spec_cat['specz'][specz_idx])
                redshift = float(cat['zphot'][i])
                starting_z = current_specz
            elif len(specz_idx) == 0:
                current_specz = -99.0
                redshift = float(cat['zphot'][i])
                starting_z = redshift
            else:
                print "Got other than 1 or 0 matches for the specz for ID", current_id, "in", current_field
                print "This much be fixed. Check why it happens. Exiting now."
                sys.exit(0)

            # Check that the starting redshfit is within the required range
            if (starting_z < 0.6) or (starting_z > 1.235):
                print "Current galaxy", current_id, current_field, "at starting_z", starting_z, "not within redshift range.",
                print "Moving to the next galaxy."
                skipped_galaxies += 1
                continue

            print "At ID", current_id, "in", current_field, "with specz and photo-z:", current_specz, redshift

            lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)

            if return_code == 0:
                print "Skipping due to an error with the obs data. See the error message just above this one.",
                print "Moving to the next galaxy."
                skipped_galaxies += 1
                continue

            lam_obs = lam_obs.astype(np.float64)
            flam_obs = flam_obs.astype(np.float64)
            ferr_obs = ferr_obs.astype(np.float64)

            # --------------------------------------------- Quality checks ------------------------------------------- #
            # Netsig check
            if netsig_chosen < 10:
                print "Skipping", current_id, "in", current_field, "due to low NetSig:", netsig_chosen
                skipped_galaxies += 1
                continue

            lam_em = lam_obs / (1 + starting_z)
            flam_em = flam_obs * (1 + starting_z)
            ferr_em = ferr_obs * (1 + starting_z)

            d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)
            if d4000 < 1.1:
                print "Skipping", current_id, "in", current_field, "due to low D4000:", d4000
                skipped_galaxies += 1
                continue

            total_galaxies += 1
            d4000_list.append(d4000)

    d4000_list = np.asarray(d4000_list)

    print "Started with", len(matched_cat_n), "+", len(matched_cat_s), "galaxies."
    print "D4000 low limit was 1.1."
    print "Code can go through", total_galaxies, "galaxies. Some of these will still be skipped due to NO LSF."
    print "Skipped", skipped_galaxies, "galaxies."

    np.save(figs_dir + 'massive-galaxies-figures/full_run/d4000_list_test.npy', d4000_list)

    # Total time taken
    print "Total time taken --", str("{:.2f}".format(time.time() - start))
    sys.exit(0)

