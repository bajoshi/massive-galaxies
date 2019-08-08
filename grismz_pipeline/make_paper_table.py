from __future__ import division

import numpy as np

import os
import sys
import glob

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
spz_results_dir = massive_figures_dir + 'spz_run_jan2019/'
zp_results_dir = massive_figures_dir + 'photoz_run_jan2019/'
zg_results_dir = massive_figures_dir + 'grismz_run_jan2019/'

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import spz_photoz_grismz_comparison as comp
from new_refine_grismz_gridsearch_parallel import get_data
import dn4000_catalog as dc

def main():

    # ------------------------------- Get catalog for final sample ------------------------------- #
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample.txt', dtype=None, names=True)

    spz_outdir = massive_figures_dir + 'cluster_results_covmat_3lsfsigma_June2019/'

    # Lists for storing results
    all_ids = []
    all_fields = []

    zs = []
    zp = []
    zg = []
    zspz = []

    zp_low_bound_chi2 = []
    zp_high_bound_chi2 = []
    zg_low_bound_chi2 = []
    zg_high_bound_chi2 = []
    zspz_low_bound_chi2 = []
    zspz_high_bound_chi2 = []

    # Loop over all result files and save to the empty lists defined above
    for fl in glob.glob(spz_outdir + 'redshift_fitting_results*.txt'):
        current_result = np.genfromtxt(fl, dtype=None, names=True, skip_header=1)

        all_ids.append(current_result['PearsID'])
        all_fields.append(current_result['Field'])

        zs.append(float(current_result['zspec']))
        zp.append(float(current_result['zp_minchi2']))
        zspz.append(float(current_result['zspz_minchi2']))
        zg.append(float(current_result['zg_minchi2']))

        zp_low_bound_chi2.append(float(current_result['zp_zerr_low']))
        zp_high_bound_chi2.append(float(current_result['zp_zerr_up']))
        zspz_low_bound_chi2.append(float(current_result['zspz_zerr_low']))
        zspz_high_bound_chi2.append(float(current_result['zspz_zerr_up']))
        zg_low_bound_chi2.append(float(current_result['zg_zerr_low']))
        zg_high_bound_chi2.append(float(current_result['zg_zerr_up']))

    # Read in master catalogs to get i-band mag
    # ------------------------------- Read PEARS cats ------------------------------- #
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    
    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['pearsdec'] = pears_ncat['pearsdec'] - dec_offset_goodsn_v19

    # Comment this print statement out if out don't want to actually print this list
    print_tex_format = False  # toggle this on/off for printing tex/ascii table # also change the mag cut below
    if print_tex_format:
        print "ID    Field     RA    DEC        zspec    zphot    zgrism     zspz    NetSig    D4000    D4000_err    iABmag"
    else:
        print "ID    Field     RA    DEC        zspec    zphot    low_zphot_err    high_zphot_err    zg    low_zgrism_err    high_zgrism_err",
        print "    zspz    low_zspz_err    high_zspz_err    NetSig    D4000    D4000_err    iABmag"

    total_galaxies = 0
    large_d4000_err_count = 0

    for i in range(len(all_ids)):
        current_id = all_ids[i]
        current_field = all_fields[i]

        sample_idx = np.where(final_sample['pearsid'] == current_id)[0]
        if not sample_idx.size:
            continue
        if len(sample_idx) == 1:
            if final_sample['field'][sample_idx] != current_field:
                continue

        # Check z_spec quality
        # Make sure to get the spec_z qual for the exact match
        if len(sample_idx) == 2:
            sample_idx_nested = int(np.where(final_sample['field'][sample_idx] == current_field)[0])
            current_specz_qual = final_sample['zspec_qual'][sample_idx[sample_idx_nested]]
        else:
            current_specz_qual = final_sample['zspec_qual'][int(sample_idx)]

        if current_specz_qual == '1' or current_specz_qual == 'C':
            #print "Skipping", current_field, current_id, "due to spec-z quality:", current_specz_qual
            continue

        # Get data
        grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = get_data(current_id, current_field)

        if return_code == 0:
            print current_id, current_field
            print "Return code should not have been 0. Exiting."
            sys.exit(0)

        # Get D4000 at specz
        current_specz = zs[i]

        lam_em = grism_lam_obs / (1 + current_specz)
        flam_em = grism_flam_obs * (1 + current_specz)
        ferr_em = grism_ferr_obs * (1 + current_specz)

        d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

        # Cut on large D4000 error
        if d4000_err > 0.5:
            large_d4000_err_count += 1
            continue

        # Get ra, dec, i_mag
        if current_field == 'GOODS-N':
            master_cat_idx = int(np.where(pears_ncat['id'] == current_id)[0])
            current_ra = pears_ncat['pearsra'][master_cat_idx]
            current_dec = pears_ncat['pearsdec'][master_cat_idx]
            current_imag = pears_ncat['imag'][master_cat_idx]
        elif current_field == 'GOODS-S':
            master_cat_idx = int(np.where(pears_scat['id'] == current_id)[0])
            current_ra = pears_scat['pearsra'][master_cat_idx]
            current_dec = pears_scat['pearsdec'][master_cat_idx]
            current_imag = pears_scat['imag'][master_cat_idx]

        # Cut on imag # ONLY for the table that goes into the paper!!
        #if current_imag > 23.5:
        #    continue

        # Get errors on redshifts   # from chi2 map NOT p(z) curve
        high_zperr = zp_high_bound_chi2[i] - zp[i]
        low_zperr  = zp[i] - zp_low_bound_chi2[i]

        high_zgerr = zg_high_bound_chi2[i] - zg[i]
        low_zgerr  = zg[i] - zg_low_bound_chi2[i]

        high_zspzerr = zspz_high_bound_chi2[i] - zspz[i]
        low_zspzerr  = zspz[i] - zspz_low_bound_chi2[i]

        if d4000 >= 1.1 and d4000 < 2.0:
            # Some formatting stuff just to make it easier to read on the screen and the tex file
            current_id_to_print = str(current_id)
            if len(current_id_to_print) == 5:
                current_id_to_print += ' '

            if current_dec < 0:
                current_dec_to_print = r"$-$" + str("{:.6f}".format(abs(current_dec)))
            else:
                current_dec_to_print = str("{:.6f}".format(current_dec))

            current_specz_to_print = str(current_specz)
            if len(current_specz_to_print) == 4:
                current_specz_to_print += '  '
            elif len(current_specz_to_print) == 5:
                current_specz_to_print += ' '

            current_netsig_to_print = str("{:.2f}".format(netsig_chosen))
            if len(current_netsig_to_print) == 5:
                current_netsig_to_print += ' '

            current_res_zphot = (zp[i] - current_specz) / (1 + current_specz)
            current_res_zspz = (zspz[i] - current_specz) / (1 + current_specz)

            current_res_zphot_to_print = str("{:.3f}".format(current_res_zphot))
            if current_res_zphot_to_print[0] != '-':
                current_res_zphot_to_print = '+' + current_res_zphot_to_print
            current_res_zspz_to_print = str("{:.3f}".format(current_res_zspz))
            if current_res_zspz_to_print[0] != '-':
                current_res_zspz_to_print = '+' + current_res_zspz_to_print

            if print_tex_format:
                print current_id_to_print, "  &",
                print current_field, "  &",
                print "{:.7f}".format(current_ra), "  &",
                print current_dec_to_print, "  &",
                print "{:.3f}".format(current_specz), "  &",
                print str("{:.2f}".format(zp[i])) + \
                "$\substack{+" + str("{:.2f}".format(high_zperr)) + " \\\\ -" + str("{:.2f}".format(low_zperr)) + "}$", "  &",
                print str("{:.2f}".format(zg[i])) + \
                "$\substack{+" + str("{:.2f}".format(high_zgerr)) + " \\\\ -" + str("{:.2f}".format(low_zgerr)) + "}$", "  &",
                print str("{:.2f}".format(zspz[i])) + \
                "$\substack{+" + str("{:.2f}".format(high_zspzerr)) + " \\\\ -" + str("{:.2f}".format(low_zspzerr)) + "}$", "  &",
                print current_netsig_to_print, "  &",
                print "{:.2f}".format(d4000), "  &",
                print "{:.2f}".format(d4000_err), "  &",
                #print current_res_zphot_to_print, "     ",
                #print current_res_zspz_to_print, "    ",
                print "{:.2f}".format(current_imag), "\\\\"

            else:  # print the ascii table which can just be copy pasted into a text file
                print current_id_to_print, "  ",
                print current_field, "  ",
                print "{:.7f}".format(current_ra), "  ",
                print "{:.6f}".format(current_dec), "  ",
                print "{:.3f}".format(current_specz), "  ",
                print "{:.2f}".format(zp[i]), "  ",
                print "{:.2f}".format(low_zperr), "  ",
                print "{:.2f}".format(high_zperr), "  ",
                print "{:.2f}".format(zg[i]), "  ",
                print "{:.2f}".format(low_zgerr), "  ",
                print "{:.2f}".format(high_zgerr), "  ",
                print "{:.2f}".format(zspz[i]), "  ",
                print "{:.2f}".format(low_zspzerr), "  ",
                print "{:.2f}".format(high_zspzerr), "  ",
                print current_netsig_to_print, "  ",
                print "{:.4f}".format(d4000), "  ",
                print "{:.2f}".format(d4000_err), "  ",
                #print current_res_zphot_to_print, "     ",
                #print current_res_zspz_to_print, "    ",
                print "{:.2f}".format(current_imag)

            #print current_id_to_print, current_field, current_specz, zp[i], zg[i], zspz[i], 
            #print d4000, d4000_err, current_netsig_to_print, current_imag

            total_galaxies += 1

    print "Total galaxies:", total_galaxies
    print "Galaxies with large D4000 errors (>0.5):", large_d4000_err_count

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)