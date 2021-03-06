"""
The purpose of this code is to get the final sample for the SPZ paper.
It will match the PEARS master catalog with the 3DHST photometry catalog
and the ground-based spectroscopic redshift catalog from Nimish.
"""
from __future__ import division

import numpy as np

import os
import sys

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import dn4000_catalog as dc
import new_refine_grismz_gridsearch_parallel as ngp

def get_sample_in_zrange():

    # Get correct directories 
    figs_data_dir = '/Volumes/Bhavins_backup/bc03_models_npy_spectra/'
    threedhst_datadir = "/Volumes/Bhavins_backup/3dhst_data/"
    cspout = "/Volumes/Bhavins_backup/bc03_models_npy_spectra/cspout_2016updated_galaxev/"
    # This is if working on the laptop. 
    # Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(figs_data_dir):
        import pysynphot  # only import pysynphot on firstlight becasue that's the only place where I installed it.
        figs_data_dir = figs_dir  # this path only exists on firstlight
        threedhst_datadir = home + "/Desktop/3dhst_data/"  # this path only exists on firstlight
        cspout = home + '/Documents/galaxev_bc03_2016update/bc03/src/cspout_2016updated_galaxev/'
        if not os.path.isdir(figs_data_dir):
            print "Model specta and data files not found. Exiting..."
            sys.exit(0)

    # ------------------------------- Read PEARS cats ------------------------------- #
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'pearsra', 'pearsdec', 'imag'], usecols=(0,1,2,3))
    
    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['pearsdec'] = pears_ncat['pearsdec'] - dec_offset_goodsn_v19

    # ------------------------------- Read in 3DHST photomtery catalog ------------------------------- #
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
    'f_U', 'e_U', 'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
    'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), skip_header=3)

    # ------------------------------- Read in ground-based spectroscopic redshift compilation ------------------------------- #
    # read Nimish's specz catalogs
    goods_n_specz_cat = np.genfromtxt(massive_galaxies_dir + 'goods_n_specz_0117.txt', \
        dtype=None, names=['ra','dec','z_spec','z_qual','catname','duplicate'], skip_header=13)
    goods_s_specz_cat = np.genfromtxt(massive_galaxies_dir + 'cdfs_specz_0117.txt', \
        dtype=None, names=['ra','dec','z_spec','z_qual','catname','duplicate'], skip_header=13)

    # ------------------------------- Now match the three ------------------------------- #
    # Preliminary prep before matching
    all_pears_cats = [pears_ncat, pears_scat]

    # Lists for saving to file
    pears_id = []
    pears_field = []
    pears_ra = []
    pears_dec = []
    specz = []
    specz_source = []
    specz_qual = []
    imag = []

    catcount = 0
    match_count = 0
    for cat in all_pears_cats:

        if catcount == 0:
            cat = pears_ncat
            phot_cat_3dhst = goodsn_phot_cat_3dhst
            spec_cat = goods_n_specz_cat
            current_field = 'GOODS-N'
        if catcount == 1:
            cat = pears_scat
            phot_cat_3dhst = goodss_phot_cat_3dhst
            spec_cat = goods_s_specz_cat
            current_field = 'GOODS-S'

        threed_ra = phot_cat_3dhst['ra']
        threed_dec = phot_cat_3dhst['dec']
        spec_ra = spec_cat['ra']
        spec_dec = spec_cat['dec']

        for i in range(len(cat)):

            current_id = cat['id'][i]

            # find grism obj ra,dec
            current_ra = float(cat['pearsra'][i])
            current_dec = float(cat['pearsdec'][i])

            # Matching radius
            ra_lim = 0.3/3600  # arcseconds in degrees
            dec_lim = 0.3/3600

            # ------------------------- Spec match ------------------------- #
            # First with spectroscopic catalog because that way you
            # can skip doing the photometry matching for most of them.
            # Since the biggest bottleneck for the final sample will be
            # the number of ground-based spectroscopic redshifts
            spec_idx = np.where((spec_ra >= current_ra - ra_lim) & (spec_ra <= current_ra + ra_lim) & \
                (spec_dec >= current_dec - dec_lim) & (spec_dec <= current_dec + dec_lim))[0]

            """
            If there are multiple matches within 0.5 arseconds then choose the closest one.
            """
            if len(spec_idx) > 1:
                print "Found multiple matches in ground-based spectroscopic catalog. Picking closest one."

                ra_two = current_ra
                dec_two = current_dec

                dist_list = []
                for v in range(len(spec_idx)):

                    ra_one = spec_ra[spec_idx][v]
                    dec_one = spec_dec[spec_idx][v]

                    dist = np.arccos(np.cos(dec_one*np.pi/180) * np.cos(dec_two*np.pi/180) * np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + 
                        np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
                    dist_list.append(dist)

                dist_list = np.asarray(dist_list)
                dist_idx = np.argmin(dist_list)
                spec_idx = spec_idx[dist_idx]

            elif len(spec_idx) == 0:
                print "Match not found in ground-based spectroscopic catalog. Skipping."
                continue

            # ------------------------- Now match with photometry ------------------------- #
            # Uses exact same procedure as above
            threed_phot_idx = np.where((threed_ra >= current_ra - ra_lim) & (threed_ra <= current_ra + ra_lim) & \
                (threed_dec >= current_dec - dec_lim) & (threed_dec <= current_dec + dec_lim))[0]

            """
            If there are multiple matches within 0.5 arseconds then choose the closest one.
            """
            if len(threed_phot_idx) > 1:
                print "Found multiple matches in photometry catalog. Picking closest one."

                ra_two = current_ra
                dec_two = current_dec

                dist_list = []
                for v in range(len(threed_phot_idx)):

                    ra_one = threed_ra[threed_phot_idx][v]
                    dec_one = threed_dec[threed_phot_idx][v]

                    dist = np.arccos(np.cos(dec_one*np.pi/180) * np.cos(dec_two*np.pi/180) * np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + 
                        np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
                    dist_list.append(dist)

                dist_list = np.asarray(dist_list)
                dist_idx = np.argmin(dist_list)
                threed_phot_idx = threed_phot_idx[dist_idx]

            elif len(threed_phot_idx) == 0:
                print "Match not found in Photmetry catalog. Skipping."
                continue

            # Check source of specz
            # Since you only want confirmed GROUND-BASED spectroscopic redshifts
            # You should be skipping any other redshifts from low-res grism data
            # There are some others that have G141 based redshifts but go by a 
            # different name in the specz catalog. I'm chucking those too.
            reject_source_list = ['3D_HST', 'Morris_2015', 'PEARS_ETG', 'Pirzkal_2013', \
            'GRAPES_HUDF', 'ePEARS_CDFS', 'ePEARS_HUDF', 'GRISM_HUDF']

            current_specz_source = spec_cat['catname'][spec_idx]
            if type(current_specz_source) is np.ndarray:
                current_specz_source = current_specz_source[0]

            if current_specz_source in reject_source_list:
                print "Not an emission line based redshift from the ground. Skipping."
                continue

            current_specz_qual = spec_cat['z_qual'][spec_idx]
            if type(current_specz_qual) is np.ndarray:
                current_specz_qual = current_specz_qual[0]

            # Get current i-band magnitude
            current_imag = float(cat['imag'][i])

            print "PEARS ID and Field:", current_id, current_field

            # Now append all
            pears_id.append(int(current_id))
            pears_field.append(current_field)
            pears_ra.append(current_ra)
            pears_dec.append(current_dec)
            specz.append(float(spec_cat['z_spec'][spec_idx]))
            specz_source.append(str(current_specz_source))
            specz_qual.append(current_specz_qual)
            imag.append(current_imag)

            match_count += 1

        catcount += 1
    
    print "Total matches:", match_count

    # Convertt to numpy arrays and save
    pears_id = np.asarray(pears_id)
    pears_field = np.asarray(pears_field)
    pears_ra = np.asarray(pears_ra)
    pears_dec = np.asarray(pears_dec)
    specz = np.asarray(specz)
    specz_source = np.asarray(specz_source)
    specz_qual = np.asarray(specz_qual)
    imag = np.asarray(imag)

    # Now choose only those that are within the redshift range where you can see a 4000A break
    zrange_idx = np.where((specz >= 0.6) & (specz <= 1.235))[0]
    print "Number of galaxies in SPZ paper within redshift range:", len(zrange_idx)

    pears_id = pears_id[zrange_idx]
    pears_field = pears_field[zrange_idx]
    pears_ra = pears_ra[zrange_idx]
    pears_dec = pears_dec[zrange_idx]
    specz = specz[zrange_idx]
    specz_source = specz_source[zrange_idx]
    specz_qual = specz_qual[zrange_idx]
    imag = imag[zrange_idx]

    # Save to ASCII file
    data = np.array(zip(pears_id, pears_field, pears_ra, pears_dec, specz, specz_source, specz_qual, imag),\
        dtype=[('pears_id', int), ('pears_field', '|S7'), ('pears_ra', float), ('pears_dec', float), \
        ('specz', float), ('specz_source', '|S10'), ('specz_qual', '|S1'), ('imag', float)])
    np.savetxt(massive_galaxies_dir + 'spz_paper_sample_zrange.txt', data, \
        fmt=['%d', '%s', '%.6f', '%.6f', '%.4f', '%s', '%s', '%.2f'],\
        delimiter=' ', header='pearsid field ra dec specz specz_source specz_qual imag')

    return None

def main():
    get_sample_in_zrange()
    #sys.exit(0)

    # Read in file generated by above function which saved all galaxies within zrange
    # redshift cut was applied on zspec
    samp_zrange = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample_zrange.txt', dtype=None, names=True)

    # Now get pears data to get NetSig and D4000
    final_sample = 0

    # You have to create blank lists and rewrite them 
    # because you are using continue below and you cannot
    # just reuse the arrays read in from the genfromtxt 
    # above because the indices will be different after 
    # continue is used for some galaxies.
    id_list = []
    field_list = []
    ra_list = []
    dec_list = []
    zspec_list = []
    zspec_source_list = []
    zspec_qual_list = []
    netsig_list = []
    d4000_list = []
    d4000_err_list = []
    imag_list = []
    for j in range(len(samp_zrange)):

        # Get data
        current_id = samp_zrange['pearsid'][j]
        current_field = samp_zrange['field'][j]
        current_zspec = samp_zrange['specz'][j]

        print "At:", current_field, current_id, "with zspec:", current_zspec

        grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, current_field)

        if return_code == 0:
            print current_field, current_id,
            print "Skipping due to an error with the obs data. See the error message just above this one.",
            print "Moving to the next galaxy."
            continue

        if netsig_chosen < 10:
            print current_field, current_id,
            print "Skipping due to low NetSig:", netsig_chosen
            continue

        # Now get D4000 based on zspec
        lam_em = grism_lam_obs / (1 + current_zspec)
        flam_em = grism_flam_obs * (1 + current_zspec)
        ferr_em = grism_ferr_obs * (1 + current_zspec)

        # These two checks will only be trigerred if the galaxy in question is 
        # at the correct zspec but our wavelngth data array is incomplete.
        # So it should have been in the sample if we had all the data points
        # i.e. in these cases len(lam_em) < 88
        # I've pushed the checking limits a bit inward because I also don't 
        # want the D4000 code extrapolating too much.
        if lam_em[-1] < 4150:
            print current_field, current_id,
            print "Skipping due to incomplete wavelength array."
            continue
        if lam_em[0] > 3850:
            print current_field, current_id,
            print "Skipping due to incomplete wavelength array."
            continue

        #try:
        d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)
        #except IndexError:
        #    print current_field, current_id
        #    print lam_em, len(lam_em)
        #    sys.exit(0)

        if d4000 < 1.1:
            print current_field, current_id,
            print "Skipping due to low D4000:", d4000
            continue

        # These are some additional checks here
        # Both of these done in post processing for now
        # I've computed SPZs regardless of the specz quality and D4000 err
        current_specz_qual = samp_zrange['specz_qual'][j]
        """
        # Skip huge D4000 errors
        if d4000_err > 0.5:  # doesn't matter if you have >= 0.5
            continue

        # Skip worst specz quality
        if current_specz_qual == '1' or current_specz_qual == 'C':
            #print "Skipping", current_field, current_id, "due to spec-z quality:", current_specz_qual
            continue
        """

        id_list.append(current_id)
        field_list.append(current_field)
        ra_list.append(samp_zrange['ra'][j])
        dec_list.append(samp_zrange['dec'][j])
        zspec_list.append(current_zspec)
        zspec_source_list.append(samp_zrange['specz_source'][j])
        zspec_qual_list.append(current_specz_qual)
        netsig_list.append(netsig_chosen)
        d4000_list.append(d4000)
        d4000_err_list.append(d4000_err)
        imag_list.append(samp_zrange['imag'][j])

        final_sample += 1

    print "Total galaxies in final sample:", final_sample

    # Resave file with D4000 and NetSig info
    pears_id = np.asarray(id_list)
    pears_field = np.asarray(field_list)
    pears_ra = np.asarray(ra_list)
    pears_dec = np.asarray(dec_list)
    zspec = np.asarray(zspec_list)
    zspec_source = np.asarray(zspec_source_list)
    zspec_qual = np.asarray(zspec_qual_list)
    netsig_arr = np.asarray(netsig_list)
    d4000_arr = np.asarray(d4000_list)
    d4000_err_arr = np.asarray(d4000_err_list)
    imag_arr = np.asarray(imag_list)

    # Resave to ASCII file
    data = np.array(zip(pears_id, pears_field, pears_ra, pears_dec, zspec, zspec_source, zspec_qual, netsig_arr, d4000_arr, d4000_err_arr, imag_arr),\
        dtype=[('pears_id', int), ('pears_field', '|S7'), ('pears_ra', float), ('pears_dec', float), \
        ('zspec', float), ('zspec_source', '|S10'), ('zspec_qual', '|S1'), ('netsig_arr', float), \
        ('d4000_arr', float), ('d4000_err_arr', float), ('imag_arr', float)])
    np.savetxt(massive_galaxies_dir + 'spz_paper_sample.txt', data, \
        fmt=['%d', '%s', '%.6f', '%.6f', '%.4f', '%s', '%s', '%.2f', '%.2f', '%.2f', '%.2f'],\
        delimiter=' ', header='pearsid field ra dec zspec zspec_source zspec_qual netsig d4000 d4000_err imag')

    return None

if __name__ == '__main__':
    main()
    sys.exit()