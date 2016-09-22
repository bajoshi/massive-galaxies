from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys
import glob

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"

def get_stellar_masses_redshifts(figs_mat, field, threed_cat):

    stellarmass = []
    redshift = []
    use_phot = []
    redshift_type = []

    if (field == 'gn1') or (field == 'gn2'):

        threed_n_ind = np.where(threed_cat[1].data['field'] == 'GOODS-N')[0]

        for i in figs_mat['threed_north_idv41']:
            index = np.where(threed_cat[1].data['id'][threed_n_ind] == i)[0]
            stellarmass.append(threed_cat[1].data['lmass'][threed_n_ind][index])
            if threed_cat[1].data['z_spec'][threed_n_ind][index] != -1.0: # i.e. use spectroscopic redshift if available
                redshift.append(threed_cat[1].data['z_spec'][threed_n_ind][index])
                redshift_type.append('spectro_z')
            else:
                redshift.append(threed_cat[1].data['z_peak'][threed_n_ind][index])
                redshift_type.append('photo_z')
            use_phot.append(threed_cat[1].data['use_phot'][threed_n_ind][index])

    elif (field == 'gs1') or (field == 'gs2'):

        threed_s_ind = np.where(threed_cat[1].data['field'] == 'GOODS-S')[0]

        for i in figs_mat['threed_south_idv41']:
            index = np.where(threed_cat[1].data['id'][threed_s_ind] == i)[0]
            stellarmass.append(threed_cat[1].data['lmass'][threed_s_ind][index])
            if threed_cat[1].data['z_spec'][threed_s_ind][index] != -1.0: # i.e. use spectroscopic redshift if available
                redshift.append(threed_cat[1].data['z_spec'][threed_s_ind][index])
                redshift_type.append('spectro_z')
            else:
                redshift.append(threed_cat[1].data['z_peak'][threed_s_ind][index])
                redshift_type.append('photo_z')
            use_phot.append(threed_cat[1].data['use_phot'][threed_s_ind][index])

    stellarmass = np.asarray(stellarmass)
    stellarmass = stellarmass.reshape(len(stellarmass))

    redshift = np.asarray(redshift)
    redshift = redshift.reshape(len(redshift))

    redshift_type = np.asarray(redshift_type)
    redshift_type = redshift_type.reshape(len(redshift_type))

    use_phot = np.asarray(use_phot)
    use_phot = use_phot.reshape(len(use_phot))

    return stellarmass, redshift, redshift_type, use_phot

if __name__ == '__main__':

    # read 3dhst photometry cat
    threed_cat = fits.open(savefits_dir + '3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat.FITS')

    # read in matched figs and 3dhst files
    # I am ignoring GS2 for now.
    gn1_mat = np.genfromtxt(massive_galaxies_dir + 'gn1_threedhst_matches.txt', dtype=None, names=True, skip_header=1)
    gn2_mat = np.genfromtxt(massive_galaxies_dir + 'gn2_threedhst_matches.txt', dtype=None, names=True, skip_header=1)
    gs1_mat = np.genfromtxt(massive_galaxies_dir + 'gs1_threedhst_matches.txt', dtype=None, names=True, skip_header=1)
    #gs2_mat = np.genfromtxt(massive_galaxies_dir + 'gs2_threedhst_matches.txt', dtype=None, names=True, skip_header=1)

    # get stellar masses and the indices for the massive galaxies
    stellarmass_gn1, redshift_gn1, redshift_type_gn1, use_phot_gn1 = get_stellar_masses_redshifts(gn1_mat, 'gn1', threed_cat)
    stellarmass_gn2, redshift_gn2, redshift_type_gn2, use_phot_gn2 = get_stellar_masses_redshifts(gn2_mat, 'gn2', threed_cat)
    stellarmass_gs1, redshift_gs1, redshift_type_gs1, use_phot_gs1 = get_stellar_masses_redshifts(gs1_mat, 'gs1', threed_cat)

    stellarmass_gn1_indices = np.where(stellarmass_gn1 >= 11.0)[0]
    stellarmass_gn2_indices = np.where(stellarmass_gn2 >= 11.0)[0]
    stellarmass_gs1_indices = np.where(stellarmass_gs1 >= 11.0)[0]

    # print len(stellarmass_gn1_indices) + len(stellarmass_gn2_indices) + len(stellarmass_gs1_indices)
    # total 68 galaxies with stellar mass >10^11 M_sol in GN1, GN2, and GS1 combined over all redshifts.

    for i in range(len(stellarmass_gn1[stellarmass_gn1_indices])):
        if (redshift_gn1[stellarmass_gn1_indices][i] >= 1.2) and (redshift_gn1[stellarmass_gn1_indices][i] <= 1.76):
            if redshift_type_gn1[stellarmass_gn1_indices][i] == 'spectro_z':
                print gn1_mat['figs_id'][stellarmass_gn1_indices][i], stellarmass_gn1[stellarmass_gn1_indices][i], redshift_gn1[stellarmass_gn1_indices][i], redshift_type_gn1[stellarmass_gn1_indices][i]
            elif (redshift_type_gn1[stellarmass_gn1_indices][i] == 'photo_z') and (use_phot_gn1[stellarmass_gn1_indices][i] != 0):
                print gn1_mat['figs_id'][stellarmass_gn1_indices][i], stellarmass_gn1[stellarmass_gn1_indices][i], redshift_gn1[stellarmass_gn1_indices][i], redshift_type_gn1[stellarmass_gn1_indices][i]

    for i in range(len(stellarmass_gn2[stellarmass_gn2_indices])):
        if (redshift_gn2[stellarmass_gn2_indices][i] >= 1.2) and (redshift_gn2[stellarmass_gn2_indices][i] <= 1.76):
            if redshift_type_gn2[stellarmass_gn2_indices][i] == 'spectro_z':
                print gn2_mat['figs_id'][stellarmass_gn2_indices][i], stellarmass_gn2[stellarmass_gn2_indices][i], redshift_gn2[stellarmass_gn2_indices][i], redshift_type_gn2[stellarmass_gn2_indices][i]
            elif (redshift_type_gn2[stellarmass_gn2_indices][i] == 'photo_z') and (use_phot_gn2[stellarmass_gn2_indices][i] != 0):
                print gn2_mat['figs_id'][stellarmass_gn2_indices][i], stellarmass_gn2[stellarmass_gn2_indices][i], redshift_gn2[stellarmass_gn2_indices][i], redshift_type_gn2[stellarmass_gn2_indices][i]

    for i in range(len(stellarmass_gs1[stellarmass_gs1_indices])):
        if (redshift_gs1[stellarmass_gs1_indices][i] >= 1.2) and (redshift_gs1[stellarmass_gs1_indices][i] <= 1.76):
            if redshift_type_gs1[stellarmass_gs1_indices][i] == 'spectro_z':
                print gs1_mat['figs_id'][stellarmass_gs1_indices][i], stellarmass_gs1[stellarmass_gs1_indices][i], redshift_gs1[stellarmass_gs1_indices][i],redshift_type_gs1[stellarmass_gs1_indices][i]
            elif (redshift_type_gs1[stellarmass_gs1_indices][i] == 'photo_z') and (use_phot_gs1[stellarmass_gs1_indices][i] != 0):
                print gs1_mat['figs_id'][stellarmass_gs1_indices][i], stellarmass_gs1[stellarmass_gs1_indices][i], redshift_gs1[stellarmass_gs1_indices][i],redshift_type_gs1[stellarmass_gs1_indices][i]

    sys.exit(0)
