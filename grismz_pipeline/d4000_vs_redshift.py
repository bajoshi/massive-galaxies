from __future__ import division

import numpy as np

import sys
import os

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck15

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"

sys.path.append(massive_galaxies_dir + 'grism_pipeline/')
import new_refine_grismz_gridsearch_parallel as ngp
import dn4000_catalog as dc

def get_all_d4000():

    # read in matched files to get id
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)
    # read in all arrays neded
    id_n = np.load(massive_figures_dir + 'full_run/id_list_gn.npy')
    id_s = np.load(massive_figures_dir + 'full_run/id_list_gs.npy')

    zgrism_n = np.load(massive_figures_dir + 'full_run/zgrism_list_gn.npy')
    zgrism_s = np.load(massive_figures_dir + 'full_run/zgrism_list_gs.npy')

    #  Initialize empty lists for storing final values
    redshift_pears_arr = []
    d4000_pears_arr = []
    d4000_err_pears_arr = []

    # Now loop over all objects and get d4000
    allcats = [matched_cat_n, matched_cat_s]

    catcount = 0
    for cat in allcats:

        if catcount == 0:
            fieldname = 'GOODS-N'
        elif catcount == 1:
            fieldname = 'GOODS-S'

        # amke sure all objects are unique
        pears_unique_ids, pears_unique_ids_indices = np.unique(cat['pearsid'], return_index=True)

        for i in range(len(pears_unique_ids)):

            current_id = cat['pearsid'][pears_unique_ids_indices][i]
            current_zphot = cat['zphot'][pears_unique_ids_indices][i]

            # get data and then d4000
            lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, fieldname)
            if return_code == 0:
                continue

            lam_em = lam_obs / (1 + current_zphot)
            flam_em = flam_obs * (1 + current_zphot)
            ferr_em = ferr_obs * (1 + current_zphot)

            d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

            # Now replace zphot with zgrism for those galaxies whose D4000 is >= 1.5
            if np.isfinite(d4000) and np.isfinite(d4000_err):
                if d4000 >= 1.5:

                    if fieldname == 'GOODS-N':
                        id_idx = np.where(id_n == current_id)[0]
                        current_zgrism = zgrism_n[id_idx]

                        if np.isfinite(current_zgrism):
                            d4000_pears_arr.append(d4000)
                            d4000_err_pears_arr.append(d4000_err)
                            redshift_pears_arr.append(current_zgrism)

                    elif fieldname == 'GOODS-S':
                        id_idx = np.where(id_s == current_id)[0]
                        current_zgrism = zgrism_s[id_idx]

                        if np.isfinite(current_zgrism):
                            d4000_pears_arr.append(d4000)
                            d4000_err_pears_arr.append(d4000_err)
                            redshift_pears_arr.append(current_zgrism)
                else:
                    if np.isfinite(current_zphot):
                        d4000_pears_arr.append(d4000)
                        d4000_err_pears_arr.append(d4000_err)
                        redshift_pears_arr.append(current_zphot)

        catcount += 1

    # Convert to numpy arrays
    redshift_pears_arr = np.asarray(redshift_pears_arr)
    d4000_pears_arr = np.asarray(d4000_pears_arr)
    d4000_err_pears_arr = np.asarray(d4000_err_pears_arr)

    np.save(massive_figures_dir + 'all_redshift_array.npy', redshift_pears_arr)
    np.save(massive_figures_dir + 'all_d4000_arr.npy', d4000_pears_arr)
    np.save(massive_figures_dir + 'all_d4000_err_arr.npy', d4000_err_pears_arr)

    return redshift_pears_arr, d4000_pears_arr, d4000_err_pears_arr

if __name__ == '__main__':
    
    # Only need to run this function once
    # So it can store data. After that simply
    # read the stored data. Commented out 
    # right now because I already have the arrays.
    #redshift_pears_arr, d4000_pears_arr, d4000_err_pears_arr = get_all_d4000()

    # read arrays
    redshift_pears_arr = np.load(massive_figures_dir + 'all_redshift_array.npy')
    d4000_pears_arr = np.load(massive_figures_dir + 'all_d4000_arr.npy')
    d4000_err_pears_arr = np.load(massive_figures_dir + 'all_d4000_err_arr.npy')

    # Only consider finite elements
    valid_idx1 = np.where(np.isfinite(redshift_pears_arr))[0]
    valid_idx2 = np.where(np.isfinite(d4000_pears_arr))[0]
    valid_idx3 = np.where(np.isfinite(d4000_err_pears_arr))[0]

    valid_idx = reduce(np.intersect1d, (valid_idx1, valid_idx2, valid_idx3))

    redshift_pears_plot = redshift_pears_arr[valid_idx]
    d4000_pears_plot = d4000_pears_arr[valid_idx]
    d4000_err_pears_plot = d4000_err_pears_arr[valid_idx]

    # d4000 vs redshift 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.errorbar(redshift_pears_plot, d4000_pears_plot, yerr=d4000_err_pears_plot,\
    fmt='.', color='k', markeredgecolor='k', capsize=0, markersize=4, elinewidth=0.25)

    ax.axhline(y=1, linewidth=1, linestyle='--', color='r', zorder=10)

    # labels and grid
    ax.set_xlabel(r'$\mathrm{Redshift}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{D}4000$', fontsize=15)
    ax.grid(True)

    ax.set_ylim(0.5, 4.0)

    # parallel x axis for age of the Universe
    # This solution came from 
    # http://www.astropy.org/astropy-tutorials/edshift_plot.html
    ax2 = ax.twiny()
    ages = np.arange(3,9,0.5)*u.Gyr

    ageticks = [z_at_value(Planck15.age, age) for age in ages]
    ageticks = np.asarray(ageticks)
    ageticks = ageticks[::-1]
    ax2.set_xticks(ageticks)
    ax2.set_xticklabels(['{:g}'.format(age) for age in ages.value])

    ax.set_xlim(0.5,1.3)
    ax2.set_xlim(0.5,1.3)

    ax2.set_xlabel(r'$\mathrm{Time\ since\ Big\ Bang\ (Gyr)}$', fontsize=15)

    # -------------
    """
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())

    ax2.set_yticklabels(ax2.get_yticks())

    print ageticks
    print ax.get_xticks()
    print ax.get_yticks()

    l1_list = ax.get_xticklabels()
    l2_list = ax.get_yticklabels()
    
    print ax2.get_xticks()
    print ax2.get_yticks()
    
    l3_list = ax2.get_xticklabels()
    l4_list = ax2.get_yticklabels()

    for l1 in l1_list:
        print l1.get_text()

    for l2 in l2_list:
        print l2.get_text()

    for l3 in l3_list:
        print l3.get_text()

    for l4 in l4_list:
        print l4.get_text()
    """
    # --------------

    plt.show()

    # save the figure
    #fig.savefig(massive_figures_dir + 'd4000_redshift.png', dpi=300, bbox_inches='tight')

    sys.exit(0)