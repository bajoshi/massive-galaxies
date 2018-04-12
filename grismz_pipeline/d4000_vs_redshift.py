from __future__ import division

import numpy as np

import sys
import os

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck15

home = os.getenv('HOME')
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"

sys.path.append(massive_galaxies_dir + 'grism_pipeline/')
import new_refine_grismz_gridsearch_parallel as ngp

if __name__ == '__main__':
    
    # read in matched files to get id
    matched_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_north_matched_3d.txt', \
        dtype=None, names=True, skip_header=1)
    matched_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_south_matched_santini_3d.txt', \
        dtype=None, names=True, skip_header=1)

    allcats = [matched_cat_n, matched_cat_s]

    catcount = 0
    for cat in allcats:

        if catcount == 0:
            fieldname = 'GOODS-N'
        elif catcount == 1:
            fieldname = 'GOODS-S'

        unique_ids = np.unique(cat['pearsid'])

        for current_id in unique_ids:

            # get data and then d4000
            lam_obs, flam_obs, ferr_obs, pa_chosen, netsig_chosen, return_code = ngp.get_data(current_id, fieldname)

            current_zphot = 

            lam_em = lam_obs / (1 + zspec_arr[i])
            flam_em = flam_obs * (1 + zspec_arr[i])
            ferr_em = ferr_obs * (1 + zspec_arr[i])

            d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

    # read in all arrays neded
    id_n = np.load(massive_figures_dir + 'full_run/id_list_gn.npy')
    id_s = np.load(massive_figures_dir + 'full_run/id_list_gs.npy')

    d4000_n = np.load(massive_figures_dir + 'full_run/d4000_list_gn.npy')
    d4000_s = np.load(massive_figures_dir + 'full_run/d4000_list_gs.npy')

    d4000_err_n = np.load(massive_figures_dir + 'full_run/d4000_err_list_gn.npy')
    d4000_err_s = np.load(massive_figures_dir + 'full_run/d4000_err_list_gs.npy')

    zphot_n = np.load(massive_figures_dir + 'full_run/zphot_list_gn.npy')
    zphot_s = np.load(massive_figures_dir + 'full_run/zphot_list_gs.npy')

    zgrism_n = np.load(massive_figures_dir + 'full_run/zgrism_list_gn.npy')
    zgrism_s = np.load(massive_figures_dir + 'full_run/zgrism_list_gs.npy')

    # Now replace zphot with zgrism for those galaxies whose D4000 is >= 1.5
    idx_to_replace_n = np.where(d4000_n >= 1.5)[0]
    redshift_n = zphot_n
    redshift_n[idx_to_replace_n] = zgrism_n[idx_to_replace_n]

    idx_to_replace_s = np.where(d4000_s >= 1.5)[0]
    redshift_s = zphot_s
    redshift_s[idx_to_replace_s] = zgrism_s[idx_to_replace_s]

    # concatenate north and south
    redshift_pears_plot = np.concatenate((redshift_n, redshift_s))
    d4000_pears_plot = np.concatenate((d4000_n, d4000_s))
    d4000_err_pears_plot = np.concatenate((d4000_err_n, d4000_err_s))

    # Only consider finite elements
    valid_idx1 = np.where(np.isfinite(redshift_pears_plot))[0]
    valid_idx2 = np.where(np.isfinite(d4000_pears_plot))[0]
    valid_idx3 = np.where(np.isfinite(d4000_err_pears_plot))[0]

    valid_idx = reduce(np.intersect1d, (valid_idx1, valid_idx2, valid_idx3))

    redshift_pears_plot = redshift_pears_plot[valid_idx]
    d4000_pears_plot = d4000_pears_plot[valid_idx]
    d4000_err_pears_plot = d4000_err_pears_plot[valid_idx]

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

    #ax.set_ylim(0.5, 5.0)

    # parallel x axis for age of the Universe
    # This solution came from 
    # http://www.astropy.org/astropy-tutorials/edshift_plot.html
    ax2 = ax.twiny()
    ages = np.arange(3,9,0.5)*u.Gyr

    ageticks = [z_at_value(Planck15.age, age) for age in ages]
    ax2.set_xticks(ageticks)
    ax2.set_xticklabels(['{:g}'.format(age) for age in ages.value])

    ax.set_xlim(0.5,1.3)
    ax2.set_xlim(0.5,1.3)

    ax2.set_xlabel(r'$\mathrm{Time\ since\ Big\ Bang\ (Gyr)}$', fontsize=15)

    # save the figure
    fig.savefig(massive_figures_dir + 'd4000_redshift.eps', dpi=300, bbox_inches='tight')

    sys.exit(0)