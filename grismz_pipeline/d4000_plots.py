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

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
sys.path.append(massive_galaxies_dir + 'codes/')
import mag_hist as mh
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

            #print lam_em
            d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

            # Now replace zphot with zgrism for those galaxies whose D4000 is >= 1.4
            if np.isfinite(d4000) and np.isfinite(d4000_err):
                if d4000 >= 1.4:

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

def make_d4000_vs_redshift_plot():

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
    fmt='.', color='k', markeredgecolor='k', capsize=0, markersize=0.65, elinewidth=0.1)

    ax.axhline(y=1, linewidth=1, linestyle='--', color='r', zorder=10)

    # labels and grid
    ax.set_xlabel(r'$\mathrm{Redshift}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{D}4000$', fontsize=15)
    ax.grid(True, color=mh.rgb_to_hex(240, 240, 240))

    # parallel x axis for age of the Universe
    # This solution came from 
    # http://www.astropy.org/astropy-tutorials/edshift_plot.html
    ax2 = ax.twiny()

    ages = np.arange(3,9,0.5)*u.Gyr
    ageticks = [z_at_value(Planck15.age, age) for age in ages]
    ax2.set_xticks(ageticks)

    ages_ticklabels = ['{:g}'.format(age) for age in ages.value]
    ax2.set_xticklabels(ages_ticklabels)

    ax2.set_xlim(0.5, 1.3)
    ax.set_xlim(0.5, 1.3)
    ax.set_ylim(0.5, 4.0)

    ax2.set_xlabel(r'$\mathrm{Time\ since\ Big\ Bang\ (Gyr)}$', fontsize=15)

    # Turn on minor ticks
    ax.minorticks_on()  # Only ax and not ax2. See comment below.
    ax2.minorticks_off()

    """
    Struggled iwth the following error for a couple days (!!)
    Everytime I tried to show the figure or save it, I got the error:
    /Users/bhavinjoshi/anaconda/lib/python2.7/site-packages/matplotlib/ticker.py:2531: 
    RuntimeWarning: invalid value encountered in log10
    x = int(np.round(10 ** (np.log10(majorstep) % 1)))
    ValueError: cannot convert float NaN to integer

    Later, I turned off the plotting of minor ticks that I'd turned on
    in my matplotlibrc by default and it worked. I think, the reason for 
    it to fail with that error is because I was giving it these "weird" 
    locations for the major ticks on ax2 (age ticks) and then when I 
    asked to turn on minor ticks on the ax2 axis it has no clue where 
    to put them.
    """

    # save the figure
    fig.savefig(massive_figures_dir + 'd4000_redshift.eps', dpi=300, bbox_inches='tight')

    return None

def make_d4000_hist():

    # read arrays
    d4000_pears_arr = np.load(massive_figures_dir + 'all_d4000_arr.npy')

    print "Number of galaxies with D4000 measurements:", len(d4000_pears_arr)

    # Only consider finite elements
    valid_idx = np.where(np.isfinite(d4000_pears_arr))[0]
    d4000_pears_plot = d4000_pears_arr[valid_idx]

    # ----------------------- PLOT ----------------------- #
    # PEARS dn4000 histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get total bins and plot histogram
    iqr = np.std(d4000_pears_plot, dtype=np.float64)
    binsize = 2*iqr*np.power(len(d4000_pears_plot),-1/3)
    totalbins = np.floor((max(d4000_pears_plot) - min(d4000_pears_plot))/binsize)

    ncount, edges, patches = ax.hist(d4000_pears_plot, totalbins, range=[0.0,4.0], color='lightgray', align='mid', zorder=10)
    ax.grid(True, color=mh.rgb_to_hex(240, 240, 240))

    # shade the selection region
    edges_plot = np.where(edges >= 1.1)[0]
    patches_plot = [patches[edge_ind] for edge_ind in edges_plot[:-1]]
    # I put in the [:-1] because for some reason edges was 1 element longer than patches
    col = np.full(len(patches_plot), 'lightblue', dtype='|S9')
    # make sure the length of the string given in the array initialization is the same as the color name
    for c, p in zip(col, patches_plot):
        plt.setp(p, 'facecolor', c)

    ax.set_xlabel(r'$\mathrm{D}4000$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    d40001p1 = np.where(d4000_pears_plot >= 1.1)[0]
    print "Number of galaxies with valid D4000 measurements in plot:", len(d4000_pears_plot)
    print "Number of galaxies with D4000 >= 1.1:", len(d40001p1)
    print "Fraction of total galaxies with D4000 >= 1.1:", len(d40001p1) / len(d4000_pears_plot)

    # save figure
    fig.savefig(massive_figures_dir + 'pears_d4000_hist.eps', dpi=300, bbox_inches='tight')

    return None

def make_redshift_hist():

    # read arrays
    zgrism_n = np.load(massive_figures_dir + 'full_run/zgrism_list_gn.npy')
    zgrism_s = np.load(massive_figures_dir + 'full_run/zgrism_list_gs.npy')

    print len(zgrism_n), len(zgrism_s)

    # concatenate
    redshift_pears_arr = np.concatenate((zgrism_n, zgrism_s))

    # Only consider finite elements
    valid_idx = np.where(np.isfinite(redshift_pears_arr))[0]
    redshift_pears_plot = redshift_pears_arr[valid_idx]

    # ----------------------- PLOT ----------------------- #
    # PEARS dn4000 histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get total bins and plot histogram
    iqr = np.std(redshift_pears_plot, dtype=np.float64)
    binsize = 2*iqr*np.power(len(redshift_pears_plot),-1/3)
    totalbins = np.floor((max(redshift_pears_plot) - min(redshift_pears_plot))/binsize)

    ncount, edges, patches = ax.hist(redshift_pears_plot, totalbins, \
        color='red', histtype='step', align='mid', zorder=10)
    ax.grid(True, color=mh.rgb_to_hex(240, 240, 240))

    ax.set_xlabel(r'$\mathrm{z_{grism}}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    # save figure
    fig.savefig(massive_figures_dir + 'pears_redshift_hist.eps', dpi=300, bbox_inches='tight')

    return None

if __name__ == '__main__':
    
    #make_d4000_vs_redshift_plot()
    make_d4000_hist()
    #make_redshift_hist()

    sys.exit(0)