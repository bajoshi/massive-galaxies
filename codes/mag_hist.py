from __future__ import division

import numpy as np

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')  # does not have a trailing slash
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"

# this following rgb_to_hex function came from
# https://stackoverflow.com/a/214657
def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)

if __name__ == '__main__':
    
    # read in PEARS master catalogs 
    pears_master_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag', 'netsig_corr'], usecols=(0,1,2,3,6))
    pears_master_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag', 'netsig_corr'], usecols=(0,1,2,3,6))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_master_ncat['dec'] = pears_master_ncat['dec'] - dec_offset_goodsn_v19

    # read in refined redshifts catalogs
    pears_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-N.txt',\
     dtype=None, names=True, skip_header=1)
    pears_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-S.txt',\
     dtype=None, names=True, skip_header=1)

    print "Number of objects in refined catalog for GOODS-N", len(pears_cat_n)
    print "Number of objects in refined catalog for GOODS-S", len(pears_cat_s)

    # concatenate arrays for plotting
    north_imag = pears_master_ncat['imag']
    south_imag = pears_master_scat['imag']
    total_imag_master = np.concatenate((north_imag, south_imag))

    # have to loop over both refined catalogs to get magnitudes for the refined sample
    allrefcats = [pears_cat_n, pears_cat_s]
    north_ref_imag = []
    south_ref_imag = []

    for cat in allrefcats:
        for i in range(len(cat)):
            current_id = cat['pearsid'][i]
            current_field = cat['field'][i]

            if current_field == 'GOODS-N':
                id_idx = np.where(pears_master_ncat['id'] == current_id)[0]
                north_ref_imag.append(pears_master_ncat['imag'][id_idx])
            elif current_field == 'GOODS-S':
                id_idx = np.where(pears_master_scat['id'] == current_id)[0]
                south_ref_imag.append(pears_master_scat['imag'][id_idx])

    north_ref_imag = np.asarray(north_ref_imag)
    south_ref_imag = np.asarray(south_ref_imag)
    total_imag_ref = np.concatenate((north_ref_imag, south_ref_imag))
 
    # make histograms
    # modify rc Params
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.sans-serif"] = ["Computer Modern Sans"]
    mpl.rcParams["text.usetex"] = True  
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"

    gs = gridspec.GridSpec(20,20)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=10, hspace=10)

    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(gs[0:10,:])
    ax2 = fig.add_subplot(gs[10:,:])

    # define colors
    myblue = rgb_to_hex(0, 100, 180)
    myred = rgb_to_hex(214, 39, 40)  # tableau 20 red

    ax2.set_xlabel('i-band magnitude', fontsize=14)

    ax1.hist(total_imag_master, 40, range=[14,27], color=myblue, zorder=10)
    ax2.hist(total_imag_ref, 40, range=[14,27], color=myred, zorder=10)

    ax1.set_xticklabels(ax1.get_xticks().tolist(), size=12)
    ax1.set_yticklabels(ax1.get_yticks().tolist(), size=12)
    ax2.set_xticklabels(ax2.get_xticks().tolist(), size=12)
    ax2.set_yticklabels(ax2.get_yticks().tolist(), size=12)

    ax1.minorticks_on()
    ax1.tick_params('both', width=1, length=3, which='minor')
    ax1.tick_params('both', width=1, length=4.7, which='major')
    ax1.grid(True, alpha=0.4)

    ax2.minorticks_on()
    ax2.tick_params('both', width=1, length=3, which='minor')
    ax2.tick_params('both', width=1, length=4.7, which='major')
    ax2.grid(True, alpha=0.4)

    fig.savefig(massive_figures_dir + 'mag_histograms.eps', dpi=150, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    # ---------------- repeating the above plot but overplotting instead of two separate plots ----------------  #
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('i-band magnitude', fontsize=14)

    ax.hist(total_imag_master, 40, range=[14,27], color=myblue, zorder=10)
    ax.hist(total_imag_ref, 40, range=[14,27], color=myred, zorder=10)

    ax.set_xticklabels(ax.get_xticks().tolist(), size=12)
    ax.set_yticklabels(ax.get_yticks().tolist(), size=12)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True, alpha=0.4)

    ax.text(0.05, 0.87, r'$\mathrm{Total\ PEARS\ catalog}$',\
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color=myblue, size=14)
    ax.text(0.05, 0.81, r'$\mathrm{Galaxies\ with\ refined}$' + '\n' +  r'$\mathrm{redshifts}$',\
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color=myred, size=14)

    fig.savefig(massive_figures_dir + 'mag_histograms_overplot.eps', dpi=150, bbox_inches='tight')

    sys.exit(0)