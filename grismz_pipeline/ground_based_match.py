from __future__ import division

import numpy as np

import sys
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
figs_dir = home + "/Desktop/FIGS/"

if __name__ == '__main__':

    # Read in MOSDEF and Barger catalogs
    mosdef_cat = np.genfromtxt(home + '/Desktop/MOSDEF_survey_final_redshift_release.txt', \
        dtype=None, names=True, skip_header=1)
    
    # the Barger catalog has to be read line by line because it has gaps and can't be read with genfromtxt
    with open(home + '/Desktop/barger_2008_specz.cat') as f:
        lines = f.readlines()

        # initialize arrays
        barger_ra = []
        barger_dec = []
        barger_zspec = []
        barger_kflux = []
        barger_kmag = []
        count = 0

        # loop line by line
        for line in lines[32:]:  # skipe 32 lines of header
            a = line.split()
            if len(a) > 11:
                if a[11] == 's' or a[11] == 'n':
                    continue
                else:
                    barger_ra.append(float(a[1]))
                    barger_dec.append(float(a[2]))
                    barger_zspec.append(float(a[11]))
                    barger_kflux.append(float(a[3]))
                    barger_kmag.append(float(a[5]))

    # convert to numpy arrays
    barger_ra = np.asarray(barger_ra)
    barger_dec = np.asarray(barger_dec)
    barger_zspec = np.asarray(barger_zspec)
    barger_kflux = np.asarray(barger_kflux)
    barger_kmag = np.asarray(barger_kmag)   

    # Now match and check redshifts
    matchcount = 0
    barger_zspec_list_tocompare = []
    mosdef_zspec_list_tocompare = []

    for i in range(len(barger_ra)):

        current_ra = barger_ra[i]
        current_dec = barger_dec[i]

        mosdef_idx = np.where((abs(mosdef_cat['RA'] - current_ra) < 0.5/3600) & (abs(mosdef_cat['DEC'] - current_dec) < 0.5/3600))[0]

        if len(mosdef_idx) == 1:
            matchcount += 1

            barger_zspec_list_tocompare.append(barger_zspec[i])
            mosdef_zspec_list_tocompare.append(mosdef_cat['Z_MOSFIRE'][mosdef_idx])

    print "Number of matches:", matchcount

    # convert to numpy arrays
    z_barger = np.asarray(barger_zspec_list_tocompare)
    z_mosdef = np.asarray(mosdef_zspec_list_tocompare)

    z_barger = z_barger.reshape(int(matchcount))
    z_mosdef = z_mosdef.reshape(int(matchcount))

    # Only show valid redshifts
    valid_idx1 = np.where((z_barger > 0) & (z_barger <= 4))[0]
    valid_idx2 = np.where((z_mosdef > 0) & (z_mosdef <= 4))[0]
    valid_idx = reduce(np.intersect1d, (valid_idx1, valid_idx2))

    z_barger = z_barger[valid_idx]
    z_mosdef = z_mosdef[valid_idx]

    print "Total valid objects:", len(z_barger)

    # plot
    # --------- redshift accuracy comparison ---------- # 
    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[:8,:])
    ax2 = fig.add_subplot(gs[8:,:])

    # ------- z_mock vs test_redshift ------- #
    ax1.plot(z_barger, z_mosdef, 'o', markersize=2.0, color='k', markeredgecolor='k', zorder=10)
    ax1.plot(np.arange(0,8,0.01), np.arange(0,8,0.01), '--', color='r', linewidth=2.0)

    ax1.set_xlim(0.0, 4.0)
    ax1.set_ylim(0.0, 4.0)

    ax1.set_ylabel(r'$\mathrm{z_{MOSDEF}}$', fontsize=12, labelpad=-2)
    ax1.xaxis.set_ticklabels([])

    # ------- residuals ------- #
    resid = (z_barger - z_mosdef)/(1+z_barger)
    ax2.plot(z_barger, (z_barger - z_mosdef)/(1+z_barger), 'o', markersize=2.0, color='k', markeredgecolor='k', zorder=10)
    ax2.axhline(y=0, linestyle='--', color='r')

    ax2.set_xlim(0.0, 4.0)
    #ax2.set_xticklabels([])

    ax2.set_xlabel(r'$\mathrm{z_{Barger}}$', fontsize=12)
    ax2.set_ylabel(r'$\mathrm{Residuals}$', fontsize=12, labelpad=-1)

    plt.show()
    fig.savefig(massive_figures_dir + 'model_mockspectra_fits/groundbased_zspec_comparison.eps', dpi=300, bbox_inches='tight')

    sys.exit(0)