from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, AnchoredText

import os
import sys

home = os.getenv('HOME')  # No trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"

def get_ra_deg(ra_str):

    hr, ramin, rasec = np.asarray(ra_str.split(' ')).astype(np.float)

    ra = hr*15 + (ramin*15/60) + (rasec*15/3600)
 
    return ra

def get_dec_deg(dec_str):

    deg, arcmin, arcsec = np.asarray(dec_str.split(' ')).astype(np.float)

    if deg < 0.0:
        dec = deg - (arcmin/60) - (arcsec/3600)
    elif deg >= 0.0:
        dec = deg + (arcmin/60) + (arcsec/3600)

    return dec

if __name__ == '__main__':

    # Read in pointings information
    names_header = ['dataset', 'targ_name', 'ra', 'dec', 'exp_time', 'inst', 'camera', 'filter']

    figs_pt = np.genfromtxt(home + '/Desktop/FIGS/figs_pointings.txt', delimiter=',',
     dtype=None, names=names_header, skip_header=11)

    pears_pt = np.genfromtxt(home + '/Desktop/FIGS/pears_pointings.txt', delimiter=',',
     dtype=None, names=names_header, skip_header=11)

    threed_pt_12177 = np.genfromtxt(home + '/Desktop/FIGS/threed_pointings_12177.txt', delimiter=',',
     dtype=None, names=names_header, skip_header=11)

    threed_pt_11600 = np.genfromtxt(home + '/Desktop/FIGS/threed_pointings_11600.txt', delimiter=',',
     dtype=None, names=names_header, skip_header=11)

    # define FOV for wfc3 and acs
    # in arcseconds
    wfc3_half_fov = 60  # this is not exactly half! The WFC3 FOV is not exactly square
    acs_half_fov = 101

    # Convert the sexagesimal coordinates to degrees and then
    # Separate RA and DEC for north and south
    ##### FIGS #####
    figs_ra = np.zeros(len(figs_pt))
    figs_dec = np.zeros(len(figs_pt))

    for i in range(len(figs_pt)):

        figs_ra[i] = get_ra_deg(figs_pt['ra'][i])
        figs_dec[i] = get_dec_deg(figs_pt['dec'][i])

    figs_north_ind = np.where(figs_dec >= 0)[0]
    figs_north_ra = figs_ra[figs_north_ind]
    figs_north_dec = figs_dec[figs_north_ind]

    figs_south_ind = np.where(figs_dec < 0)[0]
    figs_south_ra = figs_ra[figs_south_ind]
    figs_south_dec = figs_dec[figs_south_ind]

    ##### PEARS #####
    pears_ra = np.zeros(len(pears_pt))
    pears_dec = np.zeros(len(pears_pt))

    for i in range(len(pears_pt)):

        pears_ra[i] = get_ra_deg(pears_pt['ra'][i])
        pears_dec[i] = get_dec_deg(pears_pt['dec'][i])

    pears_north_ind = np.where(pears_dec >= 0)[0]
    pears_north_ra = pears_ra[pears_north_ind]
    pears_north_dec = pears_dec[pears_north_ind]

    pears_south_ind = np.where(pears_dec < 0)[0]
    pears_south_ra = pears_ra[pears_south_ind]
    pears_south_dec = pears_dec[pears_south_ind]

    ##### 3DHST 12177 #####
    threed_12177_ra = np.zeros(len(threed_pt_12177))
    threed_12177_dec = np.zeros(len(threed_pt_12177))

    for i in range(len(threed_pt_12177)):

        threed_12177_ra[i] = get_ra_deg(threed_pt_12177['ra'][i])
        threed_12177_dec[i] = get_dec_deg(threed_pt_12177['dec'][i])

    threed_12177_north_ind = np.where(threed_12177_dec >= 0)[0]
    threed_12177_north_ra = threed_12177_ra[threed_12177_north_ind]
    threed_12177_north_dec = threed_12177_dec[threed_12177_north_ind]

    threed_12177_south_ind = np.where(threed_12177_dec < 0)[0]
    threed_12177_south_ra = threed_12177_ra[threed_12177_south_ind]
    threed_12177_south_dec = threed_12177_dec[threed_12177_south_ind]

    ## ------------------------------------------------------------------------------------------- ##
    ################ SOUTH ################
    ## ------------------------------------------------------------------------------------------- ##

    # loop over all pointings and plot them
    # FIGS south prime orbits
    figs_wfc3_south_ind = np.where((figs_pt['inst'] == 'WFC3') & (figs_dec < 0) & (figs_pt['filter'] == 'G102'))[0]

    x_figs_wfc3_south = np.zeros(len(figs_wfc3_south_ind) * 5)
    y_figs_wfc3_south = np.zeros(len(figs_wfc3_south_ind) * 5)

    for i in range(len(figs_wfc3_south_ind)):
        for j in range(4):
            if j == 0:
                x_figs_wfc3_south[j + i*5] = (figs_ra[figs_wfc3_south_ind][i] - wfc3_half_fov/3600)
                y_figs_wfc3_south[j + i*5] = (figs_dec[figs_wfc3_south_ind][i] - wfc3_half_fov/3600)
            if j == 1:
                x_figs_wfc3_south[j + i*5] = (figs_ra[figs_wfc3_south_ind][i] + wfc3_half_fov/3600)
                y_figs_wfc3_south[j + i*5] = (figs_dec[figs_wfc3_south_ind][i] - wfc3_half_fov/3600)
            if j == 2:
                x_figs_wfc3_south[j + i*5] = (figs_ra[figs_wfc3_south_ind][i] + wfc3_half_fov/3600)
                y_figs_wfc3_south[j + i*5] = (figs_dec[figs_wfc3_south_ind][i] + wfc3_half_fov/3600)
            if j == 3:
                x_figs_wfc3_south[j + i*5] = (figs_ra[figs_wfc3_south_ind][i] - wfc3_half_fov/3600)
                y_figs_wfc3_south[j + i*5] = (figs_dec[figs_wfc3_south_ind][i] + wfc3_half_fov/3600)
                x_figs_wfc3_south[j + i*5 +1] = (figs_ra[figs_wfc3_south_ind][i] - wfc3_half_fov/3600)
                y_figs_wfc3_south[j + i*5 +1] = (figs_dec[figs_wfc3_south_ind][i] - wfc3_half_fov/3600)

    # FIGS south parallel orbits
    figs_acs_south_ind = np.where((figs_pt['inst'] == 'ACS') & (figs_dec < 0) & (figs_pt['filter'] == 'G800L;CLEAR2L'))[0]

    x_figs_acs_south = np.zeros(len(figs_acs_south_ind) * 5)
    y_figs_acs_south = np.zeros(len(figs_acs_south_ind) * 5)

    for i in range(len(figs_acs_south_ind)):
        for j in range(4):
            if j == 0:
                x_figs_acs_south[j + i*5] = (figs_ra[figs_acs_south_ind][i] - acs_half_fov/3600)
                y_figs_acs_south[j + i*5] = (figs_dec[figs_acs_south_ind][i] - acs_half_fov/3600)
            if j == 1:
                x_figs_acs_south[j + i*5] = (figs_ra[figs_acs_south_ind][i] + acs_half_fov/3600)
                y_figs_acs_south[j + i*5] = (figs_dec[figs_acs_south_ind][i] - acs_half_fov/3600)
            if j == 2:
                x_figs_acs_south[j + i*5] = (figs_ra[figs_acs_south_ind][i] + acs_half_fov/3600)
                y_figs_acs_south[j + i*5] = (figs_dec[figs_acs_south_ind][i] + acs_half_fov/3600)
            if j == 3:
                x_figs_acs_south[j + i*5] = (figs_ra[figs_acs_south_ind][i] - acs_half_fov/3600)
                y_figs_acs_south[j + i*5] = (figs_dec[figs_acs_south_ind][i] + acs_half_fov/3600)
                x_figs_acs_south[j + i*5 +1] = (figs_ra[figs_acs_south_ind][i] - acs_half_fov/3600)
                y_figs_acs_south[j + i*5 +1] = (figs_dec[figs_acs_south_ind][i] - acs_half_fov/3600)    

    # PEARS south prime ACS obits 
    pears_acs_south_ind = np.where((pears_pt['inst'] == 'ACS') & (pears_dec < 0) & (pears_pt['filter'] == 'G800L;CLEAR2L'))[0]

    x_pears_acs_south = np.zeros(len(pears_acs_south_ind) * 5)
    y_pears_acs_south = np.zeros(len(pears_acs_south_ind) * 5)

    for i in range(len(pears_acs_south_ind)):
        for j in range(4):
            if j == 0:
                x_pears_acs_south[j + i*5] = (pears_ra[pears_acs_south_ind][i] - acs_half_fov/3600)
                y_pears_acs_south[j + i*5] = (pears_dec[pears_acs_south_ind][i] - acs_half_fov/3600)
            if j == 1:
                x_pears_acs_south[j + i*5] = (pears_ra[pears_acs_south_ind][i] + acs_half_fov/3600)
                y_pears_acs_south[j + i*5] = (pears_dec[pears_acs_south_ind][i] - acs_half_fov/3600)
            if j == 2:
                x_pears_acs_south[j + i*5] = (pears_ra[pears_acs_south_ind][i] + acs_half_fov/3600)
                y_pears_acs_south[j + i*5] = (pears_dec[pears_acs_south_ind][i] + acs_half_fov/3600)
            if j == 3:
                x_pears_acs_south[j + i*5] = (pears_ra[pears_acs_south_ind][i] - acs_half_fov/3600)
                y_pears_acs_south[j + i*5] = (pears_dec[pears_acs_south_ind][i] + acs_half_fov/3600)
                x_pears_acs_south[j + i*5 +1] = (pears_ra[pears_acs_south_ind][i] - acs_half_fov/3600)
                y_pears_acs_south[j + i*5 +1] = (pears_dec[pears_acs_south_ind][i] - acs_half_fov/3600) 

    # 3DHST 12177 south wfc3 prime orbits
    threed_12177_wfc3_south_ind = np.where((threed_pt_12177['inst'] == 'WFC3') & (threed_12177_dec < 0) & (threed_pt_12177['filter'] == 'G141'))[0]

    x_threed_12177_wfc3_south = np.zeros(len(threed_12177_wfc3_south_ind) * 5)
    y_threed_12177_wfc3_south = np.zeros(len(threed_12177_wfc3_south_ind) * 5)

    for i in range(len(threed_12177_wfc3_south_ind)):
        for j in range(4):
            if j == 0:
                x_threed_12177_wfc3_south[j + i*5] = (threed_12177_ra[threed_12177_wfc3_south_ind][i] - wfc3_half_fov/3600)
                y_threed_12177_wfc3_south[j + i*5] = (threed_12177_dec[threed_12177_wfc3_south_ind][i] - wfc3_half_fov/3600)
            if j == 1:
                x_threed_12177_wfc3_south[j + i*5] = (threed_12177_ra[threed_12177_wfc3_south_ind][i] + wfc3_half_fov/3600)
                y_threed_12177_wfc3_south[j + i*5] = (threed_12177_dec[threed_12177_wfc3_south_ind][i] - wfc3_half_fov/3600)
            if j == 2:
                x_threed_12177_wfc3_south[j + i*5] = (threed_12177_ra[threed_12177_wfc3_south_ind][i] + wfc3_half_fov/3600)
                y_threed_12177_wfc3_south[j + i*5] = (threed_12177_dec[threed_12177_wfc3_south_ind][i] + wfc3_half_fov/3600)
            if j == 3:
                x_threed_12177_wfc3_south[j + i*5] = (threed_12177_ra[threed_12177_wfc3_south_ind][i] - wfc3_half_fov/3600)
                y_threed_12177_wfc3_south[j + i*5] = (threed_12177_dec[threed_12177_wfc3_south_ind][i] + wfc3_half_fov/3600)
                x_threed_12177_wfc3_south[j + i*5 +1] = (threed_12177_ra[threed_12177_wfc3_south_ind][i] - wfc3_half_fov/3600)
                y_threed_12177_wfc3_south[j + i*5 +1] = (threed_12177_dec[threed_12177_wfc3_south_ind][i] - wfc3_half_fov/3600)

    # Make plots
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    # plot 3DHST wfc3 prime orbits
    count = 0
    for i in range(len(threed_12177_wfc3_south_ind)):
        ax.plot(x_threed_12177_wfc3_south[count:count+5], y_threed_12177_wfc3_south[count:count+5], color='gray', linewidth=1)
        #ax.fill_between(x_threed_12177_wfc3_south[count:count+2], y_threed_12177_wfc3_south[count+1], y_threed_12177_wfc3_south[count+2], color='lightyellow', alpha=0.8)
        count += 5

    # plot PEARS acs prime orbits
    count = 0
    for i in range(len(pears_acs_south_ind)):
        ax.plot(x_pears_acs_south[count:count+5], y_pears_acs_south[count:count+5], color='green', linewidth=1)
        ax.fill_between(x_pears_acs_south[count:count+2], y_pears_acs_south[count+1], y_pears_acs_south[count+2], color='lightgreen', alpha=0.8)
        count += 5

    # plot FIGS acs parallel orbits
    count = 0
    for i in range(len(figs_acs_south_ind)):
        ax.plot(x_figs_acs_south[count:count+5], y_figs_acs_south[count:count+5], color='red', linewidth=1)
        ax.fill_between(x_figs_acs_south[count:count+2], y_figs_acs_south[count+1], y_figs_acs_south[count+2], color='pink', alpha=0.8)
        count += 5

    # plot FIGS wfc3 prime orbits
    count = 0
    for i in range(len(figs_wfc3_south_ind)):
        ax.plot(x_figs_wfc3_south[count:count+5], y_figs_wfc3_south[count:count+5], color='blue', linewidth=1)
        ax.fill_between(x_figs_wfc3_south[count:count+2], y_figs_wfc3_south[count+1], y_figs_wfc3_south[count+2], color='lightblue', alpha=0.8)
        count += 5

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    #gs1labelbox = TextArea("GS1", textprops=dict(color='k', size=8))
    #anc_gs1labelbox = AnchoredOffsetbox(loc=2, child=gs1labelbox, pad=0.0, frameon=False,\
    #                                     bbox_to_anchor=(0.03, 0.95),\
    #                                     bbox_transform=ax.transAxes, borderpad=0.0)
    #ax.add_artist(anc_gs1labelbox)

    #gs2labelbox = TextArea("GS2", textprops=dict(color='k', size=8))
    #anc_gs2labelbox = AnchoredOffsetbox(loc=2, child=gs2labelbox, pad=0.0, frameon=False,\
    #                                     bbox_to_anchor=(0.05, 0.95),\
    #                                     bbox_transform=ax.transAxes, borderpad=0.0)
    #ax.add_artist(anc_gs2labelbox)

    figsprimelabelbox = TextArea("FIGS-WFC3-prime; G102", textprops=dict(color='blue', size=12))
    anc_figsprimelabelbox = AnchoredOffsetbox(loc=2, child=figsprimelabelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.65, 0.95),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_figsprimelabelbox)

    figsparallellabelbox = TextArea("FIGS-ACS-parallel; G800L", textprops=dict(color='red', size=12))
    anc_figsparallellabelbox = AnchoredOffsetbox(loc=2, child=figsparallellabelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.65, 0.90),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_figsparallellabelbox)

    pearslabelbox = TextArea("PEARS; G800L", textprops=dict(color='green', size=12))
    anc_pearslabelbox = AnchoredOffsetbox(loc=2, child=pearslabelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.65, 0.85),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_pearslabelbox)

    threedlabelbox = TextArea("3DHST; G141", textprops=dict(color='gray', size=12))
    anc_threedlabelbox = AnchoredOffsetbox(loc=2, child=threedlabelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.65, 0.80),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_threedlabelbox)

    fig.savefig(massive_figures_dir + 'field_footprint_goodss.eps')
    plt.clf()

    ## ------------------------------------------------------------------------------------------- ##
    ################ NORTH ################
    ## ------------------------------------------------------------------------------------------- ##

    # Convert the sexagesimal coordinates to degrees and then
    # Separate RA and DEC for north and south
    # 3DHST 11600
    threed_11600_ra = np.zeros(len(threed_pt_11600))
    threed_11600_dec = np.zeros(len(threed_pt_11600))

    for i in range(len(threed_pt_11600)):

        threed_11600_ra[i] = get_ra_deg(threed_pt_11600['ra'][i])
        threed_11600_dec[i] = get_dec_deg(threed_pt_11600['dec'][i])

    threed_11600_north_ind = np.where(threed_11600_dec >= 0)[0]
    threed_11600_north_ra = threed_11600_ra[threed_11600_north_ind]
    threed_11600_north_dec = threed_11600_dec[threed_11600_north_ind]

    # FIGS north prime orbits
    figs_wfc3_north_ind = np.where((figs_pt['inst'] == 'WFC3') & (figs_dec >= 0) & (figs_pt['filter'] == 'G102'))[0]

    x_figs_wfc3_north = np.zeros(len(figs_wfc3_north_ind) * 5)
    y_figs_wfc3_north = np.zeros(len(figs_wfc3_north_ind) * 5)

    for i in range(len(figs_wfc3_north_ind)):
        for j in range(4):
            if j == 0:
                x_figs_wfc3_north[j + i*5] = (figs_ra[figs_wfc3_north_ind][i] - wfc3_half_fov/3600)
                y_figs_wfc3_north[j + i*5] = (figs_dec[figs_wfc3_north_ind][i] - wfc3_half_fov/3600)
            if j == 1:
                x_figs_wfc3_north[j + i*5] = (figs_ra[figs_wfc3_north_ind][i] + wfc3_half_fov/3600)
                y_figs_wfc3_north[j + i*5] = (figs_dec[figs_wfc3_north_ind][i] - wfc3_half_fov/3600)
            if j == 2:
                x_figs_wfc3_north[j + i*5] = (figs_ra[figs_wfc3_north_ind][i] + wfc3_half_fov/3600)
                y_figs_wfc3_north[j + i*5] = (figs_dec[figs_wfc3_north_ind][i] + wfc3_half_fov/3600)
            if j == 3:
                x_figs_wfc3_north[j + i*5] = (figs_ra[figs_wfc3_north_ind][i] - wfc3_half_fov/3600)
                y_figs_wfc3_north[j + i*5] = (figs_dec[figs_wfc3_north_ind][i] + wfc3_half_fov/3600)
                x_figs_wfc3_north[j + i*5 +1] = (figs_ra[figs_wfc3_north_ind][i] - wfc3_half_fov/3600)
                y_figs_wfc3_north[j + i*5 +1] = (figs_dec[figs_wfc3_north_ind][i] - wfc3_half_fov/3600)

    # FIGS north parallel orbits
    figs_acs_north_ind = np.where((figs_pt['inst'] == 'ACS') & (figs_dec >= 0) & (figs_pt['filter'] == 'G800L;CLEAR2L'))[0]

    x_figs_acs_north = np.zeros(len(figs_acs_north_ind) * 5)
    y_figs_acs_north = np.zeros(len(figs_acs_north_ind) * 5)

    for i in range(len(figs_acs_north_ind)):
        for j in range(4):
            if j == 0:
                x_figs_acs_north[j + i*5] = (figs_ra[figs_acs_north_ind][i] - acs_half_fov/3600)
                y_figs_acs_north[j + i*5] = (figs_dec[figs_acs_north_ind][i] - acs_half_fov/3600)
            if j == 1:
                x_figs_acs_north[j + i*5] = (figs_ra[figs_acs_north_ind][i] + acs_half_fov/3600)
                y_figs_acs_north[j + i*5] = (figs_dec[figs_acs_north_ind][i] - acs_half_fov/3600)
            if j == 2:
                x_figs_acs_north[j + i*5] = (figs_ra[figs_acs_north_ind][i] + acs_half_fov/3600)
                y_figs_acs_north[j + i*5] = (figs_dec[figs_acs_north_ind][i] + acs_half_fov/3600)
            if j == 3:
                x_figs_acs_north[j + i*5] = (figs_ra[figs_acs_north_ind][i] - acs_half_fov/3600)
                y_figs_acs_north[j + i*5] = (figs_dec[figs_acs_north_ind][i] + acs_half_fov/3600)
                x_figs_acs_north[j + i*5 +1] = (figs_ra[figs_acs_north_ind][i] - acs_half_fov/3600)
                y_figs_acs_north[j + i*5 +1] = (figs_dec[figs_acs_north_ind][i] - acs_half_fov/3600) 

    # PEARS north prime ACS obits 
    pears_acs_north_ind = np.where((pears_pt['inst'] == 'ACS') & (pears_dec >= 0) & (pears_pt['filter'] == 'G800L;CLEAR2L'))[0]

    x_pears_acs_north = np.zeros(len(pears_acs_north_ind) * 5)
    y_pears_acs_north = np.zeros(len(pears_acs_north_ind) * 5)

    for i in range(len(pears_acs_north_ind)):
        for j in range(4):
            if j == 0:
                x_pears_acs_north[j + i*5] = (pears_ra[pears_acs_north_ind][i] - acs_half_fov/3600)
                y_pears_acs_north[j + i*5] = (pears_dec[pears_acs_north_ind][i] - acs_half_fov/3600)
            if j == 1:
                x_pears_acs_north[j + i*5] = (pears_ra[pears_acs_north_ind][i] + acs_half_fov/3600)
                y_pears_acs_north[j + i*5] = (pears_dec[pears_acs_north_ind][i] - acs_half_fov/3600)
            if j == 2:
                x_pears_acs_north[j + i*5] = (pears_ra[pears_acs_north_ind][i] + acs_half_fov/3600)
                y_pears_acs_north[j + i*5] = (pears_dec[pears_acs_north_ind][i] + acs_half_fov/3600)
            if j == 3:
                x_pears_acs_north[j + i*5] = (pears_ra[pears_acs_north_ind][i] - acs_half_fov/3600)
                y_pears_acs_north[j + i*5] = (pears_dec[pears_acs_north_ind][i] + acs_half_fov/3600)
                x_pears_acs_north[j + i*5 +1] = (pears_ra[pears_acs_north_ind][i] - acs_half_fov/3600)
                y_pears_acs_north[j + i*5 +1] = (pears_dec[pears_acs_north_ind][i] - acs_half_fov/3600)

    # 3DHST 11600 north wfc3 prime orbits
    threed_11600_wfc3_north_ind = np.where((threed_pt_11600['inst'] == 'WFC3') & (threed_11600_dec >= 0) & (threed_pt_11600['filter'] == 'G141'))[0]

    x_threed_11600_wfc3_north = np.zeros(len(threed_11600_wfc3_north_ind) * 5)
    y_threed_11600_wfc3_north = np.zeros(len(threed_11600_wfc3_north_ind) * 5)

    for i in range(len(threed_11600_wfc3_north_ind)):
        for j in range(4):
            if j == 0:
                x_threed_11600_wfc3_north[j + i*5] = (threed_11600_ra[threed_11600_wfc3_north_ind][i] - wfc3_half_fov/3600)
                y_threed_11600_wfc3_north[j + i*5] = (threed_11600_dec[threed_11600_wfc3_north_ind][i] - wfc3_half_fov/3600)
            if j == 1:
                x_threed_11600_wfc3_north[j + i*5] = (threed_11600_ra[threed_11600_wfc3_north_ind][i] + wfc3_half_fov/3600)
                y_threed_11600_wfc3_north[j + i*5] = (threed_11600_dec[threed_11600_wfc3_north_ind][i] - wfc3_half_fov/3600)
            if j == 2:
                x_threed_11600_wfc3_north[j + i*5] = (threed_11600_ra[threed_11600_wfc3_north_ind][i] + wfc3_half_fov/3600)
                y_threed_11600_wfc3_north[j + i*5] = (threed_11600_dec[threed_11600_wfc3_north_ind][i] + wfc3_half_fov/3600)
            if j == 3:
                x_threed_11600_wfc3_north[j + i*5] = (threed_11600_ra[threed_11600_wfc3_north_ind][i] - wfc3_half_fov/3600)
                y_threed_11600_wfc3_north[j + i*5] = (threed_11600_dec[threed_11600_wfc3_north_ind][i] + wfc3_half_fov/3600)
                x_threed_11600_wfc3_north[j + i*5 +1] = (threed_11600_ra[threed_11600_wfc3_north_ind][i] - wfc3_half_fov/3600)
                y_threed_11600_wfc3_north[j + i*5 +1] = (threed_11600_dec[threed_11600_wfc3_north_ind][i] - wfc3_half_fov/3600)

    # Make plots
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    ## plot 3DHST wfc3 prime orbits
    #count = 0
    #for i in range(len(threed_11600_wfc3_north_ind)):
    #    ax.plot(x_threed_11600_wfc3_north[count:count+5], y_threed_11600_wfc3_north[count:count+5], color='gray', linewidth=1)
    #    count += 5

    # plot PEARS acs prime orbits
    count = 0
    for i in range(len(pears_acs_north_ind)):
        ax.plot(x_pears_acs_north[count:count+5], y_pears_acs_north[count:count+5], color='green', linewidth=1)
        print x_pears_acs_north[count:count+5], y_pears_acs_north[count:count+5]
        ax.fill_between(x_pears_acs_north[count:count+2], y_pears_acs_north[count+1], y_pears_acs_north[count+2], color='lightgreen', alpha=0.8)
        count += 5

    ## plot FIGS acs parallel orbits
    #count = 0
    #for i in range(len(figs_acs_north_ind)):
    #    ax.plot(x_figs_acs_north[count:count+5], y_figs_acs_north[count:count+5], color='red', linewidth=1)
    #    ax.fill_between(x_figs_acs_north[count:count+2], y_figs_acs_north[count+1], y_figs_acs_north[count+2], color='pink', alpha=0.8)
    #    count += 5

    ## plot FIGS wfc3 prime orbits
    #count = 0
    #for i in range(len(figs_wfc3_north_ind)):
    #    ax.plot(x_figs_wfc3_north[count:count+5], y_figs_wfc3_north[count:count+5], color='blue', linewidth=1)
    #    ax.fill_between(x_figs_wfc3_north[count:count+2], y_figs_wfc3_north[count+1], y_figs_wfc3_north[count+2], color='lightblue', alpha=0.8)
    #    count += 5

    #ax.set_xlim(188.9, 189.7)
    #ax.set_ylim(62.1, 62.4)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)
    
    #gn1labelbox = TextArea("GN1", textprops=dict(color='k', size=8))
    #anc_gn1labelbox = AnchoredOffsetbox(loc=2, child=gn1labelbox, pad=0.0, frameon=False,\
    #                                     bbox_to_anchor=(0.03, 0.95),\
    #                                     bbox_transform=ax.transAxes, borderpad=0.0)
    #ax.add_artist(anc_gn1labelbox)

    #gn2labelbox = TextArea("GN2", textprops=dict(color='k', size=8))
    #anc_gn2labelbox = AnchoredOffsetbox(loc=2, child=gn2labelbox, pad=0.0, frameon=False,\
    #                                     bbox_to_anchor=(0.05, 0.95),\
    #                                     bbox_transform=ax.transAxes, borderpad=0.0)
    #ax.add_artist(anc_gn2labelbox)

    figsprimelabelbox = TextArea("FIGS-WFC3-prime; G102", textprops=dict(color='blue', size=12))
    anc_figsprimelabelbox = AnchoredOffsetbox(loc=2, child=figsprimelabelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.65, 0.95),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_figsprimelabelbox)

    figsparallellabelbox = TextArea("FIGS-ACS-parallel; G800L", textprops=dict(color='red', size=12))
    anc_figsparallellabelbox = AnchoredOffsetbox(loc=2, child=figsparallellabelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.65, 0.90),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_figsparallellabelbox)

    pearslabelbox = TextArea("PEARS; G800L", textprops=dict(color='green', size=12))
    anc_pearslabelbox = AnchoredOffsetbox(loc=2, child=pearslabelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.65, 0.85),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_pearslabelbox)

    threedlabelbox = TextArea("3DHST; G141", textprops=dict(color='gray', size=12))
    anc_threedlabelbox = AnchoredOffsetbox(loc=2, child=threedlabelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.65, 0.80),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_threedlabelbox)

    fig.savefig(massive_figures_dir + 'field_footprint_goodsn.eps')
    plt.clf()

    sys.exit(0)

