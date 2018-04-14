from __future__ import division

import numpy as np
from astropy.io import fits

import sys
import os
import glob

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, AnchoredText

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
newcodes_dir = home + "/Desktop/FIGS/new_codes/"

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir)
import dn4000_catalog as dct
import cosmology_calculator as cc

import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck15

def check_sdss():

    sdss_spectra_dr9_dir = newcodes_dir + 'sdss_spectra_dr9/'
    specdetails = np.genfromtxt(sdss_spectra_dr9_dir + 'spec_details.txt', dtype=None, delimiter=' ', names=['filename', 'z', 'dn4000'])

    #for fname in glob.glob('lite/' + specdetails[filename])

    #os.path.isfile('lite/' + )

    return None

def save_speclist_sdss(plateid, mjd, fiberid, redshift_sdss, dn4000_sdss, sdss_use_indx, sig_4000break_indices_sdss, dn4000_sdss_range_indx, gen_list):

    if gen_list == 'spec':

        # write fits names to a text file to download these spectra
        fh = open(newcodes_dir + 'sdss_spectra_dr9/' + 'speclist.txt', 'wa')

        for i in range(700):
            current_plateid = plateid[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx][i+100]
            current_mjd = mjd[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx][i+100]
            current_fiberid = str(fiberid[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx][i+100])
            if len(current_fiberid) == 1:
                current_fiberid = "000" + current_fiberid
            elif len(current_fiberid) == 2:
                current_fiberid = "00" + current_fiberid
            elif len(current_fiberid) == 3:
                current_fiberid = "0" + current_fiberid
            specstring = "0" + str(current_plateid) + "/" + "spec-0" + str(current_plateid) + "-" + str(current_mjd) + "-" + current_fiberid + ".fits"
            fh.write(specstring + '\n')

        fh.close()
        del fh
    
    # WGET command used
    # cd into sdss_spectra_dr9 directory first
    # wget -nv -r -nH --cut-dirs=7 -i speclist.txt -B http://data.sdss3.org/sas/dr9/sdss/spectro/redux/26/spectra/lite/

    elif gen_list == 'spec_details':

        # write fits names to a text file to download these spectra
        fh = open(newcodes_dir + 'sdss_spectra_dr9/' + 'spec_details.txt', 'wa')

        for i in range(700):
            current_plateid = str(plateid[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx][i+100])
            current_mjd = str(mjd[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx][i+100])
            current_fiberid = str(fiberid[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx][i+100])
            if len(current_fiberid) == 1:
                current_fiberid = "000" + current_fiberid
            elif len(current_fiberid) == 2:
                current_fiberid = "00" + current_fiberid
            elif len(current_fiberid) == 3:
                current_fiberid = "0" + current_fiberid
            specstring = "0" + current_plateid + "/" + "spec-0" + current_plateid + "-" + current_mjd + "-" + current_fiberid + ".fits"

            current_z = str(redshift_sdss[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx][i+100])
            current_dn4000 = str(dn4000_sdss[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx][i+100])
            detailstring = current_z + ' ' + current_dn4000

            fh.write(specstring + ' ' + detailstring + '\n')

        fh.close()
        del fh

    return None

if __name__ == '__main__':
    
    # read in 4000 break catalogs 
    pears_cat_n = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/pears_4000break_catalog_GOODS-N.txt',\
     dtype=None, names=True, skip_header=1)
    pears_cat_s = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/pears_4000break_catalog_GOODS-S.txt',\
     dtype=None, names=True, skip_header=3)
    #gn1_cat = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/figs_gn1_4000break_catalog.txt',\
    # dtype=None, names=True, skip_header=1)
    #gn2_cat = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/figs_gn2_4000break_catalog.txt',\
    # dtype=None, names=True, skip_header=1)
    #gs1_cat = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/figs_gs1_4000break_catalog.txt',\
    # dtype=None, names=True, skip_header=1)

    # get ids from old catalog to check which ones to replace with corrected values
    pears_ids_n = pears_cat_n['pearsid']
    pears_ids_s = pears_cat_s['pearsid']

    # replace d4000 with corrected d4000 from refined redshift catalog
    # first read the refined cats and make arrays
    pears_ref_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-N.txt',\
     dtype=None, names=True, skip_header=1)
    pears_ref_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-S.txt',\
     dtype=None, names=True, skip_header=1)

    refined_ids_n = pears_ref_cat_n['pearsid']
    refined_ids_s = pears_ref_cat_s['pearsid']

    count = 0
    count_ref_n = 0
    for old_id in pears_ids_n:

        if old_id in refined_ids_n:
            id_indx = np.where(refined_ids_n == old_id)[0]
            pears_cat_n['d4000'][count] = pears_ref_cat_n['d4000'][id_indx]
            pears_cat_n['d4000_err'][count] = pears_ref_cat_n['d4000_err'][id_indx]
            pears_cat_n['redshift'][count] = pears_ref_cat_n['new_z'][id_indx]
            count_ref_n += 1

        count += 1

    count = 0
    count_ref_s = 0
    for old_id in pears_ids_s:

        if old_id in refined_ids_s:
            id_indx = np.where(refined_ids_s == old_id)[0]
            pears_cat_s['d4000'][count] = pears_ref_cat_s['d4000'][id_indx]
            pears_cat_s['d4000_err'][count] = pears_ref_cat_s['d4000_err'][id_indx]
            pears_cat_s['redshift'][count] = pears_ref_cat_s['new_z'][id_indx]
            count_ref_s += 1

        count += 1

    print count_ref_n, count_ref_s

    #### PEARS ####
    pears_redshift_indices_n = np.where((pears_cat_n['redshift'] >= 0.6) & (pears_cat_n['redshift'] <= 1.235))[0]

    # galaxies that are outside the redshift range
    # not sure how these originally got into the pears and 3dhst matched sample....need to check again
    # these were originally selected to be within the above written redshift range
    #print np.setdiff1d(np.arange(len(pears_cat)), pears_redshift_indices)  # [1136 2032 2265]

    # galaxies with significant breaks
    sig_4000break_indices_pears_n = np.where(((pears_cat_n['d4000'][pears_redshift_indices_n] / pears_cat_n['d4000_err'][pears_redshift_indices_n]) >= 3.0) &\
        ((pears_cat_n['d4000'][pears_redshift_indices_n] / pears_cat_n['d4000_err'][pears_redshift_indices_n]) <= 20.0))[0]

    # Galaxies with breaks that can be used for refining redshifts; im calling them proper breaks
    prop_4000break_indices_pears_n = \
    np.where((pears_cat_n['d4000'][pears_redshift_indices_n][sig_4000break_indices_pears_n] >= 1.05) & \
        (pears_cat_n['d4000'][pears_redshift_indices_n][sig_4000break_indices_pears_n] <= 3.5))[0]

    # same stuff for south
    pears_redshift_indices_s = np.where((pears_cat_s['redshift'] >= 0.6) & (pears_cat_s['redshift'] <= 1.235))[0]

    sig_4000break_indices_pears_s = np.where(((pears_cat_s['d4000'][pears_redshift_indices_s] / pears_cat_s['d4000_err'][pears_redshift_indices_s]) >= 3.0) &\
        ((pears_cat_s['d4000'][pears_redshift_indices_s] / pears_cat_s['d4000_err'][pears_redshift_indices_s]) <= 20.0))[0]

    prop_4000break_indices_pears_s = \
    np.where((pears_cat_s['d4000'][pears_redshift_indices_s][sig_4000break_indices_pears_s] >= 1.05) & \
        (pears_cat_s['d4000'][pears_redshift_indices_s][sig_4000break_indices_pears_s] <= 3.5))[0]

    """
    #### FIGS ####
    # galaxies with significant breaks
    sig_4000break_indices_gn1 = np.where(((gn1_cat['dn4000'] / gn1_cat['dn4000_err']) >= 3.0) &\
        ((gn1_cat['dn4000'] / gn1_cat['dn4000_err']) <= 20.0))[0]
    sig_4000break_indices_gn2 = np.where(((gn2_cat['dn4000'] / gn2_cat['dn4000_err']) >= 3.0) &\
        ((gn2_cat['dn4000'] / gn2_cat['dn4000_err']) <= 20.0))[0]
    sig_4000break_indices_gs1 = np.where(((gs1_cat['dn4000'] / gs1_cat['dn4000_err']) >= 3.0) &\
        ((gs1_cat['dn4000'] / gs1_cat['dn4000_err']) <= 20.0))[0]

    # Galaxies with believable breaks
    prop_4000break_indices_gn1 = \
    np.where((gn1_cat['dn4000'][sig_4000break_indices_gn1] >= 1.2) & \
        (gn1_cat['dn4000'][sig_4000break_indices_gn1] <= 2.5))[0]
    prop_4000break_indices_gn2 = \
    np.where((gn2_cat['dn4000'][sig_4000break_indices_gn2] >= 1.2) & \
        (gn2_cat['dn4000'][sig_4000break_indices_gn2] <= 2.5))[0]
    prop_4000break_indices_gs1 = \
    np.where((gs1_cat['dn4000'][sig_4000break_indices_gs1] >= 1.2) & \
        (gs1_cat['dn4000'][sig_4000break_indices_gs1] <= 2.5))[0]

    #print len(prop_4000break_indices_gn1)  # 6
    #print len(prop_4000break_indices_gn2)  # 7
    #print len(prop_4000break_indices_gs1)  # 5

    #print len(sig_4000break_indices_pears)  # 1226
    #print len(sig_4000break_indices_gn1)  # 33
    #print len(sig_4000break_indices_gn2)  # 37
    #print len(sig_4000break_indices_gs1)  # 28

    #### SDSS ####
    # these are from SDSS DR9
    galspecindx = fits.open(newcodes_dir + 'sdss_fits_files/' + 'galSpecindx-dr9.fits')
    galspecinfo = fits.open(newcodes_dir + 'sdss_fits_files/' + 'galSpecinfo-dr9.fits')
 
    # get id, plate, mjd, and fiber numbers
    sdss_id = galspecindx[1].data['SPECOBJID']
    plateid = galspecinfo[1].data['PLATEID']
    mjd = galspecinfo[1].data['MJD']
    fiberid = galspecinfo[1].data['FIBERID']

    # get dn4000 and redshift arrays
    dn4000_sdss = galspecindx[1].data['D4000_N_SUB']
    dn4000_err_sdss = galspecindx[1].data['D4000_N_SUB_ERR']
    redshift_sdss = galspecinfo[1].data['Z']

    # apply basic cuts
    sdss_use_indx = np.where((galspecinfo[1].data['Z_WARNING'] == 0) &\
     (galspecinfo[1].data['TARGETTYPE'] == 'GALAXY') & (galspecinfo[1].data['SPECTROTYPE'] == 'GALAXY') & (galspecinfo[1].data['PRIMTARGET'] == 64) &\
     (galspecinfo[1].data['RELIABLE'] == 1))[0]

    print len(sdss_use_indx)  # 869029

    # apply more cuts i.e. significance and break range
    sig_4000break_indices_sdss = np.where(((dn4000_sdss[sdss_use_indx] / dn4000_err_sdss[sdss_use_indx]) >= 3.0) &\
     ((dn4000_sdss[sdss_use_indx] / dn4000_err_sdss[sdss_use_indx]) <= 20.0))[0]
    dn4000_sdss_sig = dn4000_sdss[sdss_use_indx][sig_4000break_indices_sdss]
    dn4000_sdss_range_indx = np.where((dn4000_sdss_sig >= 0) & (dn4000_sdss_sig <= 3))[0]
    dn4000_sdss_plot = dn4000_sdss[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx]

    redshift_sdss_plot = redshift_sdss[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx]

    print len(dn4000_sdss_plot)  # 74343

    #save_speclist_sdss(plateid, mjd, fiberid, redshift_sdss, dn4000_sdss, sdss_use_indx, sig_4000break_indices_sdss, dn4000_sdss_range_indx, 'spec_details')
    #sys.exit(0)

    #### SHELS ####

    shels_names = ['ID', 'z', 'Dn4000']
    shels_cat = np.genfromtxt(massive_galaxies_dir + 'shels_gal_prop.txt', dtype=None, names=shels_names, usecols=(0, 4, 8), skip_header=33)

    # get dn4000 adn redshifts
    dn4000_shels = shels_cat['Dn4000']
    redshift_shels = shels_cat['z']

    # apply cuts
    shels_use_indx = np.where((dn4000_shels != -9.99) & (redshift_shels != -9.99))[0]

    dn4000_shels_plot = dn4000_shels[shels_use_indx]
    redshift_shels_plot = redshift_shels[shels_use_indx]

    # Make sure that all the galaxies in the sample are unique
    # there is the possibility of duplicate entries only in hte redshift
    # range where PEARS and FIGS overlap i.e. 1.2 <= z <= 1.32
    # the catalog will somehow have to be matched with itself?
    # it can't just be np.unique. 
    print len(pears_cat)
    print len(np.unique(pears_cat['ra']))

    # could you somehow demonstrate that the dn4000 that you get is the same that 
    # other studies got.
    # there are some sdss galaxies with the same redshifts as PEARS. 
    # IDK if there is any overlap in the fields?
    # You could also download a convincing number of SDSS spectra and run your code on them
    # and see if you get the same number for dn4000 as they do.
    """
    

    # ------------------------- PLOTS ------------------------- #

    # make histograms of dn4000 distribution for PEARS and FIGS 
    # also showing (shaded or with vertical lines) the values of 
    # dn4000 that are selected in the final sample
    # also show the numbers on top of the bars

    # PEARS dn4000 histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # create proper array for plotting
    d4000_pears_n = pears_cat_n['d4000'][pears_redshift_indices_n][sig_4000break_indices_pears_n]
    d4000_pears_plot_n = d4000_pears_n[np.where(d4000_pears_n <= 4)[0]]

    d4000_pears_s = pears_cat_s['d4000'][pears_redshift_indices_s][sig_4000break_indices_pears_s]
    d4000_pears_plot_s = d4000_pears_s[np.where(d4000_pears_s <= 4)[0]]

    d4000_pears_plot = np.concatenate((d4000_pears_plot_n, d4000_pears_plot_s))

    print len(d4000_pears_n) + len(d4000_pears_s), len(d4000_pears_plot)

    # get total bins and plot histogram
    iqr = np.std(d4000_pears_plot, dtype=np.float64)
    binsize = 2*iqr*np.power(len(d4000_pears_plot),-1/3)
    totalbins = np.floor((max(d4000_pears_plot) - min(d4000_pears_plot))/binsize)

    ncount, edges, patches = ax.hist(d4000_pears_plot, totalbins, color='lightgray', align='mid')
    ax.grid(True)

    # shade the selection region
    edges_plot = np.where(edges >= 1.05)[0]
    patches_plot = [patches[edge_ind] for edge_ind in edges_plot[:-1]]
    # I put in the [:-1] because for some reason edges was 1 element longer than patches
    col = np.full(len(patches_plot), 'lightblue', dtype='|S9')
    # make sure the length of the string given in the array initialization is the same as the color name
    for c, p in zip(col, patches_plot):
        plt.setp(p, 'facecolor', c)

    ax.set_xlabel(r'$\mathrm{D}(4000)$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    # save figure
    fig.savefig(massive_figures_dir + 'pears_d4000_hist.eps', dpi=300, bbox_inches='tight')

    """
    # FIGS dn4000 histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # create proper array for plotting
    dn4000_gn1 = gn1_cat['dn4000'][sig_4000break_indices_gn1]
    dn4000_gn2 = gn2_cat['dn4000'][sig_4000break_indices_gn2]
    dn4000_gs1 = gs1_cat['dn4000'][sig_4000break_indices_gs1]

    dn4000_figs = np.concatenate((dn4000_gn1, dn4000_gn2, dn4000_gs1))
    dn4000_figs_plot = dn4000_figs[np.where(dn4000_figs <= 3)[0]]
    #print len(np.where(dn4000_figs > 3)[0])  # 12
    print len(dn4000_figs_plot)
    # Same thing done as PEARS. I did not plot the 12 out of 98 objects 
    # that have dn4000 values greater than 3 so that my plot can look better

    # get total bins and plot histogram
    #iqr = np.std(dn4000_figs_plot, dtype=np.float64)
    #binsize = 2*iqr*np.power(len(dn4000_figs_plot),-1/3)
    #totalbins = np.floor((max(dn4000_figs_plot) - min(dn4000_figs_plot))/binsize)
    # the freedman-diaconis rule here does not seem to give me a proper number of bins.
    # which it generally does not for arrays that do not have a large number of elements.
    totalbins = 3

    ncount, edges, patches = ax.hist(dn4000_figs_plot, totalbins, color='lightgray', align='mid')
    ax.grid(True)    
   
    ## shade the selection region
    #edges_plot = np.where((edges >= 1.2) & (edges <= 2.5))[0]
    #patches_plot = [patches[edge_ind] for edge_ind in edges_plot]
    #col = np.full(len(patches_plot), 'lightblue', dtype='|S9')  
    ## make sure the length of the string given in the array initialization is the same as the color name
    #for c, p in zip(col, patches_plot):
    #    plt.setp(p, 'facecolor', c)

    # save figure
    fig.savefig(massive_figures_dir + 'figs_dn4000_hist.eps', dpi=300, bbox_inches='tight')    
    """

    # dn4000 vs redshift 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get proper redshift and d4000_err arrays for PEARS and FIGS for plotting
    # north
    redshift_pears_n = pears_cat_n['redshift'][pears_redshift_indices_n][sig_4000break_indices_pears_n]
    redshift_pears_plot_n = redshift_pears_n[np.where(d4000_pears_n <= 4)[0]]

    d4000_err_pears_n = pears_cat_n['d4000_err'][pears_redshift_indices_n][sig_4000break_indices_pears_n]
    d4000_err_pears_plot_n = d4000_err_pears_n[np.where(d4000_pears_n <= 4)[0]]

    # south 
    redshift_pears_s = pears_cat_s['redshift'][pears_redshift_indices_s][sig_4000break_indices_pears_s]
    redshift_pears_plot_s = redshift_pears_s[np.where(d4000_pears_s <= 4)[0]]

    d4000_err_pears_s = pears_cat_s['d4000_err'][pears_redshift_indices_s][sig_4000break_indices_pears_s]
    d4000_err_pears_plot_s = d4000_err_pears_s[np.where(d4000_pears_s <= 4)[0]]

    # concatenate north and south
    redshift_pears_plot = np.concatenate((redshift_pears_plot_n, redshift_pears_plot_s))
    d4000_pears_plot = np.concatenate((d4000_pears_plot_n, d4000_pears_plot_s))
    d4000_err_pears_plot = np.concatenate((d4000_err_pears_plot_n, d4000_err_pears_plot_s))

    #redshift_gn1 = gn1_cat['redshift'][sig_4000break_indices_gn1]
    #redshift_gn2 = gn2_cat['redshift'][sig_4000break_indices_gn2]
    #redshift_gs1 = gs1_cat['redshift'][sig_4000break_indices_gs1]
    #redshift_figs = np.concatenate((redshift_gn1, redshift_gn2, redshift_gs1))
    #redshift_figs_plot = redshift_figs[np.where(dn4000_figs <= 3)[0]]

    #dn4000_err_gn1 = gn1_cat['dn4000_err'][sig_4000break_indices_gn1]
    #dn4000_err_gn2 = gn2_cat['dn4000_err'][sig_4000break_indices_gn2]
    #dn4000_err_gs1 = gs1_cat['dn4000_err'][sig_4000break_indices_gs1]
    #dn4000_err_figs = np.concatenate((dn4000_err_gn1, dn4000_err_gn2, dn4000_err_gs1))
    #dn4000_err_figs_plot = dn4000_err_figs[np.where(dn4000_figs <= 3)[0]] 

    #ax.plot(redshift_pears_plot, dn4000_pears_plot, 'o', markersize=2, color='k', markeredgecolor='k')
    #ax.plot(redshift_figs_plot, dn4000_figs_plot, 'o', markersize=2, color='b', markeredgecolor='b')

    ax.errorbar(redshift_pears_plot, d4000_pears_plot, yerr=d4000_err_pears_plot,\
     fmt='.', color='k', markeredgecolor='k', capsize=0, markersize=4, elinewidth=0.25)
    #ax.errorbar(redshift_figs_plot, dn4000_figs_plot, yerr=dn4000_err_figs_plot,\
    # fmt='.', color='b', markeredgecolor='b', capsize=0, markersize=7, elinewidth=0.5)
    #ax.plot(redshift_sdss_plot, dn4000_sdss_plot, '.', markersize=2, color='slategray')
    #ax.plot(redshift_shels_plot, dn4000_shels_plot, '.', markersize=2, color='seagreen')

    ax.axhline(y=1, linewidth=1, linestyle='--', color='r', zorder=10)

    """
    # read in max break val at redshift text file and plot
    max_break_at_z_solar = np.genfromtxt(massive_galaxies_dir + 'max_break_at_redshift_solar.txt', dtype=None, names=True)
    ax.plot(max_break_at_z_solar['redshift'], max_break_at_z_solar['max_break'], '-', color='seagreen')
    max_break_at_z_highest_metals = np.genfromtxt(massive_galaxies_dir + 'max_break_at_redshift_highest_metals.txt', dtype=None, names=True)
    ax.plot(max_break_at_z_highest_metals['redshift'], max_break_at_z_highest_metals['max_break'], '-', color='darkgoldenrod')

    # labels
    #figslabelbox = TextArea("FIGS", textprops=dict(color='blue', size=12))
    #anc_figslabelbox = AnchoredOffsetbox(loc=2, child=figslabelbox, pad=0.0, frameon=False,\
    #                                     bbox_to_anchor=(0.85, 0.97),\
    #                                     bbox_transform=ax.transAxes, borderpad=0.0)
    #ax.add_artist(anc_figslabelbox)

    pearslabelbox = TextArea("PEARS", textprops=dict(color='black', size=12))
    anc_pearslabelbox = AnchoredOffsetbox(loc=2, child=pearslabelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.85, 0.92),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_pearslabelbox)

    #sdsslabelbox = TextArea("SDSS", textprops=dict(color='slategray', size=12))
    #anc_sdsslabelbox = AnchoredOffsetbox(loc=2, child=sdsslabelbox, pad=0.0, frameon=False,\
    #                                     bbox_to_anchor=(0.85, 0.87),\
    #                                     bbox_transform=ax.transAxes, borderpad=0.0)
    #ax.add_artist(anc_sdsslabelbox)

    #shelslabelbox = TextArea("SHELS", textprops=dict(color='seagreen', size=12))
    #anc_shelslabelbox = AnchoredOffsetbox(loc=2, child=shelslabelbox, pad=0.0, frameon=False,\
    #                                     bbox_to_anchor=(0.85, 0.82),\
    #                                     bbox_transform=ax.transAxes, borderpad=0.0)
    #ax.add_artist(anc_shelslabelbox)

    # metallicity labels
    # for both these labels
    # had to use unicode character for long dash because matplotlib doesn't like \text{} inside math mode in a string
    solarmetals_labelbox = TextArea(ur"\u2014" + r"$\mathrm{Z}_\odot$", textprops=dict(color='seagreen', size=12))
    anc_solarmetals_labelbox = AnchoredOffsetbox(loc=2, child=solarmetals_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.85, 0.87),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_solarmetals_labelbox)

    highestmetals_labelbox = TextArea(ur"\u2014" + r"$2.5\, \mathrm{Z}_\odot$", textprops=dict(color='darkgoldenrod', size=12))
    anc_highestmetals_labelbox = AnchoredOffsetbox(loc=2, child=highestmetals_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.85, 0.82),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_highestmetals_labelbox)
    """

    # labels and minor ticks
    ax.set_xlabel(r'$\mathrm{Redshift}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{D}(4000)$', fontsize=15)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    #ax.set_ylim(0,4)

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
