from __future__ import division

import numpy as np

import os
import sys
import time
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, AnchoredText

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
new_codes_dir = home + "/Desktop/FIGS/new_codes/"

sys.path.append(stacking_analysis_dir + 'codes/')
sys.path.append(massive_galaxies_dir)
sys.path.append(massive_galaxies_dir + 'codes/')
import grid_coadd as gd
import matching as mt
import mag_hist as mh

# modify rc Params
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.sans-serif"] = ["Computer Modern Sans"]
#mpl.rcParams["text.usetex"] = True  
# this above line will add like 15-20 seconds to your run time but some times its necessary
# in here I need it to get teh \text to work
mpl.rcParams["text.latex.preamble"] = [r'\usepackage{amsmath}']
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"

def plot_for_comparison():

    # plot for comparison
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(lam_em_specz, flam_em_specz, '-', color='royalblue')
    ax.plot(lam_em_grismz, flam_em_grismz, '-', color='red')
    ax.plot(lam_em_photz, flam_em_photz, '-', color='green')
    
    # label all important quanitities
    # id and field label
    id_labelbox = TextArea(current_field + "  " + str(current_id), textprops=dict(color='k', size=10))
    anc_id_labelbox = AnchoredOffsetbox(loc=2, child=id_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.05, 0.98),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_id_labelbox)
    
    # spec z label
    spec_z_labelbox = TextArea(r"$z_{\mathrm{spec}} = $" + str("{:.3}".format(current_specz)), textprops=dict(color='k', size=10))
    anc_spec_z_labelbox = AnchoredOffsetbox(loc=2, child=spec_z_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.05, 0.93),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_spec_z_labelbox)
    
    # spec z quality and source label
    spec_z_info_labelbox = TextArea(r"$\mathrm{z_{spec}\ quality} = $" + str(current_specz_qual) + ", Source=" + str(current_specz_source), textprops=dict(color='k', size=10))
    anc_spec_z_info_labelbox = AnchoredOffsetbox(loc=2, child=spec_z_info_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.25, 0.93),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_spec_z_info_labelbox)
    
    # grism z label
    grism_z_labelbox = TextArea(r"$z_{\mathrm{grism}} = $" + str("{:.3}".format(current_grismz)) + r"$\pm$" + str("{:.3}".format(current_grismz_err)), textprops=dict(color='k', size=10))
    anc_grism_z_labelbox = AnchoredOffsetbox(loc=2, child=grism_z_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.05, 0.88),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_grism_z_labelbox)
    
    # phot z label
    phot_z_labelbox = TextArea(r"$z_{\mathrm{phot}} = $" + str("{:.3}".format(current_photz)), textprops=dict(color='k', size=10))
    anc_phot_z_labelbox = AnchoredOffsetbox(loc=2, child=phot_z_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.05, 0.83),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_phot_z_labelbox)
    
    # netsig label
    netsig_labelbox = TextArea(r"$N_s = $" + str("{:.3}".format(netsig_chosen_specz)) + "," + r"$N_g = $" + str("{:.3}".format(netsig_chosen_grismz)) +\
     "," + r"$N_p = $" + str("{:.3}".format(netsig_chosen_photz)) + "," + r"$N_{corr} = $" + str(netsig_corr),\
     textprops=dict(color='k', size=10))
    anc_netsig_labelbox = AnchoredOffsetbox(loc=2, child=netsig_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.05, 0.78),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_netsig_labelbox)
    
    # i magnitude label
    imag_labelbox = TextArea(r"$i = $" + str(imag) + r"$\ \mathrm{AB\ mag}$", textprops=dict(color='k', size=10))
    anc_imag_labelbox = AnchoredOffsetbox(loc=2, child=imag_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.05, 0.73),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_imag_labelbox)

    plt.show()
    del fig, ax

    return None

def get_pears_catalogs(refined=True):

    if refined:
        # read pears refined z cat
        pears_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-N.txt',\
         dtype=None, names=True, skip_header=1)
        pears_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-S.txt',\
         dtype=None, names=True, skip_header=1)

        print "Total galaxies in refined grism-z sample:", len(pears_cat_n) + len(pears_cat_s)

    else:
        # match with break catalogs instead of refined
        pears_cat_n = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/pears_4000break_catalog_GOODS-N.txt',\
         dtype=None, names=True, skip_header=1)
        pears_cat_s = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/pears_4000break_catalog_GOODS-S.txt',\
         dtype=None, names=True, skip_header=3)

        print "Total galaxies in break catalog sample:", len(pears_cat_n) + len(pears_cat_s)

    return pears_cat_n, pears_cat_s

def plot_z_comparison(lam_em_specz, flam_em_specz, current_specz, lam_em_grismz, flam_em_grismz, current_grismz):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(lam_em_specz, flam_em_specz, '--', color='k', lw=1)
    ax.plot(lam_em_grismz, flam_em_grismz, '--', color='r', lw=1)

    plt.show()

    return None

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # read nimish's specz catalog
    cdfs_cat = np.genfromtxt(massive_galaxies_dir + 'cdfs_specz_0117.txt', dtype=None, names=['ra','dec','z_spec','z_qual','catname','duplicate'], skip_header=13)
    goods_n_cat = np.genfromtxt(massive_galaxies_dir + 'goods_n_specz_0117.txt', dtype=None, names=['ra','dec','z_spec','z_qual','catname','duplicate'], skip_header=13)

    # read in pears catalogs
    pears_cat_n, pears_cat_s = get_pears_catalogs(refined=True)

    # match spec catalog with refined cat
    # create arrays
    specz_n_ra = goods_n_cat['ra']
    specz_n_dec = goods_n_cat['dec']

    specz_s_ra = cdfs_cat['ra']
    specz_s_dec = cdfs_cat['dec']

    pears_n_ra = pears_cat_n['ra']
    pears_n_dec = pears_cat_n['dec']
    pears_n_id = pears_cat_n['pearsid']

    pears_s_ra = pears_cat_s['ra']
    pears_s_dec = pears_cat_s['dec']
    pears_s_id = pears_cat_s['pearsid']

    # run matching code
    # north
    deltaRA, deltaDEC, specz_n_ra_matches, specz_n_dec_matches, pears_n_ra_matches, pears_n_dec_matches, specz_n_ind, pears_n_ind, num_single_matches = \
    mt.match(specz_n_ra, specz_n_dec, pears_n_ra, pears_n_dec, lim=0.1*1/3600)

    print "There were", num_single_matches, "single matches found out of", len(pears_cat_n), "objects in the PEARS NORTH catalog with spec z catalog from N. Hathi."
    mt.plot_diff(deltaRA, deltaDEC, name='pears_specz_goodsn')

    # south
    deltaRA, deltaDEC, specz_s_ra_matches, specz_s_dec_matches, pears_s_ra_matches, pears_s_dec_matches, specz_s_ind, pears_s_ind, num_single_matches = \
    mt.match(specz_s_ra, specz_s_dec, pears_s_ra, pears_s_dec, lim=0.1*1/3600)

    print "There were", num_single_matches, "single matches found out of", len(pears_cat_s), "objects in the PEARS SOUTH catalog with spec z catalog from N. Hathi."
    mt.plot_diff(deltaRA, deltaDEC, name='pears_specz_goodss')

    # plots
    z_spec = np.concatenate((goods_n_cat['z_spec'][specz_n_ind], cdfs_cat['z_spec'][specz_s_ind]), axis=0)
    z_grism = np.concatenate((pears_cat_n['new_z'][pears_n_ind], pears_cat_s['new_z'][pears_s_ind]), axis=0)
    z_phot = np.concatenate((pears_cat_n['old_z'][pears_n_ind], pears_cat_s['old_z'][pears_s_ind]), axis=0)
    z_grism_std = np.concatenate((pears_cat_n['new_z_err'][pears_n_ind], pears_cat_s['new_z_err'][pears_s_ind]), axis=0)

    # loop through the problem cases
    # read master catalog to get magnitude
    pears_master_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag', 'netsig_corr'], usecols=(0,1,2,3,6))
    pears_master_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag', 'netsig_corr'], usecols=(0,1,2,3,6))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_master_ncat['dec'] = pears_master_ncat['dec'] - dec_offset_goodsn_v19

    # set up arrays for looping
    all_pears_ref_cats = [pears_cat_n, pears_cat_s]
    all_spec_cats = [goods_n_cat, cdfs_cat]

    z_spec_plot = []
    z_grism_plot = []
    z_phot_plot = []
    z_grism_std_plot = []
    imag_plot = []
    spec_z_source_list = []

    specz_sample_ids = []
    specz_sample_field = []
    specz_sample_z = []

    skipped = 0
    weird = 0
    catcount = 0
    spec_count = 0
    for cat in all_spec_cats:  # dummy loop variable. I know I could use it but I like the clarity of the current way better.
        
        if catcount == 0:
            spec_cat = goods_n_cat
            spec_ind = specz_n_ind
            pears_cat = pears_cat_n
            pears_ind = pears_n_ind
            #print spec_cat[spec_ind]
            #sys.exit(0)

        elif catcount == 1:
            spec_cat = cdfs_cat
            spec_ind = specz_s_ind
            pears_cat = pears_cat_s
            pears_ind = pears_s_ind

        print "In field", catcount, "with", len(spec_cat['z_spec'][spec_ind]), "objects."

        for i in range(len(spec_cat['z_spec'][spec_ind])):
    
            current_id = pears_cat['pearsid'][pears_ind][i]
            current_grismz = pears_cat['new_z'][pears_ind][i]
            current_grismz_err = pears_cat['new_z_err'][pears_ind][i]
            current_photz = pears_cat['old_z'][pears_ind][i]
            current_field = pears_cat['field'][pears_ind][i]

            current_specz = spec_cat['z_spec'][spec_ind][i]
            current_specz_source = spec_cat['catname'][spec_ind][i]
            current_specz_qual = spec_cat['z_qual'][spec_ind][i]

            # get i band mag
            if current_field == 'GOODS-N':
                idarg = np.where(pears_master_ncat['id'] == current_id)[0]
                imag = pears_master_ncat['imag'][idarg]
                netsig_corr = pears_master_ncat['netsig_corr'][idarg]
            elif current_field == 'GOODS-S':
                idarg = np.where(pears_master_scat['id'] == current_id)[0]
                imag = pears_master_scat['imag'][idarg]
                netsig_corr = pears_master_scat['netsig_corr'][idarg]

            if current_specz_qual == 'A' or current_specz_qual == '4':
                if current_specz_source == '3D_HST':
                    continue
                else:
                    spec_count += 1
                    print "\n", "At id:", current_id, "in", current_field,
                    print "Corrected NetSig:", netsig_corr, "  i-band mag:", imag
                    print "Spec-z is", current_specz, "from", current_specz_source, "with quality", current_specz_qual,
                    print "Photo-z is", pears_cat['old_z'][pears_ind][i]

            """
            #if abs(spec_cat['z_spec'][spec_ind][i] - pears_cat['new_z'][pears_ind][i]) >= 0.03:

            #if (current_grismz > 1.235) or (current_grismz < 0.6):
            #    continue
            # skip if grism z is not within range
            # this condition probalby shouldn't be here
            # i hsould be checking if the spec_z is in range
            # if spec_z is in range then I need to find why 
            # the grism z finding fails.
    
            # other spec z quality cuts
            if (current_specz_qual != 'A'):
                #if (current_specz_qual == "C") or (current_specz_qual == "D") or (current_specz_qual == "Z"):
                skipped += 1
                continue

            try:
                if (int(current_specz_qual) < 4):
                    skipped += 1
                    continue
            except ValueError as e:
                pass
    
            if current_specz_source == "3D_HST":
                skipped += 1
                continue
    
            print "\n", "At id:", current_id, "in", current_field
            print "Spec-z is", current_specz, "from", current_specz_source, "with quality", current_specz_qual
            print "Photo-z is", pears_cat['old_z'][pears_ind][i]
            print "Corrected NetSig:", netsig_corr, "  i-band mag:", imag

            spec_z_source_list.append(current_specz_source)
            z_spec_plot.append(spec_cat['z_spec'][spec_ind][i])
            z_grism_plot.append(pears_cat['new_z'][pears_ind][i])
            z_phot_plot.append(pears_cat['old_z'][pears_ind][i])
            z_grism_std_plot.append(pears_cat['new_z_err'][pears_ind][i])
            imag_plot.append(imag)

            specz_sample_field.append(current_field)
            specz_sample_ids.append(current_id)
            specz_sample_z.append(current_specz)

            # find which galaxies have large (z_spec - z_grism)/(1+z_spec)
            #if abs((current_specz - current_grismz)/(1 + current_specz)) > 0.05:
            #    weird += 1
            #    print "large diff between spec and grism z", current_grismz, current_specz, current_id, current_field, current_photz
            #    lam_em_specz, flam_em_specz, ferr_specz, specname_specz, pa_forlsf_specz, netsig_chosen_specz = gd.fileprep(current_id, current_specz, current_field, apply_smoothing=True, width=1.5, kernel_type='gauss')
            #    lam_em_grismz, flam_em_grismz, ferr_grismz, specname_grismz, pa_forlsf_grismz, netsig_chosen_grismz = gd.fileprep(current_id, current_grismz, current_field, apply_smoothing=True, width=1.5, kernel_type='gauss')
            #    #lam_em_photz, flam_em_photz, ferr_photz, specname_photz, pa_forlsf_photz, netsig_chosen_photz = gd.fileprep(current_id, current_photz, current_field, apply_smoothing=True, width=1.5, kernel_type='gauss')
            #    current_d4000 = pears_cat['d4000'][pears_ind][i]
            #    current_err_frac = np.sum(abs(ferr_specz)) / np.sum(abs(flam_em_specz))
            #    print "D(4000), Netsig, and Overall error fraction is", current_d4000, netsig_chosen_specz, current_err_frac, '\n'
            #    #plot_z_comparison(lam_em_specz, flam_em_specz, current_specz, lam_em_grismz, flam_em_grismz, current_grismz)

            #else:
            #    if current_specz_source == "3D_HST":
            #        skipped += 1
            #        continue
            #    spec_z_source_list.append(current_specz_source)
            #    z_spec_plot.append(spec_cat['z_spec'][spec_ind][i])
            #    z_grism_plot.append(pears_cat['new_z'][pears_ind][i])
            #    z_phot_plot.append(pears_cat['old_z'][pears_ind][i])
            #    z_grism_std_plot.append(pears_cat['new_z_err'][pears_ind][i])
            #    imag_plot.append(imag)
            """

        catcount += 1

    print "Total number in high quality specz sample:", spec_count
    sys.exit(0)

    print skipped, "galaxies were skipped due to bad spectroscopic z quality."
    print weird, "galaxies have (z_spec - z_grism) >= 0.03."
    print len(z_spec_plot), "galaxies in spectroscopic comparison sample."

    print np.unique(spec_z_source_list)

    # convert to numpy arrays for operations 
    z_spec_plot = np.asarray(z_spec_plot)
    z_grism_plot = np.asarray(z_grism_plot)
    z_phot_plot = np.asarray(z_phot_plot)
    z_grism_std_plot = np.asarray(z_grism_std_plot)
    imag_plot = np.asarray(imag_plot)
    specz_sample_ids = np.asarray(specz_sample_ids)
    specz_sample_field = np.asarray(specz_sample_field)
    specz_sample_z = np.asarray(specz_sample_z)

    # save the comparison sample ids
    np.save(massive_galaxies_dir + 'specz_sample_field.npy', specz_sample_field)
    np.save(massive_galaxies_dir + 'specz_sample_ids.npy', specz_sample_ids)
    np.save(massive_galaxies_dir + 'specz_sample_z.npy', specz_sample_z)

    print len(imag_plot), len(z_grism_plot)

    threepercent_acc_grism = np.where((abs(z_spec_plot - z_grism_plot) / (1 + z_spec_plot)) <= 0.03)[0]
    onepercent_acc_grism = np.where((abs(z_spec_plot - z_grism_plot) / (1 + z_spec_plot)) <= 0.01)[0]
    print len(threepercent_acc_grism), "galaxies in specz sample with grism-z accuracy better than 3%"
    print len(onepercent_acc_grism), "galaxies in specz sample with grism-z accuracy better than 1%"

    threepercent_acc_photo = np.where((abs(z_spec_plot - z_phot_plot) / (1 + z_spec_plot)) <= 0.03)[0]
    onepercent_acc_photo = np.where((abs(z_spec_plot - z_phot_plot) / (1 + z_spec_plot)) <= 0.01)[0]
    print len(threepercent_acc_photo), "galaxies in specz sample with photo-z accuracy better than 3%"
    print len(onepercent_acc_photo), "galaxies in specz sample with photo-z accuracy better than 1%"

    # z_grism vs z_phot vs z_spec
    gs = gridspec.GridSpec(15,32)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=20.0, hspace=0.00)

    fig_gs = plt.figure()
    ax1 = fig_gs.add_subplot(gs[:10,:10])
    ax2 = fig_gs.add_subplot(gs[10:,:10])
    ax3 = fig_gs.add_subplot(gs[:10,11:21])
    ax4 = fig_gs.add_subplot(gs[10:,11:21])
    ax5 = fig_gs.add_subplot(gs[:10,22:])
    ax6 = fig_gs.add_subplot(gs[10:,22:])

    # ------------------------------------------------------------------------------------------------- #
    # first panel # z_spec vs z_grism
    ax1.plot(z_spec_plot, z_grism_plot, 'o', markersize=2.0, color='k', markeredgecolor='k')
    ax1.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r')

    ax1.set_xlim(0.6, 1.24)
    ax1.set_ylim(0.6, 1.24)

    ax1.set_ylabel(r'$z_\mathrm{g}$', fontsize=12, labelpad=1)

    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels(['', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2'], fontsize=8, rotation=45)

    ax1.minorticks_on()
    ax1.tick_params('both', width=1, length=3, which='minor')
    ax1.tick_params('both', width=1, length=4.7, which='major')
    ax1.grid(True)

    # residuals for first panel
    ax2.plot(z_spec_plot, (z_spec_plot - z_grism_plot)/(1+z_spec_plot), 'o', markersize=2.0, color='k', markeredgecolor='k')
    ax2.axhline(y=0, linestyle='--', color='r')

    ax2.set_xlim(0.6, 1.24)
    ax2.set_ylim(-0.2, 0.2)

    ax2.xaxis.set_ticklabels(['', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2'], fontsize=8, rotation=45)
    ax2.yaxis.set_ticklabels(['-0.2', '-0.15', '-0.1', '-0.05', '0.0', '0.05', '0.1', '0.15', ''], fontsize=8, rotation=45)

    ax2.set_xlabel(r'$z_\mathrm{s}$', fontsize=12)
    ax2.set_ylabel(r'$(z_\mathrm{s} - z_\mathrm{g})/(1+z_\mathrm{s})$', fontsize=12, labelpad=-2)

    ax2.minorticks_on()
    ax2.tick_params('both', width=1, length=3, which='minor')
    ax2.tick_params('both', width=1, length=4.7, which='major')
    ax2.grid(True)

    # ------------------------------------------------------------------------------------------------- #
    # second panel # z_spec vs z_phot
    ax3.plot(z_spec_plot, z_phot_plot, 'o', markersize=2.0, color='k', markeredgecolor='k')
    ax3.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r')

    ax3.set_xlim(0.6, 1.24)
    ax3.set_ylim(0.6, 1.24)

    ax3.set_ylabel(r'$z_\mathrm{p}$', fontsize=12, labelpad=-2)

    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])

    ax3.minorticks_on()
    ax3.tick_params('both', width=1, length=3, which='minor')
    ax3.tick_params('both', width=1, length=4.7, which='major')
    ax3.grid(True)

    # residuals for second panel
    ax4.plot(z_spec_plot, (z_spec_plot - z_phot_plot)/(1+z_spec_plot), 'o', markersize=2.0, color='k', markeredgecolor='k')
    ax4.axhline(y=0, linestyle='--', color='r')

    ax4.set_xlim(0.6, 1.24)
    ax4.set_ylim(-0.2, 0.2)

    ax4.xaxis.set_ticklabels(['', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2'], fontsize=8, rotation=45)
    ax4.yaxis.set_ticklabels([])

    ax4.set_xlabel(r'$z_\mathrm{s}$', fontsize=12)
    ax4.set_ylabel(r'$(z_\mathrm{s} - z_\mathrm{p})/(1+z_\mathrm{s})$', fontsize=12, labelpad=-2)

    ax4.minorticks_on()
    ax4.tick_params('both', width=1, length=3, which='minor')
    ax4.tick_params('both', width=1, length=4.7, which='major')
    ax4.grid(True)

    # ------------------------------------------------------------------------------------------------- #
    # third panel # z_grism vs z_phot
    ax5.plot(z_grism_plot, z_phot_plot, 'o', markersize=2.0, color='k', markeredgecolor='k')
    ax5.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r')

    ax5.set_xlim(0.6, 1.24)
    ax5.set_ylim(0.6, 1.24)

    ax5.set_ylabel(r'$z_\mathrm{p}$', fontsize=12, labelpad=-2)

    ax5.xaxis.set_ticklabels([])
    ax5.yaxis.set_ticklabels([])

    ax5.minorticks_on()
    ax5.tick_params('both', width=1, length=3, which='minor')
    ax5.tick_params('both', width=1, length=4.7, which='major')
    ax5.grid(True)

    # residuals for third panel
    ax6.plot(z_grism_plot, (z_grism_plot - z_phot_plot)/(1+z_grism_plot), 'o', markersize=2.0, color='k', markeredgecolor='k')
    ax6.axhline(y=0, linestyle='--', color='r')

    ax6.set_xlim(0.6, 1.24)
    ax6.set_ylim(-0.2, 0.2)

    ax6.xaxis.set_ticklabels(['', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2'], fontsize=8, rotation=45)
    ax6.yaxis.set_ticklabels([])

    ax6.set_xlabel(r'$z_\mathrm{g}$', fontsize=12)
    ax6.set_ylabel(r'$(z_\mathrm{g} - z_\mathrm{p})/(1+z_\mathrm{g})$', fontsize=12, labelpad=-2)

    ax6.minorticks_on()
    ax6.tick_params('both', width=1, length=3, which='minor')
    ax6.tick_params('both', width=1, length=4.7, which='major')
    ax6.grid(True)

    fig_gs.savefig(massive_figures_dir + "zspec_comparison.eps", dpi=300, bbox_inches='tight')
    #plt.show()

    print "median abs(z_spec - z_grism)/ 1+z_spec", np.median(abs(z_spec_plot - z_grism_plot)/(1+z_spec_plot))
    print "median abs(z_spec - z_phot)/ 1+z_spec", np.median(abs(z_spec_plot - z_phot_plot)/(1+z_spec_plot))
    acc = abs(z_spec_plot - z_grism_plot)/(1+z_spec_plot)
    print "Number of catastrophic failures i.e. comparison with specz gives error greater than 10%", len(np.where(acc >= 0.1)[0])
    print "Number of galaxies with abs(zspec - zgrism)/(1 + zspec) less than or equal to 0.0001:", len(np.where(acc <= 0.001)[0])
    print "Number of galaxies with abs(zspec - zgrism)/(1 + zspec) less than or equal to 0.0003:", len(np.where(acc <= 0.003)[0])

    plt.cla()
    plt.clf()
    plt.close()

    # ---------------------------- histogram of residuals overlaid ---------------------------- #

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # define colors
    myblue = mh.rgb_to_hex(0, 100, 180)
    myred = mh.rgb_to_hex(214, 39, 40)  # tableau 20 red

    grism_resid_hist_arr = (z_spec_plot - z_grism_plot)/(1+z_spec_plot)
    photz_resid_hist_arr = (z_spec_plot - z_phot_plot)/(1+z_spec_plot)

    ax.hist(photz_resid_hist_arr, 15, range=[-0.06,0.06], color=myred, alpha=0.75, zorder=10)
    ax.hist(grism_resid_hist_arr, 15, range=[-0.06,0.06], color=myblue, alpha=0.6, zorder=10)
    # this plot really needs an alpha channel
    # otherwise you wont see that the photo-z histogram under the grism-z histogram
    # is actually fatter around 0 whereas the grism-z histogram is thinner.

    ax.text(0.72, 0.97, r'$\mathrm{Grism{-}z}$' + '\n' + r'$\mathrm{residuals}$',\
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color=myblue, size=10)
    ax.text(0.835, 0.96, r'$\mathrm{\equiv \frac{z_s - z_g}{1 + z_s}}$',\
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color=myblue, size=14)

    ax.text(0.72, 0.87, r'$\mathrm{Photo{-}z}$' + '\n' + r'$\mathrm{residuals}$',\
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color=myred, size=10)
    ax.text(0.835, 0.86, r'$\mathrm{\equiv \frac{z_s - z_p}{1 + z_s}}$',\
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color=myred, size=14)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True, alpha=0.5)

    fig.savefig(massive_figures_dir + 'residual_histogram.png', dpi=300, bbox_inches='tight')
    # has to be png NOT eps. see comment above on alpha channel requirement.

    plt.cla()
    plt.clf()
    plt.close()

    # -------- plot of fraction of galaxies with specified error limit vs magnitude -------- #

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(imag_plot[threepercent_acc_grism], 10)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.hist(imag_plot[onepercent_acc_grism], 10)

    plt.show()

    sys.exit(0)

