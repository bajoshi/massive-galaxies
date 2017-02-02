from __future__ import division

import numpy as np

import os
import sys
import time
import datetime

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
import grid_coadd as gd
import matching as mt

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # read nimish's specz catalog
    cdfs_cat = np.genfromtxt(massive_galaxies_dir + 'cdfs_specz_0117.txt', dtype=None, names=['ra','dec','z_spec','z_qual','catname','duplicate'], skip_header=13)
    goods_n_cat = np.genfromtxt(massive_galaxies_dir + 'goods_n_specz_0117.txt', dtype=None, names=['ra','dec','z_spec','z_qual','catname','duplicate'], skip_header=13)

    # read pears refined z cat
    pears_cat_n = np.genfromtxt(home + '/Desktop/pears_refined_4000break_catalog_GOODS-N.txt',\
     dtype=None, names=True, skip_header=1)
    pears_cat_s = np.genfromtxt(home + '/Desktop/pears_refined_4000break_catalog_GOODS-S.txt',\
     dtype=None, names=True, skip_header=1)
    
    # match with break catalogs instead of refined
    #pears_cat_n = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/pears_4000break_catalog_GOODS-N.txt',\
    # dtype=None, names=True, skip_header=1)
    #pears_cat_s = np.genfromtxt(home + '/Desktop/FIGS/massive-galaxies/pears_4000break_catalog_GOODS-S.txt',\
    # dtype=None, names=True, skip_header=3)

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

    skipped = 0
    weird = 0
    catcount = 0
    for cat in all_spec_cats:  # dummy loop variable. I know I could use it but I like the clarity of the current way better.
        
        if catcount == 0:
            spec_cat = goods_n_cat
            spec_ind = specz_n_ind
            pears_cat = pears_cat_n
            pears_ind = pears_n_ind

        elif catcount == 1:
            spec_cat = cdfs_cat
            spec_ind = specz_s_ind
            pears_cat = pears_cat_s
            pears_ind = pears_s_ind

        print len(spec_cat['z_spec'][spec_ind])

        for i in range(len(spec_cat['z_spec'][spec_ind])):

            if abs(spec_cat['z_spec'][spec_ind][i] - pears_cat['new_z'][pears_ind][i]) >= 0.03:
    
                current_id = pears_cat['pearsid'][pears_ind][i]
                current_grismz = pears_cat['new_z'][pears_ind][i]
                current_grismz_err = pears_cat['new_z_err'][pears_ind][i]
                current_photz = pears_cat['old_z'][pears_ind][i]
                current_field = pears_cat['field'][pears_ind][i]

                current_specz = spec_cat['z_spec'][spec_ind][i]
                current_specz_source = spec_cat['catname'][spec_ind][i]
                current_specz_qual = spec_cat['z_qual'][spec_ind][i]
    
                if (current_specz_qual != "A"):
                    skipped += 1
                    continue

                try:
                    if (int(current_specz_qual) < 3):
                        skipped += 1
                        continue
                except ValueError as e:
                    pass
    
                if current_specz_source == "3D_HST":
                    skipped += 1
                    continue
    
                #lam_em_specz, flam_em_specz, ferr_specz, specname_specz, pa_forlsf_specz, netsig_chosen_specz = gd.fileprep(current_id, current_specz, current_field, apply_smoothing=True, width=1.5, kernel_type='gauss')
                #lam_em_grismz, flam_em_grismz, ferr_grismz, specname_grismz, pa_forlsf_grismz, netsig_chosen_grismz = gd.fileprep(current_id, current_grismz, current_field, apply_smoothing=True, width=1.5, kernel_type='gauss')
                #lam_em_photz, flam_em_photz, ferr_photz, specname_photz, pa_forlsf_photz, netsig_chosen_photz = gd.fileprep(current_id, current_photz, current_field, apply_smoothing=True, width=1.5, kernel_type='gauss')
    
                # get i band mag
                if current_field == 'GOODS-N':
                    idarg = np.where(pears_master_ncat['id'] == current_id)[0]
                    imag = pears_master_ncat['imag'][idarg]
                    netsig_corr = pears_master_ncat['netsig_corr'][idarg]
                elif current_field == 'GOODS-S':
                    idarg = np.where(pears_master_scat['id'] == current_id)[0]
                    imag = pears_master_scat['imag'][idarg]
                    netsig_corr = pears_master_scat['netsig_corr'][idarg]
    
                #print current_specz_source  #, netsig_corr, netsig_chosen_specz, imag
                weird += 1
                z_spec_plot.append(spec_cat['z_spec'][spec_ind][i])
                z_grism_plot.append(pears_cat['new_z'][pears_ind][i])
                z_phot_plot.append(pears_cat['old_z'][pears_ind][i])
                z_grism_std_plot.append(pears_cat['new_z_err'][pears_ind][i])

                """
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
                """

            else:
                if current_specz_source == "3D_HST":
                    skipped += 1
                    continue
                #print current_specz_source
                z_spec_plot.append(spec_cat['z_spec'][spec_ind][i])
                z_grism_plot.append(pears_cat['new_z'][pears_ind][i])
                z_phot_plot.append(pears_cat['old_z'][pears_ind][i])
                z_grism_std_plot.append(pears_cat['new_z_err'][pears_ind][i])

        catcount += 1

    print skipped, "galaxies were skipped due to bad spectroscopic z quality."
    print weird, "galaxies have (z_spec - z_grism) >= 0.03."
    print len(z_spec_plot), "galaxies in spectroscopic comparison sample."

    # convert to numpy arrays for operations 
    z_spec_plot = np.asarray(z_spec_plot)
    z_grism_plot = np.asarray(z_grism_plot)
    z_phot_plot = np.asarray(z_phot_plot)
    z_grism_std_plot = np.asarray(z_grism_std_plot)

    # z_grism vs z_phot vs z_spec
    gs = gridspec.GridSpec(15,30)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=20.0, hspace=0.00)

    fig_gs = plt.figure()
    ax1 = fig_gs.add_subplot(gs[:10,:10])
    ax2 = fig_gs.add_subplot(gs[10:,:10])
    ax3 = fig_gs.add_subplot(gs[:10,10:20])
    ax4 = fig_gs.add_subplot(gs[10:,10:20])
    ax5 = fig_gs.add_subplot(gs[:10,20:])
    ax6 = fig_gs.add_subplot(gs[10:,20:])

    # ------------------------------------------------------------------------------------------------- #
    # first panel # z_spec vs z_grism
    ax1.plot(z_spec_plot, z_grism_plot, 'o', markersize=1.5, color='k', markeredgecolor='k')
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
    ax2.plot(z_spec_plot, (z_spec_plot - z_grism_plot)/(1+z_spec_plot), 'o', markersize=1.5, color='k', markeredgecolor='k')
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
    ax3.plot(z_spec_plot, z_phot_plot, 'o', markersize=1.5, color='k', markeredgecolor='k')
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
    ax4.plot(z_spec_plot, (z_spec_plot - z_phot_plot)/(1+z_spec_plot), 'o', markersize=1.5, color='k', markeredgecolor='k')
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
    ax5.plot(z_grism_plot, z_phot_plot, 'o', markersize=1.5, color='k', markeredgecolor='k')
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
    ax6.plot(z_grism_plot, (z_grism_plot - z_phot_plot)/(1+z_grism_plot), 'o', markersize=1.5, color='k', markeredgecolor='k')
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

    fig_gs.savefig(massive_figures_dir + "zspec_comparison.eps", dpi=150, bbox_inches='tight')

    # ------------------------------------------------------------------------------------------------- #
    # plot of z_spec - z_grism vs sigma_z_grism
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot((z_spec_plot - z_grism_plot)**2, z_grism_std_plot, 'o', markersize=1.5, color='k', markeredgecolor='k')
    ax.axhline(y=0, linestyle='--', color='r')

    ax.set_xlabel(r'$(z_\mathrm{s} - z_\mathrm{g})^2$', fontsize=12)
    ax.set_ylabel(r'$\mathrm{\sigma_{z_{g}}}$', fontsize=12)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    fig.savefig(massive_figures_dir + "zspec_vs_zgrism_err.eps", dpi=150, bbox_inches='tight')

    #plt.show()

    # ------------------------------------------------------------------------------------------------- #
    # histogram of delta_z/1+z

    fig = plt.figure()
    ax = fig.add_subplot(111)

    hist_arr = (z_spec_plot - z_grism_plot)/(1+z_spec_plot)

    ax.hist(hist_arr, 20, alpha=0.6)

    #plt.show()

    sys.exit(0)

