from __future__ import division

import numpy as np
from astropy.stats import mad_std

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
spz_results_dir = massive_figures_dir + 'spz_run_jan2019/'

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import mocksim_results as mr
import new_refine_grismz_gridsearch_parallel as ngp
import dn4000_catalog as dc

def get_plotting_arrays():

    # ---------------------------------------------------------------
    # Read in SPZ results 
    spz_id_list = np.load(spz_results_dir + 'spz_id_list.npy')
    spz_field_list = np.load(spz_results_dir + 'spz_field_list.npy')
    zminchi2 = np.load(spz_results_dir + 'spz_zgrism_list.npy')
    zspec = np.load(spz_results_dir + 'spz_zspec_list.npy')

    # ---------------------------------------------------------------
    # Read in results from earlier photoz run
    zphot = np.load(massive_figures_dir + 'my_photoz_list.npy')
    # Also need ID and field from photoz run to match
    photoz_id_list = np.load(massive_figures_dir + 'my_photoz_id_list.npy')
    photoz_field_list = np.load(massive_figures_dir + 'my_photoz_field_list.npy')

    # ---------------------------------------------------------------
    # You will also have to read in the spectroscopic matched files to get specz quality 
    specz_goodsn = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-N.txt', dtype=None, names=True)
    specz_goodss = np.genfromtxt(massive_galaxies_dir + 'specz_comparison_sample_GOODS-S.txt', dtype=None, names=True)

    # ---------------------------------------------------------------
    # Read in emission line catalogs (Pirzkal 2013 and Straughn 2009)
    pirzkal2013 = np.genfromtxt(massive_galaxies_dir + 'pirzkal_2013_emline.cat', \
        dtype=None, names=['field', 'pearsid'], skip_header=30, usecols=(0,1))
    straughn2009 = np.genfromtxt(massive_galaxies_dir + 'straughn_2009_emline.cat', \
        dtype=None, names=['pearsid'], skip_header=46, usecols=(0))

    pirzkal2013_emline_ids = np.unique(pirzkal2013['pearsid'])
    straughn2009_emline_ids = np.unique(straughn2009['pearsid'])

    straughn2009_emline_ids = straughn2009_emline_ids.astype(np.int)

    # assign north and south ids
    pirzkal2013_north_emline_ids = []
    pirzkal2013_south_emline_ids = []

    for i in  range(len(pirzkal2013_emline_ids)):
        if 'n' == pirzkal2013_emline_ids[i][0]:
            pirzkal2013_north_emline_ids.append(pirzkal2013_emline_ids[i][1:])
        elif 's' == pirzkal2013_emline_ids[i][0]:
            pirzkal2013_south_emline_ids.append(pirzkal2013_emline_ids[i][1:])

    pirzkal2013_north_emline_ids = np.asarray(pirzkal2013_north_emline_ids, dtype=np.int)
    pirzkal2013_south_emline_ids = np.asarray(pirzkal2013_south_emline_ids, dtype=np.int)

    chuck_em_line_galaxies = False

    # ---------------------------------------------------------------
    # Now match SPZ run and photoz run and make arrays for plotting
    zspec_plot = []
    zphot_plot = []
    zspz_plot = []
    d4000_plot = []
    specz_qual_plot = []
    netsig_plot = []

    for i in range(len(zphot)):

        # Get photoz id and field and match to spz run
        photoz_id = photoz_id_list[i]
        photoz_field = photoz_field_list[i]

        # check if it is an emission line galaxy. If it is then skip
        # Be carreful changing this check. I think it is correct as it is.
        # I don think you can simply do:
        # if (int(current_id) in pirzkal2013_emline_ids) or (int(current_id) in straughn2009_emline_ids):
        #     continue
        # This can mix up north and south IDs because the IDs are not unique in north and south.
        if chuck_em_line_galaxies:
            if photoz_field == 'GOODS-N':
                if int(photoz_id) in pirzkal2013_north_emline_ids:
                    continue
            elif photoz_field == 'GOODS-S':
                if (int(photoz_id) in pirzkal2013_south_emline_ids) or (int(photoz_id) in straughn2009_emline_ids):
                    continue

        idx = np.where((spz_id_list == photoz_id) & (spz_field_list == photoz_field))[0]

        # If a match is found then first compute weighted 
        # spz and then append to plotting arrays
        if idx.size:
            # It should only find one exact match
            idx = int(idx)

            pz = np.load(spz_results_dir + str(photoz_field) + '_' + str(photoz_id) + '_pz.npy')
            zarr = np.load(spz_results_dir + str(photoz_field) + '_' + str(photoz_id) + '_z_arr.npy')
            zspz = np.sum(pz * zarr)

            if not np.isfinite(zspz):
                continue

            # Get specz quality
            if photoz_field == 'GOODS-N':
                specz_idx = int(np.where(specz_goodsn['pearsid'] == photoz_id)[0])
                current_specz_qual = specz_goodsn['specz_qual'][specz_idx]

            elif photoz_field == 'GOODS-S':
                specz_idx = int(np.where(specz_goodss['pearsid'] == photoz_id)[0])
                current_specz_qual = specz_goodss['specz_qual'][specz_idx]

            # ONly get other quantities and append if quality is reliable
            if (current_specz_qual == 'A') or (current_specz_qual == 'B') or \
            (current_specz_qual == '3') or (current_specz_qual == '4') or (current_specz_qual == 'Z'):

                # get netsig
                grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = \
                ngp.get_data(photoz_id, photoz_field)

                # Get d4000 at SPZ
                current_specz = zspec[idx]
                lam_em = grism_lam_obs / (1 + zspz)
                flam_em = grism_flam_obs * (1 + zspz)
                ferr_em = grism_ferr_obs * (1 + zspz)

                d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

                # Append
                zspec_plot.append(current_specz)
                zphot_plot.append(zphot[i])
                zspz_plot.append(zspz)
                specz_qual_plot.append(current_specz_qual)
                netsig_plot.append(netsig_chosen)
                d4000_plot.append(d4000)

    # Convert to numpy arrays and return
    zspec_plot = np.asarray(zspec_plot)
    zphot_plot = np.asarray(zphot_plot)
    zspz_plot = np.asarray(zspz_plot)
    d4000_plot = np.asarray(d4000_plot)
    specz_qual_plot = np.asarray(specz_qual_plot)
    netsig_plot = np.asarray(netsig_plot)

    return zspec_plot, zphot_plot, zspz_plot, d4000_plot, netsig_plot, specz_qual_plot

def main():

    zspec, zphot, zspz, d4000, netsig, specz_qual = get_plotting_arrays()

    # Apply D4000 cut
    d4000_low = 1.6
    d4000_high = 1.8
    d4000_idx = np.where((d4000 >= d4000_low) & (d4000 < d4000_high))[0]

    zspec = zspec[d4000_idx]
    zphot = zphot[d4000_idx]
    zspz = zspz[d4000_idx]
    d4000 = d4000[d4000_idx]
    netsig = netsig[d4000_idx]
    specz_qual = specz_qual[d4000_idx]

    # Make figure
    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(10,24)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0)

    ax1 = fig.add_subplot(gs[:7, :11])
    ax2 = fig.add_subplot(gs[7:, :11])

    ax3 = fig.add_subplot(gs[:7, 13:])
    ax4 = fig.add_subplot(gs[7:, 13:])

    # Get residuals 
    resid_zphot = (zspec - zphot) / (1 + zspec)
    resid_zspz = (zspec - zspz) / (1 + zspec)

    # Make sure they are finite
    valid_idx1 = np.where(np.isfinite(resid_zphot))[0]
    valid_idx2 = np.where(np.isfinite(resid_zspz))[0]

    # Remove catastrophic failures
    # i.e. only choose the valid ones
    catas_fail1 = np.where(resid_zphot < 0.1)[0]
    catas_fail2 = np.where(resid_zspz < 0.1)[0]

    outlier_frac_photoz = len(np.where(resid_zphot >= 0.1)[0]) / len(valid_idx1)
    outlier_frac_spz = len(np.where(resid_zspz >= 0.1)[0]) / len(valid_idx2)

    # apply cut on netsig
    netsig_thresh = 10
    valid_idx3 = np.where(netsig > netsig_thresh)[0]
    print len(valid_idx3), "out of", len(netsig), "galaxies pass NetSig cut of", netsig_thresh

    valid_idx = reduce(np.intersect1d, (valid_idx1, valid_idx2, catas_fail1, catas_fail2, valid_idx3))

    resid_zphot = resid_zphot[valid_idx]
    resid_zspz = resid_zspz[valid_idx]
    zspec = zspec[valid_idx]
    zphot = zphot[valid_idx]
    zspz = zspz[valid_idx]

    print "Number of galaxies in plot:", len(valid_idx)

    # Print info
    mean_zphot = np.mean(resid_zphot)
    std_zphot = np.std(resid_zphot)
    nmad_zphot = mad_std(resid_zphot)

    mean_zspz = np.mean(resid_zspz)
    std_zspz = np.std(resid_zspz)
    nmad_zspz = mad_std(resid_zspz)

    print "Mean, std dev, and Sigma_NMAD for residuals for Photo-z:", \
    "{:.2e}".format(mean_zphot), "{:.2e}".format(std_zphot), "{:.2e}".format(nmad_zphot)
    print "Mean, std dev, and Sigma_NMAD for residuals for SPZs:", \
    "{:.2e}".format(mean_zspz), "{:.2e}".format(std_zspz), "{:.2e}".format(nmad_zspz)
    print "Outlier fraction for photoz:", outlier_frac_photoz
    print "Outlier fraction for SPZ:", outlier_frac_spz

    ax1.plot(zspec, zphot, 'o', markersize=5, color='k', markeredgecolor='k')
    ax2.plot(zspec, resid_zphot, 'o', markersize=5, color='k', markeredgecolor='k')

    ax3.plot(zspec, zspz, 'o', markersize=5, color='k', markeredgecolor='k')
    ax4.plot(zspec, resid_zspz, 'o', markersize=5, color='k', markeredgecolor='k')

    # Limits
    ax1.set_xlim(0.6, 1.24)
    ax1.set_ylim(0.6, 1.24)

    ax2.set_xlim(0.6, 1.24)
    ax2.set_ylim(-0.1, 0.1)

    ax3.set_xlim(0.6, 1.24)
    ax3.set_ylim(0.6, 1.24)

    ax4.set_xlim(0.6, 1.24)
    ax4.set_ylim(-0.1, 0.1)

    # Other lines on plot
    ax2.axhline(y=0.0, ls='--', color='gray')
    ax4.axhline(y=0.0, ls='--', color='gray')

    linearr = np.arange(0.5, 1.3, 0.001)
    ax1.plot(linearr, linearr, ls='--', color='darkblue')
    ax3.plot(linearr, linearr, ls='--', color='darkblue')

    ax2.axhline(y=mean_zphot, ls='-', color='darkblue')
    ax2.axhline(y=mean_zphot + nmad_zphot, ls='-', color='red')
    ax2.axhline(y=mean_zphot - nmad_zphot, ls='-', color='red')

    ax4.axhline(y=mean_zspz, ls='-', color='darkblue')
    ax4.axhline(y=mean_zspz + nmad_zspz, ls='-', color='red')
    ax4.axhline(y=mean_zspz - nmad_zspz, ls='-', color='red')

    # Text on figures
    # print D4000 range
    ax1.text(0.91, 1.1, "{:.1f}".format(d4000_low) + r"$\, \leq \mathrm{D4000} < \,$" + "{:.1f}".format(d4000_high), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=22)

    # print N, mean, nmad, outlier frac
    ax1.text(0.05, 0.97, r'$\mathrm{N = }$' + str(len(valid_idx)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=18)
    ax1.text(0.05, 0.89, r'${\left< \Delta \right>}_{\mathrm{Photo-z}} = $' + mr.convert_to_sci_not(mean_zphot), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=18)
    ax1.text(0.05, 0.81, r'$\mathrm{\sigma^{NMAD}_{Photo-z}} = $' + mr.convert_to_sci_not(nmad_zphot), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=18)
    #ax1.text(0.05, 0.73, r'$\mathrm{Outlier\ frac}$' + '\n' + r'$\mathrm{for\ Photo-z:}$' + "{:.3f}".format(outlier_frac_photoz), \
    #verticalalignment='top', horizontalalignment='left', \
    #transform=ax1.transAxes, color='k', size=18)

    ax3.text(0.05, 0.97, r'$\mathrm{N = }$' + str(len(valid_idx)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=18)
    ax3.text(0.05, 0.89, r'${\left< \Delta \right>}_{\mathrm{SPZ}} = $' + mr.convert_to_sci_not(mean_zspz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=18)
    ax3.text(0.05, 0.81, r'$\mathrm{\sigma^{NMAD}_{SPZ}} = $' + mr.convert_to_sci_not(nmad_zspz), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=18)
    #ax3.text(0.05, 0.73, r'$\mathrm{Outlier\ frac}$' + '\n' + r'$\mathrm{for\ SPZ:}$' + "{:.3f}".format(outlier_frac_spz), \
    #verticalalignment='top', horizontalalignment='left', \
    #transform=ax3.transAxes, color='k', size=18)

    # Make tick labels larger
    ax1.set_xticklabels(ax1.get_xticks().tolist(), size=14)
    ax1.set_yticklabels(ax1.get_yticks().tolist(), size=14)

    ax2.set_xticklabels(ax2.get_xticks().tolist(), size=14)
    ax2.set_yticklabels(ax2.get_yticks().tolist(), size=14)

    ax3.set_xticklabels(ax3.get_xticks().tolist(), size=14)
    ax3.set_yticklabels(ax3.get_yticks().tolist(), size=14)

    ax4.set_xticklabels(ax4.get_xticks().tolist(), size=14)
    ax4.set_yticklabels(ax4.get_yticks().tolist(), size=14)

    # Get rid of Xaxis tick labels on top subplot
    ax1.set_xticklabels([])
    ax3.set_xticklabels([])

    # Minor ticks
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    ax4.minorticks_on()

    # Axis labels
    ax1.set_ylabel(r'$\mathrm{z_{phot}}$', fontsize=20)
    ax2.set_xlabel(r'$\mathrm{z_{spec}}$', fontsize=20)
    ax2.set_ylabel(r'$\mathrm{(z_{spec} - z_{phot}) / (1+z_{spec})}$', fontsize=20)

    ax3.set_ylabel(r'$\mathrm{z_{spz}}$', fontsize=20)
    ax4.set_xlabel(r'$\mathrm{z_{spec}}$', fontsize=20)
    ax4.set_ylabel(r'$\mathrm{(z_{spec} - z_{spz}) / (1+z_{spec})}$', fontsize=20)

    # save figure and close
    fig.savefig(massive_figures_dir + 'spz_comp_photoz_' + str(d4000_low) + 'to' + str(d4000_high) + '.png', \
        dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    # --------------------------------------------------------------------------- # 
    # ---------------------- Make NetSig histogram as well ---------------------- #
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Get total bins using FD method and plot histogram
    iqr = np.std(netsig[valid_idx], dtype=np.float64)
    binsize = 2*iqr*np.power(len(netsig[valid_idx]),-1/3)
    totalbins = np.floor((max(netsig[valid_idx]) - min(netsig[valid_idx]))/binsize)
    print "Total bins in NetSig histogram:", int(totalbins)
    ax.hist(netsig[valid_idx], totalbins, histtype='step')

    # labels
    ax.set_xlabel('Net Spectral Significance', fontsize=20)
    ax.set_ylabel('N', fontsize=20)

    # minorticks
    ax.minorticks_on()

    # save figure and close
    fig.savefig(massive_figures_dir + 'netsig_hist_' + str(d4000_low) + 'to' + str(d4000_high) + '.png', \
        dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit()