from __future__ import division

import numpy as np
from astropy.stats import mad_std
from scipy.integrate import simps
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"
zg_results_dir = massive_figures_dir + 'grismz_run_jan2019/'
zp_results_dir = massive_figures_dir + 'photoz_run_jan2019/'

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import mocksim_results as mr
from new_refine_grismz_gridsearch_parallel import get_data
import dn4000_catalog as dc

def get_plotting_arrays():

    # ------------------------------- Get catalog for final sample ------------------------------- #
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample.txt', dtype=None, names=True)

    # ------------- Firstlight -------------
    id_arr_fl = np.load(zp_results_dir + 'firstlight_id_arr.npy')
    field_arr_fl = np.load(zp_results_dir + 'firstlight_field_arr.npy')
    zs_arr_fl = np.load(zp_results_dir + 'firstlight_zs_arr.npy')

    zg_minchi2_fl = np.load(zg_results_dir + 'firstlight_zg_minchi2_arr.npy')
    zg_arr_fl = zg_minchi2_fl

    # Length checks
    assert len(id_arr_fl) == len(field_arr_fl)
    assert len(id_arr_fl) == len(zs_arr_fl)
    assert len(id_arr_fl) == len(zg_minchi2_fl)

    # ------------- Jet -------------
    id_arr_jt = np.load(zp_results_dir + 'jet_id_arr.npy')
    field_arr_jt = np.load(zp_results_dir + 'jet_field_arr.npy')
    zs_arr_jt = np.load(zp_results_dir + 'jet_zs_arr.npy')

    zg_minchi2_jt = np.load(zg_results_dir + 'jet_zg_minchi2_arr.npy')
    zg_arr_jt = zg_minchi2_jt

    # Length checks
    assert len(id_arr_jt) == len(field_arr_jt)
    assert len(id_arr_jt) == len(zs_arr_jt)
    assert len(id_arr_jt) == len(zg_minchi2_jt)

    # ----- Concatenate -----
    # check for any accidental overlaps
    # I'm just doing an explicit for loop because I need to compare both ID and field
    min_len = len(id_arr_fl)  # Since firstlight went through fewer galaxies
    common_indices_jt = []
    for j in range(min_len):

        id_to_search = id_arr_fl[j]
        field_to_search = field_arr_fl[j]

        """
        Note the order of the two if statements below. 
        if (id_to_search in id_arr_jt) and (field_to_search in field_arr_jt)
        WILL NOT WORK! 
        This is because the second condition there is always true.
        """
        if (id_to_search in id_arr_jt):
            jt_idx = int(np.where(id_arr_jt == id_to_search)[0])
            if (field_arr_jt[jt_idx] == field_to_search):
                common_indices_jt.append(jt_idx)

    # Delete common galaxies from Jet arrays 
    # ONly delete from one of the set of arrays since you want these galaxies included only once
    # ----- Jet arrays with common galaxies deleted ----- 
    id_arr_jt = np.delete(id_arr_jt, common_indices_jt, axis=None)
    field_arr_jt = np.delete(field_arr_jt, common_indices_jt, axis=None)
    zs_arr_jt = np.delete(zs_arr_jt, common_indices_jt, axis=None)
    zg_arr_jt = np.delete(zg_arr_jt, common_indices_jt, axis=None)

    # I need to concatenate these arrays for hte purposes of looping and appending in hte loop below
    all_ids = np.concatenate((id_arr_fl, id_arr_jt))
    all_fields = np.concatenate((field_arr_fl, field_arr_jt))
    zs = np.concatenate((zs_arr_fl, zs_arr_jt))
    zg_array = np.concatenate((zg_arr_fl, zg_arr_jt))

    all_ids_list = []
    all_fields_list = []
    zs_list = []
    zg_list = []
    all_d4000_list = []
    all_d4000_err_list = []
    eazy_redshift_list = []

    # Read in EAZY Grism only catalogs # THese are in the Desktop folder on firstlight
    eazy_ncat = np.genfromtxt(home + '/Desktop/EAZY_grism_only/N/OUTPUT/goods_n_grism_only_0.01.zout', dtype=None, names=True)
    eazy_scat = np.genfromtxt(home + '/Desktop/EAZY_grism_only/S/OUTPUT/goods_s_grism_only_0.01.zout', dtype=None, names=True)

    # Now get D4000 and EAZY redshift
    for i in range(len(all_ids)):
        current_id = all_ids[i]
        current_field = all_fields[i]

        sample_idx = np.where(final_sample['pearsid'] == current_id)[0]
        if not sample_idx.size:
            continue
        if len(sample_idx) == 1:
            if final_sample['field'][sample_idx] != current_field:
                continue

        # Check z_spec quality
        # Make sure to get the spec_z qual for the exact match
        if len(sample_idx) == 2:
            sample_idx_nested = int(np.where(final_sample['field'][sample_idx] == current_field)[0])
            current_specz_qual = final_sample['zspec_qual'][sample_idx[sample_idx_nested]]
        else:
            current_specz_qual = final_sample['zspec_qual'][int(sample_idx)]

        if current_specz_qual == '1' or current_specz_qual == 'C':
            #print "Skipping", current_field, current_id, "due to spec-z quality:", current_specz_qual
            continue

        # Get data
        grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = get_data(current_id, current_field)

        if return_code == 0:
            print current_id, current_field
            print "Return code should not have been 0. Exiting."
            sys.exit(0)

        # Get D4000 at specz
        current_specz = zs[i]
        lam_em = grism_lam_obs / (1 + current_specz)
        flam_em = grism_flam_obs * (1 + current_specz)
        ferr_em = grism_ferr_obs * (1 + current_specz)

        d4000, d4000_err = dc.get_d4000(lam_em, flam_em, ferr_em)

        # Match and get EAZY redshift
        if current_field == 'GOODS-N':
            id_idx = np.where(eazy_ncat['id'] == current_id)[0]
            if len(id_idx) == 1:
                id_idx = int(id_idx)
                eazy_redshift_list.append(eazy_ncat['z_2'][id_idx])
            elif len(id_idx) == 0:  # i.e., no match found
                continue
        elif current_field == 'GOODS-S':
            id_idx = np.where(eazy_scat['id'] == current_id)[0]
            if len(id_idx) == 1:
                id_idx = int(id_idx)
                eazy_redshift_list.append(eazy_scat['z_2'][id_idx])
            elif len(id_idx) == 0:  # i.e., no match found
                continue

        # Append all arrays
        all_ids_list.append(current_id)
        all_fields_list.append(current_field)
        zs_list.append(current_specz)
        zg_list.append(zg_array[i])
        all_d4000_list.append(d4000)
        all_d4000_err_list.append(d4000_err)

    # Convert to numpy array
    all_ids = np.array(all_ids_list)
    all_fields = np.array(all_fields_list)
    zs = np.array(zs_list)
    zg = np.array(zg_list)
    all_d4000 = np.array(all_d4000_list)
    all_d4000_err = np.array(all_d4000_err_list)
    eazy_redshift = np.array(eazy_redshift_list)

    return all_ids, all_fields, zs, zg, eazy_redshift, all_d4000, all_d4000_err

def line_func(x, slope, intercept):
    return slope*x + intercept

def plot_eazy_grismz_comparison(resid_eazy, resid_zg, eazy_z, zs_for_eazy, zg, zs_for_zg, \
    mean_eazy, nmad_eazy, mean_zg, nmad_zg, d4000_low, d4000_high, \
    outlier_idx_eazy, outlier_idx_zg, outlier_frac_eazy, outlier_frac_zg):

    # Define figure
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(10,28)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:7, :12])
    ax2 = fig.add_subplot(gs[7:, :12])

    ax3 = fig.add_subplot(gs[:7, 16:])
    ax4 = fig.add_subplot(gs[7:, 16:])

    # Plot stuff
    ax1.plot(zs_for_eazy, eazy_z, 'o', markersize=2, color='k', markeredgecolor='k')
    ax1.scatter(zs_for_eazy[outlier_idx_eazy], eazy_z[outlier_idx_eazy], s=20, facecolor='white', edgecolors='gray', zorder=5)
    ax2.plot(zs_for_eazy, resid_eazy, 'o', markersize=2, color='k', markeredgecolor='k')
    ax2.scatter(zs_for_eazy[outlier_idx_eazy], resid_eazy[outlier_idx_eazy], s=20, facecolor='white', edgecolors='gray', zorder=5)

    ax3.plot(zs_for_zg, zg, 'o', markersize=2, color='k', markeredgecolor='k')
    ax3.scatter(zs_for_zg[outlier_idx_zg], zg[outlier_idx_zg], s=20, facecolor='white', edgecolors='gray', zorder=5)
    ax4.plot(zs_for_zg, resid_zg, 'o', markersize=2, color='k', markeredgecolor='k')
    ax4.scatter(zs_for_zg[outlier_idx_zg], resid_zg[outlier_idx_zg], s=20, facecolor='white', edgecolors='gray', zorder=5)

    # Limits
    #ax1.set_xlim(0.6, 1.24)
    #ax1.set_ylim(0.6, 1.24)

    #ax2.set_xlim(0.6, 1.24)
    #ax2.set_ylim(-0.15, 0.15)

    #ax3.set_xlim(0.6, 1.24)
    #ax3.set_ylim(0.6, 1.24)

    #ax4.set_xlim(0.6, 1.24)
    #ax4.set_ylim(-0.15, 0.15)

    # Other lines on plot
    ax2.axhline(y=0.0, ls='-', color='gray')
    ax4.axhline(y=0.0, ls='-', color='gray')

    # do the fit with scipy
    popt_eazy, pcov_eazy = curve_fit(line_func, zs_for_eazy, eazy_z, p0=[1.0, 0.6])
    popt_zg, pcov_zg = curve_fit(line_func, zs_for_zg, zg, p0=[1.0, 0.6])

    # plot line fit
    x_plot = np.arange(0.2,1.5,0.01)

    eazy_mean_line = line_func(x_plot, popt_eazy[0], popt_eazy[1])
    zg_mean_line = line_func(x_plot, popt_zg[0], popt_zg[1])

    ax1.plot(x_plot, x_plot, '-', color='gray')
    ax1.plot(x_plot, eazy_mean_line, '--', color='darkblue', lw=1)
    ax1.plot(x_plot, (1+nmad_eazy)*popt_eazy[0]*x_plot + nmad_eazy + popt_eazy[1], ls='--', color='red', lw=1)
    ax1.plot(x_plot, (1-nmad_eazy)*popt_eazy[0]*x_plot - nmad_eazy + popt_eazy[1], ls='--', color='red', lw=1)

    ax3.plot(x_plot, x_plot, '-', color='gray')
    ax3.plot(x_plot, zg_mean_line, '--', color='darkblue', lw=1)
    ax3.plot(x_plot, (1+nmad_zg)*popt_zg[0]*x_plot + nmad_zg + popt_zg[1], ls='--', color='red', lw=1)
    ax3.plot(x_plot, (1-nmad_zg)*popt_zg[0]*x_plot - nmad_zg + popt_zg[1], ls='--', color='red', lw=1)

    ax2.axhline(y=mean_eazy, ls='-', color='darkblue')
    ax2.axhline(y=mean_eazy + nmad_eazy, ls='-', color='red')
    ax2.axhline(y=mean_eazy - nmad_eazy, ls='-', color='red')

    ax4.axhline(y=mean_zg, ls='-', color='darkblue')
    ax4.axhline(y=mean_zg + nmad_zg, ls='-', color='red')
    ax4.axhline(y=mean_zg - nmad_zg, ls='-', color='red')

    # Make tick labels larger
    ax1.set_xticklabels(ax1.get_xticks().tolist(), size=10)
    ax1.set_yticklabels(ax1.get_yticks().tolist(), size=10)

    ax2.set_xticklabels(ax2.get_xticks().tolist(), size=10)
    ax2.set_yticklabels(ax2.get_yticks().tolist(), size=10)

    ax3.set_xticklabels(ax3.get_xticks().tolist(), size=10)
    ax3.set_yticklabels(ax3.get_yticks().tolist(), size=10)

    ax4.set_xticklabels(ax4.get_xticks().tolist(), size=10)
    ax4.set_yticklabels(ax4.get_yticks().tolist(), size=10)

    # Get rid of Xaxis tick labels on top subplot
    ax1.set_xticklabels([])
    ax3.set_xticklabels([])

    # Minor ticks
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    ax4.minorticks_on()

    # Axis labels
    ax1.set_ylabel(r'$\mathrm{z_{eazy}}$', fontsize=13)
    ax2.set_xlabel(r'$\mathrm{z_{s}}$', fontsize=13)
    ax2.set_ylabel(r'$\mathrm{(z_{eazy} - z_{s}) / (1+z_{s})}$', fontsize=13)

    ax3.set_ylabel(r'$\mathrm{z_{g}}$', fontsize=13)
    ax4.set_xlabel(r'$\mathrm{z_{s}}$', fontsize=13)
    ax4.set_ylabel(r'$\mathrm{(z_{g} - z_{s}) / (1+z_{s})}$', fontsize=13)

    # print D4000 range
    ax1.text(0.45, 0.11, "{:.1f}".format(d4000_low) + r"$\, \leq \mathrm{D4000} < \,$" + "{:.1f}".format(d4000_high), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=15)

    # print N, mean, nmad, outlier frac
    ax1.text(0.05, 0.97, r'$\mathrm{N = }$' + str(len(resid_eazy)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)
    ax1.text(0.04, 0.89, r'${\left< \Delta \right>}_{\mathrm{EAZY-z}} = $' + mr.convert_to_sci_not(mean_eazy), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)
    ax1.text(0.05, 0.79, r'$\mathrm{\sigma^{NMAD}_{EAZY-z}} = $' + mr.convert_to_sci_not(nmad_eazy), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)
    ax1.text(0.05, 0.7, r'$\mathrm{Out\ frac\, =\, }$' + str("{:.2f}".format(outlier_frac_eazy)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax1.transAxes, color='k', size=12)

    ax3.text(0.05, 0.97, r'$\mathrm{N = }$' + str(len(resid_zg)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12)
    ax3.text(0.04, 0.89, r'${\left< \Delta \right>}_{\mathrm{Grism-z}} = $' + mr.convert_to_sci_not(mean_zg), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12)
    ax3.text(0.05, 0.79, r'$\mathrm{\sigma^{NMAD}_{Grism-z}} = $' + mr.convert_to_sci_not(nmad_zg), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12)
    ax3.text(0.05, 0.7, r'$\mathrm{Out\ frac\, =\, }$' + str("{:.2f}".format(outlier_frac_zg)), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax3.transAxes, color='k', size=12)

    # save figure and close
    fig.savefig(massive_figures_dir + 'eazy_grismz_comp_' + \
        str(d4000_low).replace('.','p') + 'to' + str(d4000_high).replace('.','p') + '.pdf', \
        dpi=300, bbox_inches='tight')

    return None

def main():

    ids, fields, zs, zg, eazy_z, d4000, d4000_err = get_plotting_arrays()

    # Just making sure that all returned arrays have the same length.
    # Essential since I'm doing "where" operations below.
    assert len(ids) == len(fields)
    assert len(ids) == len(zs)
    assert len(ids) == len(zg)
    assert len(ids) == len(eazy_z)
    assert len(ids) == len(d4000)
    assert len(ids) == len(d4000_err)

    # Cut on D4000
    d4000_low = 1.1
    d4000_high = 2.0
    d4000_idx = np.where((d4000 >= d4000_low) & (d4000 < d4000_high) & (d4000_err < 0.5))[0]

    print "\n", "D4000 range:   ", d4000_low, "<= D4000 <", d4000_high, "\n"
    print "Galaxies within D4000 range:", len(d4000_idx)

    # Apply D4000 index
    zs = zs[d4000_idx]
    zg = zg[d4000_idx]
    eazy_z = eazy_z[d4000_idx]

    # Residuals
    resid_zg = (zg - zs) / (1 + zs)
    resid_eazy = (eazy_z - zs) / (1 + zs)

    # Make sure they are finite
    valid_idx1 = np.where(np.isfinite(resid_eazy))[0]
    valid_idx2 = np.where(np.isfinite(resid_zg))[0]

    # Apply indices
    resid_eazy = resid_eazy[valid_idx1]
    eazy_z = eazy_z[valid_idx1]
    zs_for_eazy = zs[valid_idx1]

    resid_zg = resid_zg[valid_idx2]
    zg = zg[valid_idx2]
    zs_for_zg = zs[valid_idx2]

    print "\n", "Number of galaxies in EAZY-z plot:", len(valid_idx1)
    print "Number of galaxies in Grism-z plot:", len(valid_idx2)

    # Print info
    mean_eazy = np.mean(resid_eazy)
    std_eazy = np.std(resid_eazy)
    nmad_eazy = mad_std(resid_eazy)

    mean_zg = np.mean(resid_zg)
    std_zg = np.std(resid_zg)
    nmad_zg = mad_std(resid_zg)

    print "\n", "Mean, std dev, and Sigma_NMAD for residuals for EAZY-z:", \
    "{:.3f}".format(mean_eazy), "{:.3f}".format(std_eazy), "{:.3f}".format(nmad_eazy)
    print "Mean, std dev, and Sigma_NMAD for residuals for Grism-z:", \
    "{:.3f}".format(mean_zg), "{:.3f}".format(std_zg), "{:.3f}".format(nmad_zg)

    # Compute catastrophic failures
    # i.e., How many galaxies are outside +-3-sigma given the sigma above?
    outlier_idx_eazy = np.where(abs(resid_eazy) > 3*nmad_eazy)[0]
    outlier_idx_zg = np.where(abs(resid_zg) > 3*nmad_zg)[0]

    outlier_frac_eazy = len(outlier_idx_eazy) / len(resid_eazy)
    outlier_frac_zg = len(outlier_idx_zg) / len(resid_zg)

    print "\n", "Outlier fraction for EAZY-z:", outlier_frac_eazy
    print "Outlier fraction for Grism-z:", outlier_frac_zg

    plot_eazy_grismz_comparison(resid_eazy, resid_zg, eazy_z, zs_for_eazy, zg, zs_for_zg, \
    mean_eazy, nmad_eazy, mean_zg, nmad_zg, d4000_low, d4000_high, \
    outlier_idx_eazy, outlier_idx_zg, outlier_frac_eazy, outlier_frac_zg)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)