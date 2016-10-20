# Matches PEARS and 3DHST

# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

sys.path.append(massive_galaxies_dir)
import matching as mt

def plot_spectrum(file, redshift, title, redshift_tag):
    z = redshift
    
    fitsfile = fits.open(file)
    fitsdata = fitsfile[1].data
    if len(fitsdata['FLUX'][13:101]) != 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, title=title)
        ax.set_xlabel('$\lambda\ [\AA]$')
        ax.set_ylabel('$F_{\lambda}\ [erg/cm^2/s/\AA]$')
        ax.axhline(y=0,linestyle='--')
        ax.set_xlim(6000,9600)
        ax.errorbar(fitsdata['LAMBDA'][13:101],fitsdata['FLUX'][13:101],yerr=fitsdata['FERROR'][13:101],color='black', linewidth=1)
        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')
        
        if oii * (1+z) > fitsdata['LAMBDA'][13]:
            #ax.axvline(x = oii * (1+z), linestyle='--', color='r', linewidth=1)
            flux_oii = fitsdata['FLUX'][np.argmin(abs(fitsdata['LAMBDA'] - (oii * (1+z))))]
            ax.annotate(r'$[OII]$', xy=(oii * (1+z) - 50, flux_oii + 0.2e-18), bbox=dict(boxstyle = 'round, pad=0.1', fc='b', alpha=0.2), fontsize=11)
    
        ax.axvline(x = 4000 * (1+z), linestyle='--', color='r', linewidth=1)
        flux_4000 = fitsdata['FLUX'][np.argmin(abs(fitsdata['LAMBDA'] - (4000 * (1+z))))]
        ax.annotate(r'$4000\AA$', xy=(4000 * (1+z) - 100, flux_4000 + 0.2e-18), bbox=dict(boxstyle = 'round, pad=0.1', fc='b', alpha=0.2), fontsize=11)

        #ax.axvline(x = caii_h * (1+z), linestyle='--', color='r', linewidth=1)
        #ax.axvline(x = caii_k * (1+z), linestyle='--', color='r', linewidth=1)
        flux_3950 = fitsdata['FLUX'][np.argmin(abs(fitsdata['LAMBDA'] - (3950 * (1+z))))]
        ax.annotate(r'$CaII\ H\ and\ K$', xy=(3950 * (1+z) - 150, flux_3950 - 0.4e-18), bbox=dict(boxstyle = 'round, pad=0.1', fc='b', alpha=0.2), fontsize=11)
        
        #ax.axvline(x = H_gamma * (1+z), linestyle='--', color='r', linewidth=1)
        #ax.axvline(x = H_beta * (1+z), linestyle='--', color='r', linewidth=1)
        flux_h_gamma = fitsdata['FLUX'][np.argmin(abs(fitsdata['LAMBDA'] - (H_gamma * (1+z))))]
        flux_h_beta = fitsdata['FLUX'][np.argmin(abs(fitsdata['LAMBDA'] - (H_beta * (1+z))))]
        ax.annotate(r'$H\gamma$', xy=(H_gamma * (1+z) - 50, flux_h_gamma - 0.2e-18), bbox=dict(boxstyle = 'round, pad=0.1', fc='b', alpha=0.2), fontsize=11)
        ax.annotate(r'$H\beta$', xy=(H_beta * (1+z) - 50, flux_h_beta - 0.2e-18), bbox=dict(boxstyle = 'round, pad=0.1', fc='b', alpha=0.2), fontsize=11)
        
        if len(fitsdata['LAMBDA']) >= 101:
            if oiii_5 * (1+z) < fitsdata['LAMBDA'][101]:
                #ax.axvline(x = oiii_4 * (1+z), linestyle='--', color='r', linewidth=1)
                #ax.axvline(x = oiii_5 * (1+z), linestyle='--', color='r', linewidth=1)
                flux_4983 = fitsdata['FLUX'][np.argmin(abs(fitsdata['LAMBDA'] - (4983 * (1+z))))]
                ax.annotate(r'$[OIII]\ doublet$', xy=(4983 * (1+z) - 150, flux_4983 - 0.3e-18), bbox=dict(boxstyle = 'round, pad=0.1', fc='b', alpha=0.2), fontsize=11)

        ax.annotate(redshift_tag + ' = ' + str(z), xy=(0.8,0.1), xycoords='axes fraction', bbox=dict(boxstyle = 'round, pad=0.25', fc='k', alpha=0.1), fontsize=12)
        fig.savefig(save_path+'Spectrum_'+ os.path.split(os.path.splitext(data_path + file)[0])[1], dpi=300)
        plt.close(fig)
    fitsfile.close()

    return None

def plot_delta_radec(closest_ra, closest_dec, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\Delta \alpha$')
    ax.set_ylabel(r'$\Delta \delta$')
    ax.plot(closest_ra, closest_dec, 'o', markeredgecolor='b')

    #ax.set_xlim(-0.1, 0.1)
    #ax.set_ylim(-0.1, 0.1)

    ax.axhline(y=0, linestyle='--', color='k')
    ax.axvline(x=0, linestyle='--', color='k')

    ax.minorticks_on()
    ax.tick_params('both', width=0.8, length=3, which='minor')
    ax.tick_params('both', width=0.8, length=4.7, which='major')
    #fig.savefig(home + "/Documents/4000break_check/matched_diff/" + name, dpi=300)
    plt.show()

    return None

def match_north(req_range, field, threed_hst_cat):
    closest_ra = []
    closest_dec = []
    pearsid_m = []
    threed_field = []
    threed_id_m = []
    pearsra_m = []
    pearsdec_m = []
    threed_ra_m = []
    threed_dec_m = []
    separation_arcsec_m = []
    threed_zphot = []
    threed_mass = []

    for i in range(len(req_range)):
        ang_dist = np.sqrt((threed_hst_cat[1].data['ra'][req_range[i]] - pears_ncat['ra'])**2 +\
                           (threed_hst_cat[1].data['dec'][req_range[i]] - pears_ncat['dec'])**2)
        match = np.argmin(ang_dist)
        if ang_dist[match] <= 0.1/3600:
            closest_ra.append((pears_ncat['ra'][match] - threed_hst_cat[1].data['ra'][req_range[i]]) * 3600)
            closest_dec.append((pears_ncat['dec'][match] - threed_hst_cat[1].data['dec'][req_range[i]]) * 3600)
            file = data_path + "h_pears_n_id" + str(pears_ncat['id'][match]) + ".fits"
            
            pearsid_m.append(pears_ncat['id'][match])
            threed_field.append(threed_hst_cat[1].data['field'][req_range[i]])
            threed_id_m.append(threed_hst_cat[1].data['id'][req_range[i]])
            pearsra_m.append(pears_ncat['ra'][match])
            pearsdec_m.append(pears_ncat['dec'][match])
            threed_ra_m.append(threed_hst_cat[1].data['ra'][req_range[i]])
            threed_dec_m.append(threed_hst_cat[1].data['dec'][req_range[i]])
            separation_arcsec_m.append(ang_dist[match] * 3600)
            threed_zphot.append(threed_hst_cat[1].data['z_peak'][req_range[i]])
            threed_mass.append(threed_hst_cat[1].data['lmass'][req_range[i]])
            
            """
            if threed_hst_cat[1].data['z_spec'][req_range[i]] != -1:
                plot_spectrum(file, threed_hst_cat[1].data['z_spec'][req_range[i]], 'n_' + str(pears_ncat['id'][match]), '$z_{spec}$')
            else:
                plot_spectrum(file, threed_hst_cat[1].data['z_peak'][req_range[i]], 'n_' + str(pears_ncat['id'][match]), '$z_{phot}$')
            """
    #plot_delta_radec(closest_ra, closest_dec, field)

    data = np.array(zip(pearsid_m, threed_field, threed_id_m, pearsra_m, pearsdec_m,\
                        threed_ra_m, threed_dec_m, separation_arcsec_m, threed_zphot, threed_mass),\
                    dtype=[('pearsid_m', int), ('threed_field', '|S7'), ('threed_id_m', int), ('pearsra_m', float), ('pearsdec_m', float),\
                           ('threed_ra_m', float), ('threed_dec_m', float), ('separation_arcsec_m', float),
                           ('threed_zphot', float), ('threed_mass', float)])

    np.savetxt(outdir + 'matches_' + field + '.txt', data,\
                fmt = ['%d', '%s', '%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.5f', '%.2f'], delimiter=' ',\
                header='The 3DHST ID is field specific!' + '\n' +\
                       'tolerance used for matches is 0.1"' + '\n' +\
                        'These are only those matches for which the redshift range is 0.558 < z < 1.317' + '\n' +\
                        'pearsid field threed_id pearsra pearsdec threedra threeddec separation_arcsec threed_zphot threed_mstellar')

    return None

def match_south(req_range, field, threed_hst_cat):
    closest_ra = []
    closest_dec = []
    pearsid_m = []
    threed_field = []
    threed_id_m = []
    pearsra_m = []
    pearsdec_m = []
    threed_ra_m = []
    threed_dec_m = []
    separation_arcsec_m = []
    threed_zphot = []
    threed_mass = []
    
    for i in range(len(req_range)):
        ang_dist = np.sqrt((threed_hst_cat[1].data['ra'][req_range[i]] - pears_scat['ra'])**2 +\
                           (threed_hst_cat[1].data['dec'][req_range[i]] - pears_scat['dec'])**2)
        match = np.argmin(ang_dist)
        if ang_dist[match] <= 0.1/3600:
            closest_ra.append((pears_scat['ra'][match] - threed_hst_cat[1].data['ra'][req_range[i]]) * 3600)
            closest_dec.append((pears_scat['dec'][match] - threed_hst_cat[1].data['dec'][req_range[i]]) * 3600)
            file = data_path + "h_pears_s_id" + str(pears_scat['id'][match]) + ".fits"
            
            pearsid_m.append(pears_scat['id'][match])
            threed_id_m.append(threed_hst_cat[1].data['id'][req_range[i]])
            threed_field.append(threed_hst_cat[1].data['field'][req_range[i]])
            pearsra_m.append(pears_scat['ra'][match])
            pearsdec_m.append(pears_scat['dec'][match])
            threed_ra_m.append(threed_hst_cat[1].data['ra'][req_range[i]])
            threed_dec_m.append(threed_hst_cat[1].data['dec'][req_range[i]])
            separation_arcsec_m.append(ang_dist[match] * 3600)
            threed_zphot.append(threed_hst_cat[1].data['z_peak'][req_range[i]])
            threed_mass.append(threed_hst_cat[1].data['lmass'][req_range[i]])
        
        """
            if threed_hst_cat[1].data['z_spec'][req_range[i]] != -1:
                plot_spectrum(file, threed_hst_cat[1].data['z_spec'][req_range[i]], 's_' + str(pears_scat['id'][match]), '$z_{spec}$')
            else:
                plot_spectrum(file, threed_hst_cat[1].data['z_peak'][req_range[i]], 's_' + str(pears_scat['id'][match]), '$z_{phot}$')
        """
    #plot_delta_radec(closest_ra, closest_dec, field)
    data = np.array(zip(pearsid_m, threed_field, threed_id_m, pearsra_m, pearsdec_m,\
                        threed_ra_m, threed_dec_m, separation_arcsec_m, threed_zphot, threed_mass),\
                    dtype=[('pearsid_m', int), ('threed_field', '|S7'), ('threed_id_m', int), ('pearsra_m', float), ('pearsdec_m', float),\
                            ('threed_ra_m', float), ('threed_dec_m', float), ('separation_arcsec_m', float),
                            ('threed_zphot', float), ('threed_mass', float)])
    
    np.savetxt(outdir + 'matches_' + field + '.txt', data,\
               fmt = ['%d', '%s', '%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.5f', '%.2f'], delimiter=' ',\
               header='The 3DHST ID is field specific!' + '\n' +\
               'tolerance used for matches is 0.1"' + '\n' +\
               'These are only those matches for which the redshift range is 0.6 < z < 1.235' + '\n' +\
               'pearsid field threed_id pearsra pearsdec threedra threeddec separation_arcsec threed_zphot threed_mstellar')

    return None

def read_3dhst_cats(mode='all'):

    # Read 3D-HST catalogs
    # Using v4.1 catalogs instead of the master v4.1.5 catalog
    threed_ncat = fits.open(home + '/Desktop/FIGS/new_codes/goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat.FITS')
    threed_scat = fits.open(home + '/Desktop/FIGS/new_codes/goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat.FITS')

    threed_v41_phot = fits.open(home + '/Desktop/FIGS/new_codes/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat.fits')

    if mode == 'all':
        return threed_ncat, threed_scat, threed_v41_phot
    elif mode == 'master':
        return threed_v41_phot
    elif mode == 'fields':
        return threed_ncat, threed_scat

if __name__ == '__main__':

    # Read 3dhst cats
    threed_ncat, threed_scat, threed_v41_phot = read_3dhst_cats()

    # Read PEARS cats
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS v2.0 readme
    pears_ncat['dec'] = pears_ncat['dec'] - dec_offset_goodsn_v19

    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # Match 3d-hst with PEARS
    z_low_pears = 0.558
    z_high_pears = 1.317

    # Coordinates of PEARS field's centers
    # I think I got these from the PEARS phase2 apt -- Check again.
    cdfs_udf_ra = 53.1625
    cdfs_udf_dec = -27.7914
    #
    cdfn1_ra = 189.1873
    cdfn1_dec = 62.2033
    #
    cdfn2_ra = 189.1898
    cdfn2_dec = 62.2548
    #
    cdfn3_ra = 189.3114
    cdfn3_dec = 62.2917
    #
    cdfn4_ra = 189.3741
    cdfn4_dec = 62.3199

    #
    cdfs1_ra = 53.17
    cdfs1_dec = -27.9017
    #
    cdfs2_ra = 53.1775
    cdfs2_dec = -27.842
    #
    cdfs3_ra = 53.12
    cdfs3_dec = -27.74
    #
    cdfs4_ra = 53.0667
    cdfs4_dec = -27.7092
    #
    cdfs_new_ra = 53.1632
    cdfs_new_dec = -27.9035

    # Required Ranges from 3d-hst

    req_range_cdfn1 = np.where((threed_v41_phot[1].data['z_peak'] >= z_low_pears) & (threed_v41_phot[1].data['z_peak'] <= z_high_pears) &\
                               (threed_v41_phot[1].data['field'] == 'GOODS-N') & (threed_v41_phot[1].data['ra'] >= cdfn1_ra - 101/3600) &\
                               (threed_v41_phot[1].data['ra'] <= cdfn1_ra + 101/3600) & (threed_v41_phot[1].data['dec'] >= cdfn1_dec - 101/3600) &\
                               (threed_v41_phot[1].data['dec'] <= cdfn1_dec + 101/3600) & (threed_v41_phot[1].data['use_phot'] == 1))[0]
    req_range_cdfn2 = np.where((threed_v41_phot[1].data['z_peak'] >= z_low_pears) & (threed_v41_phot[1].data['z_peak'] <= z_high_pears) &\
                               (threed_v41_phot[1].data['field'] == 'GOODS-N') & (threed_v41_phot[1].data['ra'] >= cdfn2_ra - 101/3600) &\
                               (threed_v41_phot[1].data['ra'] <= cdfn2_ra + 101/3600) & (threed_v41_phot[1].data['dec'] >= cdfn2_dec - 101/3600) &\
                               (threed_v41_phot[1].data['dec'] <= cdfn2_dec + 101/3600) & (threed_v41_phot[1].data['use_phot'] == 1))[0]
    req_range_cdfn3 = np.where((threed_v41_phot[1].data['z_peak'] >= z_low_pears) & (threed_v41_phot[1].data['z_peak'] <= z_high_pears) &\
                               (threed_v41_phot[1].data['field'] == 'GOODS-N') & (threed_v41_phot[1].data['ra'] >= cdfn3_ra - 101/3600) &\
                               (threed_v41_phot[1].data['ra'] <= cdfn3_ra + 101/3600) & (threed_v41_phot[1].data['dec'] >= cdfn3_dec - 101/3600) &\
                               (threed_v41_phot[1].data['dec'] <= cdfn3_dec + 101/3600) & (threed_v41_phot[1].data['use_phot'] == 1))[0]
    req_range_cdfn4 = np.where((threed_v41_phot[1].data['z_peak'] >= z_low_pears) & (threed_v41_phot[1].data['z_peak'] <= z_high_pears) &\
                               (threed_v41_phot[1].data['field'] == 'GOODS-N') & (threed_v41_phot[1].data['ra'] >= cdfn4_ra - 101/3600) &\
                               (threed_v41_phot[1].data['ra'] <= cdfn4_ra + 101/3600) & (threed_v41_phot[1].data['dec'] >= cdfn4_dec - 101/3600) &\
                               (threed_v41_phot[1].data['dec'] <= cdfn4_dec + 101/3600) & (threed_v41_phot[1].data['use_phot'] == 1))[0]

    req_range_cdfs1 = np.where((threed_v41_phot[1].data['z_peak'] >= z_low_pears) & (threed_v41_phot[1].data['z_peak'] <= z_high_pears) &\
                               (threed_v41_phot[1].data['field'] == 'GOODS-S') & (threed_v41_phot[1].data['ra'] >= cdfs1_ra - 101/3600) &\
                               (threed_v41_phot[1].data['ra'] <= cdfs1_ra + 101/3600) & (threed_v41_phot[1].data['dec'] >= cdfs1_dec - 101/3600) &\
                               (threed_v41_phot[1].data['dec'] <= cdfs1_dec + 101/3600) & (threed_v41_phot[1].data['use_phot'] == 1))[0]
    req_range_cdfs2 = np.where((threed_v41_phot[1].data['z_peak'] >= z_low_pears) & (threed_v41_phot[1].data['z_peak'] <= z_high_pears) &\
                               (threed_v41_phot[1].data['field'] == 'GOODS-S') & (threed_v41_phot[1].data['ra'] >= cdfs2_ra - 101/3600) &\
                               (threed_v41_phot[1].data['ra'] <= cdfs2_ra + 101/3600) & (threed_v41_phot[1].data['dec'] >= cdfs2_dec - 101/3600) &\
                               (threed_v41_phot[1].data['dec'] <= cdfs2_dec + 101/3600) & (threed_v41_phot[1].data['use_phot'] == 1))[0]
    req_range_cdfs3 = np.where((threed_v41_phot[1].data['z_peak'] >= z_low_pears) & (threed_v41_phot[1].data['z_peak'] <= z_high_pears) &\
                               (threed_v41_phot[1].data['field'] == 'GOODS-S') & (threed_v41_phot[1].data['ra'] >= cdfs3_ra - 101/3600) &\
                               (threed_v41_phot[1].data['ra'] <= cdfs3_ra + 101/3600) & (threed_v41_phot[1].data['dec'] >= cdfs3_dec - 101/3600) &\
                               (threed_v41_phot[1].data['dec'] <= cdfs3_dec + 101/3600) & (threed_v41_phot[1].data['use_phot'] == 1))[0]
    req_range_cdfs4 = np.where((threed_v41_phot[1].data['z_peak'] >= z_low_pears) & (threed_v41_phot[1].data['z_peak'] <= z_high_pears) &\
                               (threed_v41_phot[1].data['field'] == 'GOODS-S') & (threed_v41_phot[1].data['ra'] >= cdfs4_ra - 101/3600) &\
                               (threed_v41_phot[1].data['ra'] <= cdfs4_ra + 101/3600) & (threed_v41_phot[1].data['dec'] >= cdfs4_dec - 101/3600) &\
                               (threed_v41_phot[1].data['dec'] <= cdfs4_dec + 101/3600) & (threed_v41_phot[1].data['use_phot'] == 1))[0]

    req_range_cdfs_new = np.where((threed_v41_phot[1].data['z_peak'] >= z_low_pears) & (threed_v41_phot[1].data['z_peak'] <= z_high_pears) &\
                                  (threed_v41_phot[1].data['field'] == 'GOODS-S') & (threed_v41_phot[1].data['ra'] >= cdfs_new_ra - 101/3600) &\
                                  (threed_v41_phot[1].data['ra'] <= cdfs_new_ra + 101/3600) & (threed_v41_phot[1].data['dec'] >= cdfs_new_dec - 101/3600) &\
                                  (threed_v41_phot[1].data['dec'] <= cdfs_new_dec + 101/3600) & (threed_v41_phot[1].data['use_phot'] == 1))[0]
    req_range_cdfs_udf = np.where((threed_v41_phot[1].data['z_peak'] >= z_low_pears) & (threed_v41_phot[1].data['z_peak'] <= z_high_pears) &\
                             (threed_v41_phot[1].data['field'] == 'GOODS-S') & (threed_v41_phot[1].data['ra'] >= cdfs_udf_ra - 101/3600) &\
                             (threed_v41_phot[1].data['ra'] <= cdfs_udf_ra + 101/3600) & (threed_v41_phot[1].data['dec'] >= cdfs_udf_dec - 101/3600) &\
                             (threed_v41_phot[1].data['dec'] <= cdfs_udf_dec + 101/3600) & (threed_v41_phot[1].data['use_phot'] == 1))[0]

    # Common spectral features
    oii = 3726.7
    caii_h = 3968.5
    caii_k = 3933.7
    H_eps = 3970
    H_beta = 4861.3
    H_gamma = 4340.5
    oiii_4 = 4959
    oiii_5 = 5007

    outdir = home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/'

    match_north(req_range_cdfn1, "cdfn1", threed_v41_phot)
    match_north(req_range_cdfn2, "cdfn2", threed_v41_phot)
    match_north(req_range_cdfn3, "cdfn3", threed_v41_phot)
    match_north(req_range_cdfn4, "cdfn4", threed_v41_phot)

    match_south(req_range_cdfs1, "cdfs1", threed_v41_phot)
    match_south(req_range_cdfs2, "cdfs2", threed_v41_phot)
    match_south(req_range_cdfs3, "cdfs3", threed_v41_phot)
    match_south(req_range_cdfs4, "cdfs4", threed_v41_phot)
    match_south(req_range_cdfs_udf, "cdfs_udf", threed_v41_phot)
    match_south(req_range_cdfs_new, "cdfs_new", threed_v41_phot)

    sys.exit(0)
