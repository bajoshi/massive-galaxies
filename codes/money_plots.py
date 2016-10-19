from __future__ import division

import numpy as np
from astropy.io import fits

import sys
import os
import datetime
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/fits_comp_spectra/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
new_codes_dir = home + "/Desktop/FIGS/new_codes/"

sys.path.append(massive_galaxies_dir + 'codes/')
import cosmology_calculator as cc

def makefig(xlab, ylab):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlab, size=15)
    ax.set_ylabel(ylab, size=15)

    return fig, ax

if __name__ == '__main__':

    alldata = np.load(massive_galaxies_dir + "best_fit_params.npy")

    best_age_bc03 = alldata[0]
    best_tau_bc03 = alldata[1]
    best_mass_wht_age_bc03 = alldata[2]
    best_form_redshift_bc03 = alldata[3]
    best_alpha_bc03 = alldata[4]

    best_age_fsps = alldata[5]
    best_tau_fsps = alldata[6]
    best_mass_wht_age_fsps = alldata[7]
    best_form_redshift_fsps = alldata[8]
    best_alpha_fsps = alldata[9]

    stellarmass_arr = alldata[10]

    # remove z_formation = 1100
    form_z_high_idx_bc03 = np.where(best_form_redshift_bc03 > 20)[0]
    form_z_high_idx_fsps = np.where(best_form_redshift_fsps > 20)[0] 

    best_form_redshift_bc03[form_z_high_idx_bc03] = np.nan
    best_form_redshift_fsps[form_z_high_idx_fsps] = np.nan

    # average SF time
    best_avg_sf_time_bc03 = 10**best_age_bc03 - 10**best_mass_wht_age_bc03
    best_avg_sf_time_fsps = 10**best_age_fsps - 10**best_mass_wht_age_fsps

    # make figures
    pdfname = massive_figures_dir + 'final_plots.pdf'
    pdf = PdfPages(pdfname)

    # plot same quantity for both libraries

    ## Formation times
    #fig, ax = makefig(r'$\mathrm{Formation\ time,\ BC03}$', r'$\mathrm{Formation\ time,\ FSPS}$')

    #ax.plot(best_age_bc03, best_age_fsps, 'o', color='k', markersize=2, markeredgecolor=None, label='')

    #plt.show()
    #plt.close()

    ## Tau
    #fig, ax = makefig(r'$\tau_{\mathrm{BC03}}$', r'$\tau_{\mathrm{FSPS}}$')

    #ax.plot(best_tau_bc03, best_tau_fsps, 'o', color='k', markersize=2, markeredgecolor=None, label='')

    #plt.show()
    #plt.close()

    ## formation redshifts
    #fig, ax = makefig(r'$z^{\mathrm{BC03}}_{\mathrm{formation}}$', r'$z^{\mathrm{FSPS}}_{\mathrm{formation}}$')

    #ax.plot(best_form_redshift_bc03, best_form_redshift_fsps, 'o', color='k', markersize=2, markeredgecolor=None, label='')

    #plt.show()
    #plt.close()

    ## mass weighted ages
    #fig, ax = makefig(r'${{\left<t\right>}_M}^{\mathrm{BC03}}$', r'${{\left<t\right>}_M}^{\mathrm{FSPS}}$')

    #ax.plot(best_mass_wht_age_bc03, best_mass_wht_age_fsps, 'o', color='k', markersize=2, markeredgecolor=None, label='')

    #plt.show()
    #plt.close()

    # --------------------------------- #
    # other plots 
    # formation time vs tau
    fig, ax = makefig(r'$\tau$', r'$\mathrm{log(Formation\ time)}$')

    ax.plot(best_tau_bc03, best_age_bc03, 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(best_tau_fsps, best_age_fsps, 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)

    ax.set_xscale('log')
    
    pdf.savefig(bbox_inches='tight')
    #plt.show()
    plt.close()

    # formation time vs mass weighted age
    fig, ax = makefig(r'${{\left<t\right>}_M}$', r'$\mathrm{log(Formation\ time)}$')

    ax.plot(best_mass_wht_age_bc03, best_age_bc03, 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(best_mass_wht_age_fsps, best_age_fsps, 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)
    
    pdf.savefig(bbox_inches='tight')
    #plt.show()
    plt.close()

    # formation time vs stellar mass
    fig, ax = makefig(r'$\mathrm{log\left(\frac{M_*}{M_\odot}\right)}$', r'$\mathrm{log(Formation\ time)}$')

    ax.plot(stellarmass_arr, best_age_bc03, 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(stellarmass_arr, best_age_fsps, 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)
    
    pdf.savefig(bbox_inches='tight')
    #plt.show()
    plt.close()

    # formation time vs formation redshift
    fig, ax = makefig(r'$z_{\mathrm{formation}}$', r'$\mathrm{log(Formation\ time)}$')

    ax.plot(best_form_redshift_bc03, best_age_bc03, 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(best_form_redshift_fsps, best_age_fsps, 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)
    
    pdf.savefig(bbox_inches='tight')
    #plt.show()
    plt.close()

    # tau vs mass weighted age
    fig, ax = makefig(r'${{\left<t\right>}_M}$', r'$\tau$')

    ax.plot(best_mass_wht_age_bc03, best_tau_bc03, 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(best_mass_wht_age_fsps, best_tau_fsps, 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)

    ax.set_yscale('log')
    
    pdf.savefig(bbox_inches='tight')
    #plt.show()
    plt.close()

    # tau vs stellar mass
    fig, ax = makefig(r'$\mathrm{log\left(\frac{M_*}{M_\odot}\right)}$', r'$\tau$')

    ax.plot(stellarmass_arr, best_tau_bc03, 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(stellarmass_arr, best_tau_fsps, 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)

    ax.set_yscale('log')
    
    pdf.savefig(bbox_inches='tight')
    #plt.show()
    plt.close()

    # tau vs formation redshift
    fig, ax = makefig(r'$z_{\mathrm{formation}}$', r'$\tau$')

    ax.plot(best_form_redshift_bc03, best_tau_bc03, 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(best_form_redshift_fsps, best_tau_fsps, 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)

    ax.set_yscale('log')
    
    pdf.savefig(bbox_inches='tight')
    #plt.show()
    plt.close()

    # mass weighted age vs stellar mass
    fig, ax = makefig(r'$\mathrm{log\left(\frac{M_*}{M_\odot}\right)}$', r'${{\left<t\right>}_M}$')

    ax.plot(stellarmass_arr, best_mass_wht_age_bc03, 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(stellarmass_arr, best_mass_wht_age_fsps, 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)
    
    pdf.savefig(bbox_inches='tight')
    #plt.show()
    plt.close()

    # mass weighted age vs formation redshift
    fig, ax = makefig(r'$z_{\mathrm{formation}}$', r'${{\left<t\right>}_M}$')

    ax.plot(best_form_redshift_bc03, best_mass_wht_age_bc03, 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(best_form_redshift_fsps, best_mass_wht_age_fsps, 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)
    
    pdf.savefig(bbox_inches='tight')
    #plt.show()
    plt.close()

    # formation redshift vs stellar mass
    fig, ax = makefig(r'$\mathrm{log\left(\frac{M_*}{M_\odot}\right)}$', r'$z_{\mathrm{formation}}$')

    ax.plot(stellarmass_arr, best_form_redshift_bc03, 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(stellarmass_arr, best_form_redshift_fsps, 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)

    pdf.savefig(bbox_inches='tight')
    #plt.show() 
    plt.close()

    # avg sf time vs formation time
    fig, ax = makefig(r'$\mathrm{log(Formation\ time)}$', r'${{\left<t\right>}_{SF}}$')

    ax.plot(best_age_bc03, np.log10(best_avg_sf_time_bc03), 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(best_age_fsps, np.log10(best_avg_sf_time_fsps), 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)

    pdf.savefig(bbox_inches='tight')
    #plt.show() 
    plt.close()

    # avg sf time vs tau
    fig, ax = makefig(r'$\tau$', r'${{\left<t\right>}_{SF}}$')

    ax.plot(best_tau_bc03, np.log10(best_avg_sf_time_bc03), 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(best_tau_fsps, np.log10(best_avg_sf_time_fsps), 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.set_xscale('log')
    ax.legend(loc=0, numpoints=1)

    pdf.savefig(bbox_inches='tight')
    #plt.show() 
    plt.close()

    # avg sf time vs mass weighted age
    fig, ax = makefig(r'${{\left<t\right>}_M}$', r'${{\left<t\right>}_{SF}}$')

    ax.plot(best_mass_wht_age_bc03, np.log10(best_avg_sf_time_bc03), 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(best_mass_wht_age_fsps, np.log10(best_avg_sf_time_fsps), 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)

    pdf.savefig(bbox_inches='tight')
    #plt.show() 
    plt.close()

    # avg sf time vs stellar mass
    fig, ax = makefig(r'$\mathrm{log\left(\frac{M_*}{M_\odot}\right)}$', r'${{\left<t\right>}_{SF}}$')

    ax.plot(stellarmass_arr, np.log10(best_avg_sf_time_bc03), 'o', color='r', markersize=4, markeredgecolor=None, label='BC03')
    ax.plot(stellarmass_arr, np.log10(best_avg_sf_time_fsps), 'o', color='g', markersize=4, markeredgecolor=None, label='FSPS')

    ax.legend(loc=0, numpoints=1)

    pdf.savefig(bbox_inches='tight')
    #plt.show() 
    plt.close()

    # ------------------------------- #
    # Histograms #

    

    pdf.close()

