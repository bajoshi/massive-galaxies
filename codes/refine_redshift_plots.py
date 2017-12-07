from __future__ import division

import numpy as np
from astropy.io import fits

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
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd

def z_grism_std_vs_imag(z_grism_std_plot, imag_plot, samp_str):

    # z_grism_std vs imag
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(imag_plot, z_grism_std_plot, 'o', markersize=1.5, color='k', markeredgecolor='k')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    ax.set_xlabel(r'$\mathrm{i\,[AB\ mag]}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{\sigma_{z_{grism}}}$', fontsize=15)  

    fig.savefig(massive_figures_dir + 'z_err_vs_imag_' + samp_str + '.eps', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

def z_grism_std_vs_netsig_corr(z_grism_std_plot, net_sig_corr_cat_plot, d4000_plot, samp_str):

    # z_grism_std vs corrected net sig in master catalog
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=1.0, hspace=0.0)

    fig = plt.figure()

    ax1 = fig.add_subplot(gs[:,:14])
    ax2 = fig.add_subplot(gs[:,14:])

    norm = plt.Normalize()
    colors = plt.cm.OrRd(norm(d4000_plot))
    ax1.scatter(np.log10(net_sig_corr_cat_plot), z_grism_std_plot, s=5, color=colors)

    cmap = mpl.cm.OrRd
    norm = mpl.colors.Normalize(vmin=np.min(d4000_plot), vmax=np.max(d4000_plot))
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('D(4000)')
    # for more details about plotting the colorbar see
    # http://matplotlib.org/examples/api/colorbar_only.html

    ax1.minorticks_on()
    ax1.tick_params('both', width=1, length=3, which='minor')
    ax1.tick_params('both', width=1, length=4.7, which='major')
    ax1.grid(True)

    ax1.set_xlabel(r'$\mathrm{log(Corrected\ Net\ Spectral\ Significance)}$', fontsize=15)
    ax1.set_ylabel(r'$\mathrm{\sigma_{z_{grism}}}$', fontsize=15)  

    fig.savefig(massive_figures_dir + 'z_err_vs_netsig_corr_' + samp_str + '.eps', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

def z_grism_std_vs_d4000(z_grism_std_plot, d4000_plot, samp_str):

    # z_grism_std vs d4000
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(d4000_plot, z_grism_std_plot, 'o', markersize=1.5, color='k', markeredgecolor='k')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    ax.set_xlabel(r'$\mathrm{D(4000)}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{\sigma_{z_{grism}}}$', fontsize=15)  

    ax.set_xlim(0.8, 2.5)

    fig.savefig(massive_figures_dir + 'z_err_vs_d4000_' + samp_str + '.eps', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

def z_grism_std_vs_redshift(z_grism_std_plot, redshift_plot, samp_str):

    # z_grism_std vs redshift
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(redshift_plot, z_grism_std_plot, 'o', markersize=1.5, color='k', markeredgecolor='k')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    ax.set_xlabel(r'$\mathrm{z}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{\sigma_{z_{grism}}}$', fontsize=15)  

    fig.savefig(massive_figures_dir + 'z_err_vs_redshift_' + samp_str + '.eps', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    return None

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # read catalog adn rename arrays for convenience
    pears_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-N.txt',\
     dtype=None, names=True, skip_header=1)
    pears_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_GOODS-S.txt',\
     dtype=None, names=True, skip_header=1)

    print "Number of objects in refined catalog for GOODS-N", len(pears_cat_n)
    print "Number of objects in refined catalog for GOODS-S", len(pears_cat_s)

    z_grism = np.concatenate((pears_cat_n['new_z'], pears_cat_s['new_z']), axis=0)
    z_phot = np.concatenate((pears_cat_n['old_z'], pears_cat_s['old_z']), axis=0)

    old_chi2 = np.concatenate((pears_cat_n['old_chi2'], pears_cat_s['old_chi2']), axis=0)
    old_chi2 = np.log10(old_chi2 / 88)

    new_chi2 = np.concatenate((pears_cat_n['new_chi2'], pears_cat_s['new_chi2']), axis=0)
    new_chi2 = np.log10(new_chi2 / 88)

    z_grism_std = np.concatenate((pears_cat_n['new_z_err'], pears_cat_s['new_z_err']), axis=0)

    # only consider redshifts in the range defined by d4000
    valid_zgrism_indices = np.where((z_grism >= 0.6) & (z_grism <= 1.235))[0]
    z_grism = z_grism[valid_zgrism_indices]
    z_phot = z_phot[valid_zgrism_indices]
    z_grism_std = z_grism_std[valid_zgrism_indices]

    # make plots
    # z_grism vs z_phot
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=2.0)

    fig = plt.figure()
    ax1 = fig.add_subplot(gs[:10,:])
    ax2 = fig.add_subplot(gs[10:,:])

    # first subplot
    ax1.plot(z_grism, z_phot, 'o', markersize=1.5, color='k', markeredgecolor='k')
    ax1.plot(np.arange(0.2,1.5,0.01), np.arange(0.2,1.5,0.01), '--', color='r')

    ax1.set_xlim(0.55, 1.3)
    ax1.set_ylim(0.55, 1.3)

    ax1.set_ylabel(r'$z_\mathrm{CANDELS/3DHST}$', fontsize=15)

    ax1.xaxis.set_ticklabels([])

    ax1.minorticks_on()
    ax1.tick_params('both', width=1, length=3, which='minor')
    ax1.tick_params('both', width=1, length=4.7, which='major')
    ax1.grid(True)

    # second subplot
    ax2.plot(z_grism, (z_grism - z_phot)/(1+z_grism), 'o', markersize=1.5, color='k', markeredgecolor='k')
    ax2.axhline(y=0, linestyle='--', color='r')

    ax2.set_xlim(0.55, 1.3)
    ax2.set_ylim(-0.3, 0.3)

    ax2.set_xlabel(r'$z_\mathrm{grism}$', fontsize=15)
    ax2.set_ylabel(r'$(z_\mathrm{grism} - z_\mathrm{CANDELS/3DHST})/(1+z_\mathrm{grism})$', fontsize=15)

    ax2.minorticks_on()
    ax2.tick_params('both', width=1, length=3, which='minor')
    ax2.tick_params('both', width=1, length=4.7, which='major')
    ax2.grid(True)

    fig.savefig(massive_figures_dir + "refined_zspec_vs_zphot.eps", dpi=300, bbox_inches='tight')

    del fig, ax1, ax2

    print np.median(abs(z_grism - z_phot)/(1+z_grism))
    print np.mean(abs(z_grism - z_phot)/(1+z_grism)) # it seems to be convention to quote the mean or median of this quatity as a measure of accuracy of redshifts
    print len(np.where(abs(z_grism - z_phot)/(1+z_grism) > 0.1)[0]) # this is what 3DHST calls catastrophic failure
    print len(np.where(abs(z_grism - z_phot)/(1+z_grism) <= 0.01)[0])

    # redshift hist
    fig = plt.figure()
    ax = fig.add_subplot(111)

    iqr = np.std(z_grism, dtype=np.float64)
    binsize = 2*iqr*np.power(len(z_grism),-1/3)
    totalbins = np.floor((max(z_grism) - min(z_grism))/binsize)

    n, b, p = ax.hist(z_grism, totalbins, facecolor='None', align='mid', linewidth=1, edgecolor='r', histtype='step')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    ax.set_xlabel(r'$\mathrm{z}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    fig.savefig(massive_figures_dir + 'redshift_dist.eps', dpi=300, bbox_inches='tight')

    print '\n', len(z_grism), "galaxies in the refine redshift sample."
    print "Mean redshift of sample:", np.mean(z_grism)
    print "Median redshift of sample:", np.median(z_grism)

    del fig, ax

    # histograms of chi2
    fig = plt.figure()
    ax = fig.add_subplot(111)

    new_chi2 = new_chi2[np.isfinite(new_chi2)]
    old_chi2 = old_chi2[np.isfinite(old_chi2)] 

    iqr = np.std(old_chi2, dtype=np.float64)
    binsize = 2*iqr*np.power(len(old_chi2),-1/3)
    totalbins = np.floor((max(old_chi2) - min(old_chi2))/binsize)

    n_old_chi2, old_bins, patches_old = ax.hist(old_chi2, totalbins, facecolor='None', align='mid', linewidth=1, edgecolor='b', histtype='step')

    n_new_chi2, new_bins, patches_new = ax.hist(new_chi2, bins=old_bins, facecolor='None', align='mid', linewidth=1, edgecolor='r', histtype='step')

    ax.set_xlim(0,3)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    ax.set_xlabel(r'$\mathrm{log(\chi^2_{red})}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    fig.savefig(massive_figures_dir + "refined_old_new_chi2_hist.eps", dpi=300, bbox_inches='tight')

    print '\n', "Mean for old chi2", np.mean(old_chi2)
    print "Median for old chi2", np.median(old_chi2)
    print "Mode for old chi2", (old_bins[np.argmax(n_old_chi2)] + old_bins[np.argmax(n_old_chi2)+1])/2

    print '\n', "Mean for new chi2", np.mean(new_chi2)
    print "Median for new chi2", np.median(new_chi2)
    print "Mode for new chi2", (new_bins[np.argmax(n_new_chi2)] + new_bins[np.argmax(n_new_chi2)+1])/2

    # In these above lines (and below) where I am finding the peak of the histogram and I'm calling it "mode" --
    # The logic here is to find where the peak is i.e. the argmax part,
    # then take the average of the value of the left edge and the right edge between which the peak occurs.
    # This gives the x-value of where the peak is.

    # histogram of normalized error in new redshift
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    count_zero = 0
    count_nonzero_finite = 0
    norm_z_err_plot = np.empty(len(z_grism_std))
    for i in range(len(z_grism_std)):
        delta_z = z_phot[i] - z_grism[i]
        if delta_z == 0:
            count_zero += 1
        elif (delta_z != 0) & ~np.isinf(delta_z / z_grism_std[i]):
            count_nonzero_finite += 1
        norm_z_err_plot[i] = delta_z / z_grism_std[i]

    print '\n', "Total zero values in normalized error of new redshift --", count_zero
    print "Total nonzero and finite values in normalized error of new redshift --", count_nonzero_finite

    rng = 20
    norm_z_err_indx = np.where(abs(norm_z_err_plot) <= rng)[0]
    norm_z_err_plot = norm_z_err_plot[norm_z_err_indx]
    print "Total values in range +-", rng ,"for normalized error of new redshift --", len(norm_z_err_indx)

    n, b, p = ax.hist(norm_z_err_plot[np.isfinite(norm_z_err_plot)], bins='fd', facecolor='None', align='mid', linewidth=1, edgecolor='r', histtype='step') 
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)
    
    ax.set_xlabel(r'$\mathrm{Normalized\ Error\ (\Delta z_{norm} = (z_{CANDELS/3DHST} - z_{grism}) / \sigma_{z_{grism}})}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    #ax.set_xlim(-0.1, 0.2)

    print '\n'
    print "Mean of normalized error of new redshift --", np.mean(norm_z_err_plot[np.isfinite(norm_z_err_plot)])
    print "Median of normalized error of new redshift --", np.median(norm_z_err_plot[np.isfinite(norm_z_err_plot)])
    print "Mode of normalized error of new redshift --", (b[np.argmax(n)] + b[np.argmax(n)+1])/2
    print "Total values of normalized error of new redshift within 3% --", len(np.where(norm_z_err_plot[np.isfinite(norm_z_err_plot)] <= 0.03)[0])
    print "Total values of normalized error of new redshift within 1% --", len(np.where(norm_z_err_plot[np.isfinite(norm_z_err_plot)] <= 0.01)[0])

    fig.savefig(massive_figures_dir + "refined_norm_z_err_hist_inrange_pm_" + str(rng) + ".eps", dpi=300, bbox_inches='tight')
    del fig, ax
    """

    # hist of standard deviation of z_grism
    fig = plt.figure()
    ax = fig.add_subplot(111)

    z_grism_std_indx = np.where(z_grism_std <= 0.2)[0]
    z_grism_std = z_grism_std[z_grism_std_indx]
    z_grism = z_grism[z_grism_std_indx]  # also applying indices to z_grism. This will be used in the sigma/(1+z) histogram.

    n, b, p = ax.hist(z_grism_std[np.isfinite(z_grism_std)], 20, facecolor='None', align='mid', linewidth=1, edgecolor='r', histtype='step')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)
    
    ax.set_xlabel(r'$\mathrm{\sigma_{z_{grism}}}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    #ax.axvline(x=0.037, linestyle='--', color='b')

    ax.set_xlim(0.0, 0.2)

    print '\n'
    print "Mean of measurement uncertainty in new redshift--", np.mean(z_grism_std[np.isfinite(z_grism_std)])
    print "Median of measurement uncertainty in new redshift--", np.median(z_grism_std[np.isfinite(z_grism_std)])
    print "Mode of measurement uncertainty in new redshift--", (b[np.argmax(n)] + b[np.argmax(n)+1])/2
    print "Total values of measurement uncertainty in new redshift within 3% --", len(np.where(z_grism_std[np.isfinite(z_grism_std)] <= 0.03)[0])
    print "Total values of measurement uncertainty in new redshift within 1% --", len(np.where(z_grism_std[np.isfinite(z_grism_std)] <= 0.01)[0])
    print "Total values of measurement uncertainty in new redshift within 0.5% --", len(np.where(z_grism_std[np.isfinite(z_grism_std)] <= 0.005)[0])
    print "Total values of measurement uncertainty in new redshift within 0.3% --", len(np.where(z_grism_std[np.isfinite(z_grism_std)] <= 0.003)[0])
    print "Total values of measurement uncertainty in new redshift within 0.1% --", len(np.where(z_grism_std[np.isfinite(z_grism_std)] <= 0.001)[0])
    print "Total values that are catastrophic failures of redshift estimate > 5% --", len(np.where(z_grism_std[np.isfinite(z_grism_std)] >= 0.05)[0])
    print "Total values that are catastrophic failures of redshift estimate > 10% --", len(np.where(z_grism_std[np.isfinite(z_grism_std)] >= 0.1)[0])

    fig.savefig(massive_figures_dir + "refined_z_err_hist.eps", dpi=300, bbox_inches='tight')
    del fig, ax

    # hist of std of z_grism divided by (1+z_grism)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    z_grism_std_div = z_grism_std / (1 + z_grism)

    n, b, p = ax.hist(z_grism_std_div[np.isfinite(z_grism_std_div)], 20, facecolor='None', align='mid', linewidth=1, edgecolor='r', histtype='step')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)
    
    ax.set_xlabel(r'$\mathrm{\sigma_{z_{grism}}/(1+z_{grism})}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{N}$', fontsize=15)

    #ax.axvline(x=0.037, linestyle='--', color='b')

    ax.set_xlim(0.0, 0.2)

    print '\n'
    print "Numbers for sigma_z_grism divided by (1+z_grism):"
    print "Mean of measurement uncertainty in new redshift--", np.mean(z_grism_std_div[np.isfinite(z_grism_std_div)])
    print "Median of measurement uncertainty in new redshift--", np.median(z_grism_std_div[np.isfinite(z_grism_std_div)])
    print "Mode of measurement uncertainty in new redshift--", (b[np.argmax(n)] + b[np.argmax(n)+1])/2
    print "Total values of measurement uncertainty in new redshift within 3% --", len(np.where(z_grism_std_div[np.isfinite(z_grism_std_div)] <= 0.03)[0])
    print "Total values of measurement uncertainty in new redshift within 1% --", len(np.where(z_grism_std_div[np.isfinite(z_grism_std_div)] <= 0.01)[0])
    print "Total values of measurement uncertainty in new redshift within 0.5% --", len(np.where(z_grism_std_div[np.isfinite(z_grism_std_div)] <= 0.005)[0])
    print "Total values of measurement uncertainty in new redshift within 0.3% --", len(np.where(z_grism_std_div[np.isfinite(z_grism_std_div)] <= 0.003)[0])
    print "Total values of measurement uncertainty in new redshift within 0.1% --", len(np.where(z_grism_std_div[np.isfinite(z_grism_std_div)] <= 0.001)[0])
    print "Total values that are catastrophic failures of redshift estimate > 5% --", len(np.where(z_grism_std_div[np.isfinite(z_grism_std_div)] >= 0.05)[0])
    print "Total values that are catastrophic failures of redshift estimate > 10% --", len(np.where(z_grism_std_div[np.isfinite(z_grism_std_div)] >= 0.1)[0])

    fig.savefig(massive_figures_dir + "refined_z_err_div_oneplusz_hist.eps", dpi=300, bbox_inches='tight')
    del fig, ax
    sys.exit(0)

    # z_grism_std vs net sig
    #fig = plt.figure()
    #ax = fig.add_subplot(111)

    # -------------------------------------------------------------------------------------- # 

    # Read PEARS cats
    pears_master_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag', 'netsig_corr'], usecols=(0,1,2,3,6))
    pears_master_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag', 'netsig_corr'], usecols=(0,1,2,3,6))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_master_ncat['dec'] = pears_master_ncat['dec'] - dec_offset_goodsn_v19

    # Match with Ferreras et al. 2009
    ferreras_cat = np.genfromtxt(massive_galaxies_dir + 'ferreras_2009_ETG_cat.txt', dtype=None,\
                                 names=['id', 'ra', 'dec', 'z'], usecols=(0,1,2,5), skip_header=23)

    ferreras_ids_n = []
    ferreras_ids_s = []

    for i in range(len(ferreras_cat)):

        if ferreras_cat['dec'][i] > 0:
            ferreras_ids_n.append(ferreras_cat['id'][i])

        elif ferreras_cat['dec'][i] < 0:
            ferreras_ids_s.append(ferreras_cat['id'][i])


    #ferreras_prop_cat = np.genfromtxt(massive_galaxies_dir + 'ferreras_2009_ETG_prop_cat.txt', dtype=None,\
    #                             names=['id', 'mstar'], usecols=(0,1), skip_header=23)   

    z_grism_std_n = pears_cat_n['new_z_err']
    z_grism_std_s = pears_cat_s['new_z_err']

    allcats = [pears_cat_n, pears_cat_s]

    z_grism_std_plot = []
    net_sig_plot = []
    exptime_plot = []
    imag_plot = []
    net_sig_corr_cat_plot = []
    d4000_plot = []
    redshift_plot = []

    count_invalid = 0
    for pears_cat in allcats:

        for i in range(len(pears_cat)):

            current_id = pears_cat['pearsid'][i]
            current_field = pears_cat['field'][i]
            current_redshift = pears_cat['new_z'][i]
            current_z_grism_std = pears_cat['new_z_err'][i]

            if (current_redshift >= 0.6) & (current_redshift <= 1.235):
                if current_z_grism_std <= 0.2:
                    if np.isfinite(current_z_grism_std):
                        z_grism_std_plot.append(current_z_grism_std)
                        d4000_plot.append(pears_cat['d4000'][i])
                        redshift_plot.append(current_redshift)

                        # Get the correct filename and the number of extensions
                        #data_path = home + "/Documents/PEARS/data_spectra_only/"
                        #if current_field == 'GOODS-N':
                        #    filename = data_path + 'h_pears_n_id' + str(current_id) + '.fits'
                        #elif current_field == 'GOODS-S':
                        #    filename = data_path + 'h_pears_s_id' + str(current_id) + '.fits'

                        #fitsfile = fits.open(filename)
                        #n_ext = fitsfile[0].header['NEXTEND']

                        ## Find where the highest net sig is for some PA of a galaxy
                        #if n_ext > 1:
                        #    netsiglist = []
                        #    for count in range(n_ext):
                        #        fitsdata = fitsfile[count+1].data
                        #        netsig = gd.get_net_sig(fitsdata, filename)
                        #        netsiglist.append(netsig)
                        #    netsiglist = np.array(netsiglist)
                        #    netsigtoappend = np.max(netsiglist)
                        #    net_sig_plot.append(netsigtoappend)
                        #    max_ind = np.argmax(netsiglist)
                        #    exptime_plot.append(fitsfile[max_ind+1].header['EXPTIME'])
                        #elif n_ext == 1:
                        #    netsigtoappend = gd.get_net_sig(fitsfile[1].data, filename)
                        #    net_sig_plot.append(netsigtoappend)
                        #    exptime_plot.append(fitsfile[1].header['EXPTIME'])
                        
                        if current_field == 'GOODS-N':
                            id_indx = np.where(pears_master_ncat['id'] == current_id)[0]
                            imag_plot.append(pears_master_ncat['imag'][id_indx])
                            net_sig_corr_cat_plot.append(pears_master_ncat['netsig_corr'][id_indx])

                        if current_field == 'GOODS-S':
                            id_indx = np.where(pears_master_scat['id'] == current_id)[0]
                            imag_plot.append(pears_master_scat['imag'][id_indx])
                            net_sig_corr_cat_plot.append(pears_master_scat['netsig_corr'][id_indx])

                        #if netsigtoappend == -99.0:
                        #    count_invalid += 1

    print count_invalid
    print len(net_sig_corr_cat_plot), len(z_grism_std_plot), len(np.where(np.isfinite(d4000_plot) >= 0)[0])
    #ax.plot(np.log10(net_sig_plot), z_grism_std_plot, 'o', markersize=1.5, color='k', markeredgecolor='k')

    #ax.minorticks_on()
    #ax.tick_params('both', width=1, length=3, which='minor')
    #ax.tick_params('both', width=1, length=4.7, which='major')
    #ax.grid(True)

    #ax.set_xlabel(r'$\mathrm{log(Net\ Spectral\ Significance)}$', fontsize=15)
    #ax.set_ylabel(r'$\mathrm{\sigma_{z_{grism}}}$', fontsize=15)    

    #fig.savefig(massive_figures_dir + 'z_err_vs_netsig.eps', dpi=300, bbox_inches='tight')

    #plt.clf()
    #plt.cla()
    #plt.close()

    ## z_grism_std vs exptime
    #fig = plt.figure()
    #ax = fig.add_subplot(111)

    #ax.plot(np.log10(exptime_plot), z_grism_std_plot, 'o', markersize=1.5, color='k', markeredgecolor='k')

    #ax.minorticks_on()
    #ax.tick_params('both', width=1, length=3, which='minor')
    #ax.tick_params('both', width=1, length=4.7, which='major')
    #ax.grid(True)

    #ax.set_xlabel(r'$\mathrm{log(Exp.\ Time)}$', fontsize=15)
    #ax.set_ylabel(r'$\mathrm{\sigma_{z_{grism}}}$', fontsize=15)  

    #fig.savefig(massive_figures_dir + 'z_err_vs_exptime.eps', dpi=300, bbox_inches='tight')

    #plt.clf()
    #plt.cla()
    #plt.close()

    z_grism_std_vs_netsig_corr(z_grism_std_plot, net_sig_corr_cat_plot, d4000_plot, 'refine_samp')
    z_grism_std_vs_redshift(z_grism_std_plot, redshift_plot, 'refine_samp')
    z_grism_std_vs_imag(z_grism_std_plot, imag_plot, 'refine_samp')
    z_grism_std_vs_d4000(z_grism_std_plot, d4000_plot, 'refine_samp')

    del z_grism_std_plot, imag_plot, net_sig_corr_cat_plot, d4000_plot, redshift_plot

    ########################## same plots for Ferreras et al 2009 sample #############################

    allcats = [pears_cat_n, pears_cat_s]

    z_grism_std_plot = []
    imag_plot = []
    net_sig_corr_cat_plot = []
    d4000_plot = []
    redshift_plot = []

    for pears_cat in allcats:

        for i in range(len(pears_cat)):

            current_id = pears_cat['pearsid'][i]
            current_field = pears_cat['field'][i]
            current_redshift = pears_cat['new_z'][i]
            current_z_grism_std = pears_cat['new_z_err'][i]

            if (current_redshift >= 0.6) & (current_redshift <= 1.235):
                if current_z_grism_std <= 0.2:
                    if np.isfinite(current_z_grism_std):
                       
                        if current_field == 'GOODS-N':
                            if current_id in ferreras_ids_n:
                                z_grism_std_plot.append(current_z_grism_std)
                                d4000_plot.append(pears_cat['d4000'][i])
                                redshift_plot.append(current_redshift)

                                id_indx = np.where(pears_master_ncat['id'] == current_id)[0]
                                imag_plot.append(pears_master_ncat['imag'][id_indx])
                                net_sig_corr_cat_plot.append(pears_master_ncat['netsig_corr'][id_indx])

                        elif current_field == 'GOODS-S':
                            if current_id in ferreras_ids_s:
                                z_grism_std_plot.append(current_z_grism_std)
                                d4000_plot.append(pears_cat['d4000'][i])
                                redshift_plot.append(current_redshift)

                                id_indx = np.where(pears_master_scat['id'] == current_id)[0]
                                imag_plot.append(pears_master_scat['imag'][id_indx])
                                net_sig_corr_cat_plot.append(pears_master_scat['netsig_corr'][id_indx])

    print len(net_sig_corr_cat_plot), len(z_grism_std_plot), len(np.where(np.isfinite(d4000_plot) >= 0)[0])

    z_grism_std_vs_netsig_corr(z_grism_std_plot, net_sig_corr_cat_plot, d4000_plot, 'ferreras_samp')
    z_grism_std_vs_redshift(z_grism_std_plot, redshift_plot, 'ferreras_samp')
    z_grism_std_vs_imag(z_grism_std_plot, imag_plot, 'ferreras_samp')
    z_grism_std_vs_d4000(z_grism_std_plot, d4000_plot, 'ferreras_samp')

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)

