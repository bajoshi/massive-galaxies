from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy.cosmology import Planck15 as cosmo

import os
import sys
import glob
import time
import datetime

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, AnchoredText

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
new_codes_dir = home + "/Desktop/FIGS/new_codes/"

sys.path.append(stacking_analysis_dir + 'codes/')
import grid_coadd as gd
import create_fsps_miles_libraries as ct
import fast_chi2_jackknife as fcj
import fast_chi2_jackknife_massive_galaxies as fcjm
import dn4000_catalog as dc

def create_bc03_lib_ssp(pearsid, redshift, field, lam_grid):

    final_fitsname = 'all_comp_spectra_bc03_ssp_withlsf_' + field + '_' + str(pearsid) + '.fits'

    interplsf = fcjm.get_interplsf(pearsid, redshift, field)

    if interplsf is None:
        return None

    # Find total ages (and their indices in the individual fitfile's extensions) that are to be used in the fits
    example = fits.open(home + '/Documents/GALAXEV_BC03/bc03/models/Padova1994/salpeter/bc2003_hr_m22_salp_ssp.fits')
    ages = example[2].data
    age_ind = np.where((ages/1e9 < 8) & (ages/1e9 > 0.1))[0]
    total_ages = int(len(age_ind))  # 57 for SSPs

    # FITS file where the reduced number of spectra will be saved
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)
    hdulist.append(fits.ImageHDU(data=lam_grid))
    
    for filename in glob.glob(home + '/Documents/GALAXEV_BC03/bc03/models/Padova1994/salpeter/' + '*.fits'):
        
        h = fits.open(filename, memmap=False)
        currentlam = h[1].data
    
        # define and initialize numpy array so that you can resample all the spectra at once.
        # It also does the convolution in the for loop below because 
        # I wasn't sure if I gave a 2D array to convolve_fft if it would convolve each row separately.
        # I thought that it might think of the 2D ndarray as an image and convolve it that way which I don't want.
        currentspec = np.zeros([total_ages, len(currentlam)], dtype=np.float64)
        for i in range(total_ages):
            currentspec[i] = h[age_ind[i]+3].data
            currentspec[i] = convolve_fft(currentspec[i], interplsf)

        currentspec = ct.resample(currentlam, currentspec, lam_grid, total_ages)
        currentlam = lam_grid

        for i in range(total_ages):
            hdr = fits.Header()
            hdr['LOG_AGE'] = str(np.log10(ages[age_ind[i]]))

            if 'm22' in filename:
                metal_val = 0.0001
            elif 'm32' in filename:
                metal_val = 0.0004
            elif 'm42' in filename:
                metal_val = 0.004
            elif 'm52' in filename:
                metal_val = 0.008
            elif 'm62' in filename:
                metal_val = 0.02
            elif 'm72' in filename:
                metal_val = 0.05

            hdr['METAL'] = str(metal_val)
            hdulist.append(fits.ImageHDU(data=currentspec[i], header=hdr))

    hdulist.writeto(savefits_dir + final_fitsname, clobber=True)

    return None

def fit_chi2_redshift(orig_lam_grid, orig_lam_grid_model, resampled_spec, ferr, num_samp_to_draw, comp_spec, nexten, spec_hdu, old_z, pearsid, pearsfield):
    """
    This function will refine a prior supplied redshift.

    The different variable names used in here for spectra are:

    1. comp_spec : 
    This is the numpy array made from the LSF convolved model spectra that were originally saved to a fits file.
    The wavelength range used on comp_spec is the extended wavelength range using which they were resampled. 

    2. currentspec :
    Because flam and ferr and their lambda grids have different dimensions from the lambda grid for comp_spec,
    the lambda grid for comp_spec needs to be sliced to match the lambda grid for flam. currentspec is the sliced
    comp_spec array. Their first dimensions are the same i.e. they correspond to all the ages for all metallicites.

    3. current_best_fit_model :
    this is the best fit model that has the same wavelength dimension as currentspec.

    4. current_best_fit_model_whole :
    this is the best fit model that has the same wavelength dimension as comp_spec i.e. the original resampling 
    lambda grid for the models.

    5. current_best_fit_model_chopped :
    this is the model that is chopped to fit the data that is moved across the wavelength axis. It has the same 
    wavelength dimesion as flam.
    """

    fitages = []
    fitmetals = []
    best_exten = []
    old_chi2 = []
    new_chi2 = []
    new_z = []
    new_dn4000 = []
    new_dn4000_err = []
    new_d4000 = []
    new_d4000_err = []

    current_best_fit_model_plot = []
    new_lam_grid_plot = []
    bestalpha_plot = []

    for i in range(int(num_samp_to_draw)):  # loop over bootstrap runs
        # first find best fit assuming old redshift is ok
        if num_samp_to_draw == 1:
            flam = resampled_spec
        elif num_samp_to_draw > 1:
            flam = resampled_spec[i]
        orig_lam_grid_model_indx_low = np.where(orig_lam_grid_model == orig_lam_grid[0])[0][0]
        orig_lam_grid_model_indx_high = np.where(orig_lam_grid_model == orig_lam_grid[-1])[0][0]
        currentspec = comp_spec[:,orig_lam_grid_model_indx_low:orig_lam_grid_model_indx_high+1]

        chi2 = np.zeros(nexten, dtype=np.float64)
        alpha = np.sum(flam * currentspec / (ferr**2), axis=1) / np.sum(currentspec**2 / ferr**2, axis=1)
        chi2 = np.sum(((flam - (alpha * currentspec.T).T) / ferr)**2, axis=1)
        
        bc03_spec = spec_hdu
        # This is to get only physical ages
        sortargs = np.argsort(chi2)
        for k in range(len(chi2)):
            best_age = float(bc03_spec[sortargs[k] + 2].header['LOG_AGE'])
            age_at_z = cosmo.age(old_z).value * 1e9 # in yr
            if (best_age < np.log10(age_at_z)) & (best_age > 9 + np.log10(0.1)):
                fitages.append(best_age)
                fitmetals.append(bc03_spec[sortargs[k] + 2].header['METAL'])
                best_exten.append(sortargs[k] + 2)
                bestalpha_plot.append(alpha[sortargs[k]])
                old_chi2.append(chi2[sortargs[k]])
                current_best_fit_model = currentspec[sortargs[k]]
                current_best_fit_model_whole = comp_spec[sortargs[k]]
                break

        # now shift in wavelength space to get best fit on wavelength grid and correspongind redshift
        low_lim_for_comp = 2500
        high_lim_for_comp = 6500

        start_low_indx = np.argmin(abs(orig_lam_grid_model - low_lim_for_comp))
        start_high_indx = start_low_indx + len(current_best_fit_model) - 1
        # these starting low and high indices are set up in a way that the dimensions of
        # current_best_fit_model_chopped and flam and ferr are the same to be able to 
        # compute alpha and chi2 below.

        chi2_redshift_arr = []
        count = 0 
        while 1:
            current_low_indx = start_low_indx + count
            current_high_indx = start_high_indx + count

            # do the fitting again for each shifted lam grid
            current_best_fit_model_chopped = current_best_fit_model_whole[current_low_indx:current_high_indx+1]

            alpha = np.sum(flam * current_best_fit_model_chopped / (ferr**2)) / np.sum(current_best_fit_model_chopped**2 / ferr**2)
            chi2 = np.sum(((flam - (alpha * current_best_fit_model_chopped)) / ferr)**2)

            chi2_redshift_arr.append(chi2)

            count += 1
            if orig_lam_grid_model[current_high_indx] >= high_lim_for_comp:
                break

        new_chi2.append(np.min(chi2_redshift_arr))
        refined_chi2_indx = np.argmin(chi2_redshift_arr)
        new_lam_grid = orig_lam_grid_model[start_low_indx+refined_chi2_indx:start_high_indx+refined_chi2_indx+1]
        new_dn4000_temp, new_dn4000_err_temp = dc.get_dn4000(new_lam_grid, flam, ferr)
        new_d4000_temp, new_d4000_err_temp = dc.get_d4000(new_lam_grid, flam, ferr)

        new_dn4000.append(new_dn4000_temp)
        new_dn4000_err.append(new_dn4000_err_temp)
        new_d4000.append(new_d4000_temp)
        new_d4000_err.append(new_d4000_err_temp)

        new_z.append(((orig_lam_grid[0] * (1 + old_z)) / new_lam_grid[0]) - 1)
        current_best_fit_model_plot.append(current_best_fit_model)
        new_lam_grid_plot.append(new_lam_grid)
    
    new_chi2_minindx = np.argmin(new_chi2)
    print "Old chi2 -", old_chi2[new_chi2_minindx]
    print "New chi2 -", new_chi2[new_chi2_minindx]

    new_z_minchi2 = new_z[new_chi2_minindx]
    new_z_err = np.std(new_z)
    print "Old and new redshifts -", old_z, "{:.3}".format(new_z_minchi2)
    print "Error in new redshift -", new_z_err, "mean", np.mean(new_z), "median", np.median(new_z)

    new_dn4000_ret = new_dn4000[new_chi2_minindx]
    new_dn4000_err_ret = new_dn4000_err[new_chi2_minindx]
    new_d4000_ret = new_d4000[new_chi2_minindx]
    new_d4000_err_ret = new_d4000_err[new_chi2_minindx]

    # assign flam for plots
    flam = resampled_spec[new_chi2_minindx]
    current_best_fit_model = current_best_fit_model_plot[new_chi2_minindx]
    new_lam_grid = new_lam_grid_plot[new_chi2_minindx]
    bestalpha = bestalpha_plot[new_chi2_minindx]

    if new_z_err <= 0.03:
        savefolder = "err_using_std/"
        plot_comparison_old_new_redshift(orig_lam_grid, flam, ferr, current_best_fit_model,\
        bestalpha, new_lam_grid, orig_lam_grid_model, old_z, new_z_minchi2, pearsid, pearsfield, savefolder)

    if (abs(old_z - new_z_minchi2)/(1+new_z_minchi2)) <= 0.03:
        savefolder = "err_using_deltaz_over_oneplusz/"
        plot_comparison_old_new_redshift(orig_lam_grid, flam, ferr, current_best_fit_model,\
        bestalpha, new_lam_grid, orig_lam_grid_model, old_z, new_z_minchi2, pearsid, pearsfield, savefolder)

    return new_dn4000_ret, new_dn4000_err_ret, new_d4000_ret, new_d4000_err_ret,\
     old_z, new_z_minchi2, new_z_err, old_chi2[new_chi2_minindx], new_chi2[new_chi2_minindx]

def plot_comparison_old_new_redshift(orig_lam_grid, flam, ferr, current_best_fit_model,\
    bestalpha, new_lam_grid, orig_lam_grid_model, old_z, new_z, pearsid, pearsfield, savefolder):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(orig_lam_grid, current_best_fit_model*bestalpha, '-', color='k')
    ax.plot(orig_lam_grid, flam, '-', color='royalblue')
    #ax.fill_between(orig_lam_grid, flam + ferr, flam - ferr, color='lightskyblue')
    #ax.errorbar(orig_lam_grid, flam, yerr=ferr, fmt='-', color='blue')
    #sys.exit(0)

    # plot the newer shifted spectrum
    ax.plot(new_lam_grid, flam, '-', color='red')
    #ax.fill_between(new_lam_grid, flam + ferr, flam - ferr, color='lightred')
    #ax.errorbar(new_lam_grid, flam, yerr=ferr, fmt='-', color='red')

    # shade region for d4000 bands
    arg3850 = np.argmin(abs(new_lam_grid - 3850))
    arg4150 = np.argmin(abs(new_lam_grid - 4150))

    x_fill = np.arange(3750,3951,1)
    y0_fill = np.ones(len(x_fill)) * \
    (current_best_fit_model[arg3850]*bestalpha - 3*0.05*current_best_fit_model[arg3850]*bestalpha)
    y1_fill = np.ones(len(x_fill)) * \
    (current_best_fit_model[arg4150]*bestalpha + 3*0.05*current_best_fit_model[arg4150]*bestalpha)
    ax.fill_between(x_fill, y0_fill, y1_fill, color='lightsteelblue')

    x_fill = np.arange(4050,4251,1)
    ax.fill_between(x_fill, y0_fill, y1_fill, color='lightsteelblue')

    # put in labels for old and new redshifts
    id_labelbox = TextArea(pearsfield + "  " + str(pearsid), textprops=dict(color='k', size=12))
    anc_id_labelbox = AnchoredOffsetbox(loc=2, child=id_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.2, 0.9),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_id_labelbox)

    old_z_labelbox = TextArea(r"$z_{\mathrm{old}} = $" + str(old_z), textprops=dict(color='k', size=12))
    anc_old_z_labelbox = AnchoredOffsetbox(loc=2, child=old_z_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.2, 0.85),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_old_z_labelbox)

    new_z_labelbox = TextArea(r"$z_{\mathrm{new}} = $" + str("{:.3}".format(new_z)), textprops=dict(color='k', size=12))
    anc_new_z_labelbox = AnchoredOffsetbox(loc=2, child=new_z_labelbox, pad=0.0, frameon=False,\
                                         bbox_to_anchor=(0.2, 0.8),\
                                         bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anc_new_z_labelbox)

    # turn on minor ticks and add grid
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    fig.savefig(new_codes_dir + 'plots_from_refining_z_code/' + savefolder + 'refined_z_' + pearsfield + '_' + str(pearsid) + '.eps', dpi=150)

    return None

def plot_err_in_z(new_z):
    
    # This code block will show a histogram for the new z

    fig = plt.figure()
    ax = fig.add_subplot(111)

    iqr = np.std(new_z, dtype=np.float64)
    binsize = 2*iqr*np.power(len(new_z),-1/3)
    totalbins = np.floor((max(new_z) - min(new_z))/binsize)

    ax.hist(new_z, totalbins, facecolor='None', align='mid', linewidth=1, edgecolor='k', histtype='step')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True)

    ax.set_xlabel(r'$\mathrm{z_{new}}$')
    ax.set_ylabel(r'$\mathrm{N}$')

    plt.show()
    plt.cla()
    plt.clf()

    del fig, ax
    return None

def get_avg_dlam(lam):

    dlam = 0
    for i in range(len(lam) - 1):
        dlam += lam[i+1] - lam[i]

    avg_dlam = dlam / (len(lam) - 1)

    return avg_dlam

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # PEARS data path
    data_path = home + "/Documents/PEARS/data_spectra_only/"

    # read in dn4000 catalogs 
    pears_cat_n = np.genfromtxt(massive_galaxies_dir + 'pears_4000break_catalog_GOODS-N.txt',\
     dtype=None, names=True, skip_header=1)
    pears_cat_s = np.genfromtxt(massive_galaxies_dir + 'pears_4000break_catalog_GOODS-S.txt',\
     dtype=None, names=True, skip_header=3)
    #gn1_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/figs_gn1_4000break_catalog.txt',\
    # dtype=None, names=True, skip_header=1)
    #gn2_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/figs_gn2_4000break_catalog.txt',\
    # dtype=None, names=True, skip_header=1)
    #gs1_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/figs_gs1_4000break_catalog.txt',\
    # dtype=None, names=True, skip_header=1)

    print len(pears_cat_n), "objects in PEARS GOODS-N 4000 break catalog."
    print len(pears_cat_s), "objects in PEARS GOODS-S 4000 break catalog."

    # no clear evidence of a break or too noisy or have emmission lines
    # in many of these cases the fits are good but the break is not really
    # clear -- in almost all cases the flux drops at the blue end 
    # but never levels out
    skip_n = [31670,31891,33414,36546,37225,37644,37908,38345,39311,38345,\
    40613,41093,41439,42509,43584,44258,44559,45328,45370,45556,45624,46694,\
    47065,47153,47440,48176,48229,48514,48737,48873,50214,50892,52404,52497,\
    53189,53198,53362,53711,55348,55513,55580,55626,56190,57425,59873,60776,\
    60858,61113,62128,62216,62376,63419,64394,64683,64701,65831,66218,67343,\
    69419,70156,70677,70974,71198,71510,71974,71982,74794,75125,75398,75413,\
    75556,75782,75980,76107,76405,76936,77351,78300,78535,78596,78698,79716,\
    79903,80053,80348,82523,82799,83170,83628,84103,84143,84543,85100,85383,\
    85386,85579,86021,86272,86405,86522,88265,88858,89094,89239,89383,90081,\
    90082,90095,90526,90557,90831,90879,91129,91525,91792,92203,92250,92509,\
    92570,93196,93338,93364,93516,93761,93976,94016,94085,94460,94671,94812,\
    95077,95122,95351,95449,95785,95795,96098,96114,96405,96423,96426,96475,\
    96650,97136,97358,97879,98102,98209,98749,99452,100435,100703,101985,\
    102190,102314,102600,102776,102886,103139,104408,104779,105303,105709,\
    106442,106613,106752,106969,107152,107198,107263,107452,107999,108279,\
    108516,108529,108721,108799,109225,109561,109801,109851,110145,110223,\
    110489,110688,110814,111096,112181,112301,112383,113696,113746,114628,\
    115437,116636,116739,117046,117087,117153,117715,118236,118361,118642,\
    118719,119616,119621,119652,119684,119764,120544,120726,121093,122267,\
    122303,122947,123142,124190,124194,124211,124428,124893,125735,125948,\
    126356]

    skip_s = [11507,14990,15005,15391,17021,17024,17163,17494,17587,18260,\
    18337,18484,19226,19585,19652,19774,20957,21612,22217,24396,25390,25474,\
    26160,26909,28588,30887,31325,32858,33498,33636,33725,34128,35255,35475,\
    35579,35692,35765,35878,35989,37950,38744,38854,39116,40875,41338,42395,\
    43170,44769,44843,44893,45314,45727,47101,50107,51356,52086,52529,52945,\
    53650,54553,57446,57642,61334,61340,62122,62788,65609,66708,66729,67814,\
    67996,68941,68982,69084,69117,69234,70232,70878,71051,71305,71524,73385,\
    73423,74166,74950,75495,75733,76125,76349,76592,76612,78140,78177,78527,\
    80076,81328,81832,85748,85844,85918,86051,86399,87259,90734,91615,92495,\
    92503,93946,94200,94858,95513,95688,95800,96927,97487,97967,98372,98855,\
    99156,100157,100526,100543,100572,101091,101615,102156,102735,103422,103683,\
    104408,105522,107055,107266,108266,108456,108561,108795,109007,109019,\
    109396,109710,109886,109889,110187,110258,110504,110655,110733,110891,\
    111087,113184,113270,113279,117178,117180,117429,117560,117770,118251,\
    118455,119088,119489,119951,120559,120825,120845,121302,121434,121911,\
    122949,123236,123294,123477,123779,124248,124882,124945,125425,126478,\
    126934,126958,127413,128422,129156,130387,131381,137226]

    # these are cases where the old redshift looks correct
    old_z_correct_n = [36842,45958,48743,65404,117385] 
    old_z_correct_s = [40488,43758,81021,84478,101795,110801]

    #### PEARS ####

    allcats = [pears_cat_n, pears_cat_s]

    catcount = 0
    for pears_cat in allcats:

        if catcount == 0:
            fieldname = 'GOODS-N'
        elif catcount == 1:
            fieldname = 'GOODS-S'

        pears_redshift_indices = np.where((pears_cat['redshift'] >= 0.6) & (pears_cat['redshift'] <= 1.235))[0]

        # galaxies in the possible redshift range
        # this number should be the exact same as the length of the catalog
        # which it is .... so this part of the code just serves as a redundancy now.
        print len(pears_redshift_indices)

        # galaxies with significant breaks
        sig_4000break_indices_pears = np.where(((pears_cat['d4000'][pears_redshift_indices] / pears_cat['d4000_err'][pears_redshift_indices]) >= 3.0))[0]

        # use these next two lines if you want to run the code only for a specific galaxy
        #arg = np.where((pears_cat['field'] == 'GOODS-N') & (pears_cat['pears_id'] == 40991))[0]
        #print pears_cat[arg]
    
        # there are 1226 galaxies with SNR on dn4000 greater than 3sigma
        # there are 492 galaxies with SNR on dn4000 greater than 3sigma and less than 20sigma

        # Galaxies with believable breaks; im calling them proper breaks
        prop_4000break_indices_pears = \
        np.where((pears_cat['d4000'][pears_redshift_indices][sig_4000break_indices_pears] >= 1.05) &\
         (pears_cat['d4000'][pears_redshift_indices][sig_4000break_indices_pears] <= 3.5))[0]

        # there are now 483 galaxies in this dn4000 range

        all_pears_ids = pears_cat['pears_id'][pears_redshift_indices][sig_4000break_indices_pears][prop_4000break_indices_pears]
        all_pears_fields = pears_cat['field'][pears_redshift_indices][sig_4000break_indices_pears][prop_4000break_indices_pears]
        all_pears_redshifts = pears_cat['redshift'][pears_redshift_indices][sig_4000break_indices_pears][prop_4000break_indices_pears]
        all_pears_redshift_sources = pears_cat['zphot_source'][pears_redshift_indices][sig_4000break_indices_pears][prop_4000break_indices_pears]
        all_pears_ra = pears_cat['ra'][pears_redshift_indices][sig_4000break_indices_pears][prop_4000break_indices_pears]
        all_pears_dec = pears_cat['dec'][pears_redshift_indices][sig_4000break_indices_pears][prop_4000break_indices_pears]

        total_galaxies = len(all_pears_ids)
        print '\n', "Will attempt to refine redshifts for", total_galaxies, "galaxies in", fieldname

        # Make arrays for writing stuff
        pears_id_to_write = []
        pears_field_to_write = []
        pears_ra_to_write = []
        pears_dec_to_write = []
        pears_old_redshift_to_write = []
        pears_new_redshift_to_write = []
        pears_new_redshift_err_to_write = []
        pears_redshift_source_to_write = []
        pears_dn4000_refined_to_write = []
        pears_dn4000_err_refined_to_write = []
        pears_d4000_refined_to_write = []
        pears_d4000_err_refined_to_write = []
        pears_old_chi2_to_write = []
        pears_new_chi2_to_write = []

        skipped_gal = 0
        counted_gal = 0
        for i in range(total_galaxies):

            current_id = all_pears_ids[i]
            current_redshift = all_pears_redshifts[i]
            current_field = all_pears_fields[i]
            current_redshift_source = all_pears_redshift_sources[i]
            print "\n", "Working on --", current_id

            lam_em, flam_em, ferr, specname = gd.fileprep(current_id, current_redshift, current_field)

            if (current_id in skip_n) and (current_field == 'GOODS-N'):
                skipped_gal += 1
                continue

            if (current_id in skip_s) and (current_field == 'GOODS-S'):
                skipped_gal += 1
                continue
 
            # Contamination rejection
            if np.sum(abs(ferr)) > 0.2 * np.sum(abs(flam_em)):
                print 'Skipping', current_id, 'because of overall contamination.'
                skipped_gal += 1
                continue
 
            # extend lam_grid to be able to move the lam_grid later 
            avg_dlam = get_avg_dlam(lam_em)

            lam_low_to_insert = np.arange(1500, lam_em[0], avg_dlam)
            lam_high_to_append = np.arange(lam_em[-1] + avg_dlam, 7500, avg_dlam)

            resampling_lam_grid = np.insert(lam_em, obj=0, values=lam_low_to_insert)
            resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)

            #create_bc03_lib_ssp(current_id, current_redshift, current_field, resampling_lam_grid)
            #del resampling_lam_grid, avg_dlam, lam_low_to_insert, lam_high_to_append
            #continue

            # Open fits files with comparison spectra
            try:
                bc03_spec = fits.open(savefits_dir + 'all_comp_spectra_bc03_ssp_withlsf_' + current_field + '_' + str(current_id) + '.fits', memmap=False)        
            except IOError as e:
                print e
                print "LSF was not taken into account for this galaxy. Moving on to next galaxy for now."
                skipped_gal += 1
                continue

            # Find number of extensions in each
            bc03_extens = fcj.get_total_extensions(bc03_spec)
            bc03_extens -= 1  # because the first extension is just the resampling grid for the model

            # put in spectra for all ages in a properly shaped numpy array for faster computations

            comp_spec_bc03 = np.zeros([bc03_extens, len(resampling_lam_grid)], dtype=np.float64)
            for i in range(bc03_extens):
                comp_spec_bc03[i] = bc03_spec[i+2].data

            # Get random samples by bootstrapping
            num_samp_to_draw = int(100)
            if num_samp_to_draw == 1:
                resampled_spec = flam_em
            else:
                print "Running over", num_samp_to_draw, "random bootstrapped samples."
                resampled_spec = ma.empty((len(flam_em), num_samp_to_draw))
                for i in range(len(flam_em)):
                    if flam_em[i] is not ma.masked:
                        try:
                            resampled_spec[i] = np.random.normal(flam_em[i], ferr[i], num_samp_to_draw)
                        except ValueError as e:
                            print e
                            skipped_gal += 1
                            break
                    else:
                        resampled_spec[i] = ma.masked
                resampled_spec = resampled_spec.T

            if resampled_spec.size:
                # run the actual fitting function
                new_dn4000, new_dn4000_err, new_d4000, new_d4000_err, old_z, new_z, new_z_err, old_chi2, new_chi2 = \
                fit_chi2_redshift(lam_em, resampling_lam_grid, resampled_spec, ferr,\
                 num_samp_to_draw, comp_spec_bc03, bc03_extens, bc03_spec, current_redshift, current_id, current_field)
            else:
                continue

            counted_gal += 1

            # append stuff to arrays that will finally be written
            pears_id_to_write.append(current_id)
            pears_field_to_write.append(current_field)
            pears_ra_to_write.append(all_pears_ra[i])
            pears_dec_to_write.append(all_pears_dec[i])
            pears_old_redshift_to_write.append(old_z)
            pears_new_redshift_to_write.append(new_z)
            pears_new_redshift_err_to_write.append(new_z_err)
            pears_redshift_source_to_write.append(current_redshift_source)
            pears_dn4000_refined_to_write.append(new_dn4000)
            pears_dn4000_err_refined_to_write.append(new_dn4000_err)
            pears_d4000_refined_to_write.append(new_d4000)
            pears_d4000_err_refined_to_write.append(new_d4000_err)
            pears_old_chi2_to_write.append(old_chi2)
            pears_new_chi2_to_write.append(new_chi2)

        print "Skipped Galaxies :", skipped_gal
        print "Galaxies in sample :", counted_gal

        # write to plain text file
        pears_id_to_write = np.asarray(pears_id_to_write)
        pears_field_to_write = np.asarray(pears_field_to_write, dtype='|S7')
        pears_ra_to_write = np.asarray(pears_ra_to_write)
        pears_dec_to_write = np.asarray(pears_dec_to_write)
        pears_old_redshift_to_write = np.asarray(pears_old_redshift_to_write)
        pears_new_redshift_to_write = np.asarray(pears_new_redshift_to_write)
        pears_new_redshift_err_to_write = np.asarray(pears_new_redshift_err_to_write)
        pears_redshift_source_to_write = np.asarray(pears_redshift_source_to_write)
        pears_dn4000_refined_to_write = np.asarray(pears_dn4000_refined_to_write)
        pears_dn4000_err_refined_to_write = np.asarray(pears_dn4000_err_refined_to_write)
        pears_d4000_refined_to_write = np.asarray(pears_d4000_refined_to_write)
        pears_d4000_err_refined_to_write = np.asarray(pears_d4000_err_refined_to_write)
        pears_old_chi2_to_write = np.asarray(pears_old_chi2_to_write)
        pears_new_chi2_to_write = np.asarray(pears_new_chi2_to_write)

        data = np.array(zip(pears_id_to_write, pears_field_to_write, pears_ra_to_write, pears_dec_to_write, pears_old_redshift_to_write,\
            pears_new_redshift_to_write, pears_new_redshift_err_to_write, pears_redshift_source_to_write, pears_dn4000_refined_to_write,\
            pears_dn4000_err_refined_to_write, pears_d4000_refined_to_write, pears_d4000_err_refined_to_write, pears_old_chi2_to_write, pears_new_chi2_to_write),\
             dtype=[('pears_id_to_write', int), ('pears_field_to_write', '|S7'), ('pears_ra_to_write', float), ('pears_dec_to_write', float),\
             ('pears_old_redshift_to_write', float), ('pears_new_redshift_to_write', float), ('pears_new_redshift_err_to_write', float), ('pears_redshift_source_to_write', '|S7'),\
             ('pears_dn4000_refined_to_write', float), ('pears_dn4000_err_refined_to_write', float), ('pears_d4000_refined_to_write', float),\
             ('pears_d4000_err_refined_to_write', float), ('pears_old_chi2_to_write', float), ('pears_new_chi2_to_write', float)])
        np.savetxt(massive_galaxies_dir + 'pears_refined_4000break_catalog_' + fieldname + '.txt', data,\
         fmt=['%d', '%s', '%.6f', '%.6f', '%.4f', '%.4f','%.4f', '%s', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f'], delimiter=' ',\
         header='Catalog for all galaxies that now have refined redshifts. See paper/code for sample selection. \n' +\
         'pearsid field ra dec old_z new_z new_z_err source dn4000 dn4000_err d4000 d4000_err old_chi2 new_chi2')

        catcount += 1

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)

