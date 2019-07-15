from __future__ import division

import numpy as np
from numpy import nansum
import pysynphot
from scipy.interpolate import griddata
from astropy.io import fits
from joblib import Parallel, delayed

import os
import sys
import time
import datetime

figs_data_dir = "/home/bajoshi/models_and_photometry/"
threedhst_datadir = figs_data_dir
cluster_spz_scripts = "/home/bajoshi/spz_scripts/"

sys.path.append(cluster_spz_scripts)
import cluster_do_fitting as cf

def compute_filter_flam(filt, filtername, start, model_comp_spec, model_lam_grid, total_models, zrange):

    print "Working on filter:", filtername

    for z in zrange:
    
        #print "At z:", z
    
        # ------------------------------------ Now compute model filter magnitudes ------------------------------------ #
        filt_flam_model = np.zeros(total_models, dtype=np.float64)
    
        # Redshift the base models
        dl = cf.get_lum_dist(z)  # in Mpc
        dl = dl * 3.086e24  # convert Mpc to cm
        model_comp_spec_z = model_comp_spec / (4 * np.pi * dl * dl * (1+z))
        model_lam_grid_z = model_lam_grid * (1+z)
    
        # first interpolate the transmission curve to the model lam grid
        filt_interp = griddata(points=filt['wav'], values=filt['trans'], xi=model_lam_grid_z, method='linear')
    
        # multiply model spectrum to filter curve
        for i in xrange(total_models):
    
            num = nansum(model_comp_spec_z[i] * filt_interp)
            den = nansum(filt_interp)
    
            filt_flam_model = num / den
            filt_flam_model[i] = filt_flam_model
    
        # transverse array to make shape consistent with others
        # I did it this way so that in the above for loop each filter is looped over only once
        # i.e. minimizing the number of times each filter is gridded on to the model grid
        #filt_flam_model_t = filt_flam_model.T

    # save the model flux densities
    np.save(figs_data_dir + 'all_model_flam_' + filtername + '.npy', filt_flam_model)
    print "Computation done and saved for:", filtername, "\n"
    print "Total time taken:", time.time() - start

    return None

def save_hst_filters_to_npy():

    # Read in HST filter curves using pysynphot
    f435w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f435w')
    f606w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f606w')
    f775w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f775w')
    f850lp_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f850lp')

    f125w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f125w')
    f140w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f140w')
    f160w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f160w')

    # Save to simple numpy arrays
    all_hst_filters = [f435w_filt_curve, f606w_filt_curve, f775w_filt_curve, f850lp_filt_curve, \
    f125w_filt_curve, f140w_filt_curve, f160w_filt_curve]
    all_hst_filternames = ['f435w', 'f606w', 'f775w', 'f850lp', 'f125w', 'f140w', 'f160w']

    for i in range(len(all_hst_filters)):
        # Get filter and name
        filt = all_hst_filters[i]
        filtername = all_hst_filternames[i]

        # Put into numpy arrays and save
        wav = np.array(filt.binset)
        trans = np.array(filt(filt.binset))
        
        data = np.array(zip(wav, trans), dtype=[('wav', np.float64), ('trans', np.float64)])
        np.savetxt(massive_galaxies_dir + 'grismz_pipeline/' + filtername + '_filt_curve.txt', data, \
            fmt=['%.6f', '%.6f'], header='wav trans')

    return None

def main():

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Read in models with emission lines adn put in numpy array
    total_models = 37761

    model_lam_grid_withlines_mmap = np.load(figs_data_dir + 'model_lam_grid_withlines.npy', mmap_mode='r')
    model_comp_spec_withlines_mmap = np.load(figs_data_dir + 'model_comp_spec_withlines.npy', mmap_mode='r')

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

    # ------------------------------- Read in filter curves ------------------------------- #
    """
    The HST filters, in their PYSYNPHOT form, cannot be used in conjunction with
    the joblib module. So this function will read them and save them into numpy arrays.
    This function has to be run once to convert the HST filters to text files that 
    can be read with genfromtxt.
    """
    #save_hst_filters_to_npy()

    uband_curve = np.genfromtxt(figs_data_dir + 'filter_curves/kpno_mosaic_u.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=14)
    f435w_filt_curve = np.genfromtxt(figs_data_dir + 'filter_curves/f435w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    f606w_filt_curve = np.genfromtxt(figs_data_dir + 'filter_curves/f606w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    f775w_filt_curve = np.genfromtxt(figs_data_dir + 'filter_curves/f775w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    f850lp_filt_curve = np.genfromtxt(figs_data_dir + 'filter_curves/f850lp_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    f125w_filt_curve = np.genfromtxt(figs_data_dir + 'filter_curves/f125w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    f140w_filt_curve = np.genfromtxt(figs_data_dir + 'filter_curves/f140w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    f160w_filt_curve = np.genfromtxt(figs_data_dir + 'filter_curves/f160w_filt_curve.txt', \
        dtype=None, names=['wav', 'trans'])
    irac1_curve = np.genfromtxt(figs_data_dir + 'filter_curves/irac1.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac2_curve = np.genfromtxt(figs_data_dir + 'filter_curves/irac2.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac3_curve = np.genfromtxt(figs_data_dir + 'filter_curves/irac3.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac4_curve = np.genfromtxt(figs_data_dir + 'filter_curves/irac4.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)

    # IRAC wavelengths are in mixrons # convert to angstroms
    irac1_curve['wav'] *= 1e4
    irac2_curve['wav'] *= 1e4
    irac3_curve['wav'] *= 1e4
    irac4_curve['wav'] *= 1e4

    all_filters = [uband_curve, f435w_filt_curve, f606w_filt_curve, f775w_filt_curve, f850lp_filt_curve, \
    f125w_filt_curve, f140w_filt_curve, f160w_filt_curve, irac1_curve, irac2_curve, irac3_curve, irac4_curve]
    all_filter_names = ['u', 'f435w', 'f606w', 'f775w', 'f850lp', \
    'f125w', 'f140w', 'f160w', 'irac1', 'irac2', 'irac3', 'irac4']

    # Loop over all redshifts and filters and compute magnitudes
    zrange = np.arange(0.001, 6.001, 0.001)
    print "Redshift grid for models:"
    print zrange

    # Will use as many cores as filters
    processes = [mp.Process(target=compute_filter_flam, args=(all_filters[i], all_filter_names[i], start, \
        model_comp_spec_withlines, model_lam_grid_withlines, total_models, zrange)) for i in len(all_filters)]
    for p in processes:
        p.start()
        print "Current process ID:", p.pid
    for p in processes:
        p.join()

    print "All done. Total time taken:", time.time() - start
    return None

if __name__ == '__main__':
    main()