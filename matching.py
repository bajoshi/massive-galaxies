"""
    This is the module used to astrometrically match any two given catalogs.
    It will also show diagnostic plots if asked to.
    It should never be run on its own. Use it only to import functions.
"""

# Ideas
# When an object has multiple matches you could check not only for which one is the nearest
# but also which of the multiple matches has the same magnitudes. Of course, this assumes that
# both catalogs being matched also have magnitudes available.


from __future__ import division

import numpy as np
from astropy.io import fits

import glob
import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
pears_spectra_dir = home + "/Documents/PEARS/data_spectra_only/"
figures_dir = stacking_analysis_dir + "figures/"

def convert_arc2deg(deg, arcmin, arcsec):

    if deg < 0.0:
        degrees = deg - (arcmin / 60) - (arcsec / 3600)
    else:
        degrees = deg + (arcmin / 60) + (arcsec / 3600)

    return degrees
    
def convert_sex2deg(rahr, ramin, rasec):

    degrees = rahr*15 + (ramin*15/60) + (rasec*15/3600)

    return degrees

def angular_difference(ra_one, dec_one, ra_two, dec_two):

    delta = 0.00
    delta = np.arccos(np.cos(dec_one*np.pi/180) * np.cos(dec_two*np.pi/180) * np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) +\
                      np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
    # check this formula again # numpy functions accept arguments in radians...

    return delta*180/np.pi  # return result in degrees

def match(ra1, dec1, ra2, dec2, lim=0.1*1/3600):
    
    # Find if the two given regions of the sky are anywhere close
    # Keep in mind that there might exist a large offset that could prevent the catalogs from having any initial overlap
    # You have to decide how large of an initial offset you will allow
    
    delRA = []
    delDEC = []
    cat1_ra_matches = []
    cat1_dec_matches = []
    cat2_ra_matches = []
    cat2_dec_matches = []
    single_count = 0
    
    # Find out which catalog is smaller and loop over that cat
    if len(ra1) <= len(ra2):
        for i in range(len(ra1)):
            index_array = np.where((ra2 <= ra1[i]+lim) & (ra2 >= ra1[i]-lim) & (dec2 <= dec1[i]+lim) & (dec2 >= dec1[i]-lim))[0]
            if index_array.size:
                if len(index_array) == 1:
                    cat2_ramatch = ra2[index_array]
                    cat2_decmatch = dec2[index_array]
                    delRA.append(ra1[i] - cat2_ramatch)
                    delDEC.append(dec1[i] - cat2_decmatch)
                    cat1_ra_matches.append(ra1[i])
                    cat1_dec_matches.append(dec1[i])
                    cat2_ra_matches.append(cat2_ramatch)
                    cat2_dec_matches.append(cat2_decmatch)
                    single_count += 1
                
                elif len(index_array) > 1:
                    continue
                    print 'Invalid Area of code...'
                    min_index = np.argmin(angular_difference(ra1[i], dec1[i], ra2[index_array], dec2[index_array]))
                    cat2_ramatch = ra2[index_array][min_index]
                    cat2_decmatch = dec2[index_array][min_index]
                    delRA.append(ra1[i] - cat2_ramatch)
                    delDEC.append(dec1[i] - cat2_decmatch)
                    cat1_ra_matches.append(ra1[i])
                    cat1_dec_matches.append(dec1[i])
                    cat2_ra_matches.append(cat2_ramatch)
                    cat2_dec_matches.append(cat2_decmatch)
            else:
                continue
    elif len(ra1) > len(ra2):
        for i in range(len(ra2)):
            index_array = np.where((ra1 <= ra2[i]+lim) & (ra1 >= ra2[i]-lim) & (dec1 <= dec2[i]+lim) & (dec1 >= dec2[i]-lim))[0]
            if index_array.size:
                if len(index_array) == 1:
                    cat1_ramatch = ra1[index_array]
                    cat1_decmatch = dec1[index_array]
                    delRA.append(cat1_ramatch - ra2[i])
                    delDEC.append(cat1_decmatch - dec2[i])
                    cat1_ra_matches.append(cat1_ramatch)
                    cat1_dec_matches.append(cat1_decmatch)
                    cat2_ra_matches.append(ra2[i])
                    cat2_dec_matches.append(dec2[i])
                    single_count += 1
                
                elif len(index_array) > 1:
                    continue
                    print 'Invalid Area of code...'
                    min_index = np.argmin(angular_difference(ra2[i], dec2[i], ra1[index_array], dec1[index_array]))
                    cat1_ramatch = ra1[index_array][min_index]
                    cat1_decmatch = dec1[index_array][min_index]
                    delRA.append(cat1_ramatch - ra2[i])
                    delDEC.append(cat1_decmatch - dec2[i])
                    cat1_ra_matches.append(cat1_ramatch)
                    cat1_dec_matches.append(cat1_decmatch)
                    cat2_ra_matches.append(ra2[i])
                    cat2_dec_matches.append(dec2[i])
            else:
                continue

    delRA = np.asarray(delRA, dtype=np.float128)
    delDEC = np.asarray(delDEC, dtype=np.float128)
    cat1_ra_matches = np.asarray(cat1_ra_matches, dtype=np.float128)
    cat1_dec_matches = np.asarray(cat1_dec_matches, dtype=np.float128)
    cat2_ra_matches = np.asarray(cat2_ra_matches, dtype=np.float128)
    cat2_dec_matches = np.asarray(cat2_dec_matches, dtype=np.float128)

    return delRA, delDEC, cat1_ra_matches, cat1_dec_matches, cat2_ra_matches, cat2_dec_matches, single_count

def findoffset(delRA, delDEC):
    """
        This function will determine if there is an offset.
        It will rematch (after subtracting the offset value) if it finds an offset.
    """

    print "Median difference in RA is ", np.median(delRA)
    print "Median difference in DEC is ", np.median(delDEC)

    return np.median(delRA), np.median(delDEC)


def plotfield(ra, dec, catname, fig, ax):
    """
        Will only plot the points in one field.
        Needs the ra and dec of the sources and also the name of the catalog.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    ax.plot(ra, dec, 'o', label=catname, markersize=3, color='k', markeredgecolor='k')
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.legend(loc=0, numpoints=1)
    
    # Save the diagnostic plot in a new directory for plots in the current directory
    currentdir = os.getcwd()
    try:
        os.makedirs(currentdir + '/' + 'DiagnosticPlots')
    except OSError, ose:
        print ose
    fig.savefig(currentdir + '/' + 'DiagnosticPlots' + '/' + 'field', dpi=300)

    del fig, ax
    return None

def fieldoverlay(ra1, dec1, ra2, dec2, cat1name, cat2name, fig, ax):
    """
        Overlays the two catalogs that are being matched.
        Just to see where they are on the sky and if there are any obvious red flags.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    ax.plot(ra1, dec1, 'o', label=cat1name, markersize=2, color='r', markeredgecolor='r')
    ax.plot(ra2, dec2, 'o', label=cat2name, markersize=2, color='b', markeredgecolor='b')
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.legend(loc=0, numpoints=1)

    # Save the diagnostic plotin a new directory for plots in the current directory
    currentdir = os.getcwd()
    try:
        os.makedirs(currentdir + '/' + 'DiagnosticPlots')
    except OSError, ose:
        print ose
    fig.savefig(currentdir + '/' + 'DiagnosticPlots' + '/' + 'fieldsoverlay.eps', dpi=300)

    del fig, ax
    return None

def plot_diff_hist(delRA, delDEC, fig, ax):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\Delta$', fontsize = 12)
    ax.set_ylabel('N', fontsize = 12)
    
    # Convert to arcseconds
    delRA = delRA * 3600
    delDEC = delDEC * 3600

    iqr_ra = 1.349 * np.std(delRA, dtype=np.float64)
    binsize_ra = 2*iqr_ra*np.power(len(delRA), -1/3)  # Freedman-Diaconis Rule
    totalbins_ra = int(np.floor((max(delRA) - min(delRA))/binsize_ra))

    iqr_dec = 1.349 * np.std(delDEC, dtype=np.float64)
    binsize_dec = 2*iqr_dec*np.power(len(delDEC), -1/3)  # Freedman-Diaconis Rule
    totalbins_dec = int(np.floor((max(delDEC) - min(delDEC))/binsize_dec))

    ax.hist(delRA, totalbins_ra, histtype='bar', align='mid', alpha=0.5, label='RA')
    ax.hist(delDEC, totalbins_dec, histtype='bar', align='mid', alpha=0.5, label='DEC')
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    
    majorFormatter = FormatStrFormatter('%1.3f')
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.axvline(x=0, linestyle='--', color='k')
    ax.legend(loc=0)

    # Save the diagnostic plotin a new directory for plots in the current directory
    currentdir = os.getcwd()
    try:
        os.makedirs(currentdir + '/' + 'DiagnosticPlots')
    except OSError, ose:
        print ose
    fig.savefig(currentdir + '/' + 'DiagnosticPlots' + '/' + 'diff_hist', dpi=300)

    del fig, ax
    return None

def plot_diff(delRA, delDEC, inarcseconds=True, name='field'):
    """
    This function expects to get the delRA and delDEC in degress 
    but it will make the plot with the axes in arcseconds unless explicitly told to do otherwise.

    Keyword parameters:
    delRA: float
        The difference array for right ascension. Should be in degrees.

    delDEC: float
        The difference array for declination. Should be in degrees.

    fig: matplotlib figure object
        The figure on which to make the plot.

    ax: matplotlib axes object
        The axes to use with the figure.

    inarcseconds: boolean, optional, default=True
        Switch to convert the input RA and DEC difference arrays to arcseconds.
        By default it expects them in degrees and does the conversion itself.

    name: string, optional, default='field'
        The default name which will be used when saving the figures.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlabel(r'$\Delta \mathrm{RA}$')
    ax.set_ylabel(r'$\Delta \mathrm{DEC}$')
    
    if inarcseconds:
        delRA *= 3600
        delDEC *= 3600
    
    ax.plot(delRA, delDEC, 'o', markersize=1, color='k', markeredgecolor='k')
    
    ax.axhline(y=0, linestyle='--', color='k')
    ax.axvline(x=0, linestyle='--', color='k')
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    
    # Save the diagnostic plotin a new directory for plots in the current directory
    currentdir = os.getcwd()
    if not os.path.isdir(currentdir + '/' + 'Diagnostic_Matching_Plots'):
        try:
            os.makedirs(currentdir + '/' + 'Diagnostic_Matching_Plots')
        except OSError, ose:
            print ose
            print "Oops! Unexpected error encountered..."
    
    fig.savefig(currentdir + '/' + 'Diagnostic_Matching_Plots' + '/' + 'diff_matches_' + name + '.eps', dpi=300)
    
    del fig, ax
    return None
