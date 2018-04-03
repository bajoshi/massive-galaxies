from __future__ import division

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ManualInterval, ZScaleInterval, LogStretch, ImageNormalize

import sys
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

home = os.getenv('HOME')
figs_dir = home + "/Desktop/FIGS/"
taffy_dir = home + '/Desktop/ipac/taffy/'

sys.path.append(taffy_dir + 'codes/')
import vel_channel_map as vcm

# define functions to convert sexagesimal to degrees
def get_ra_deg(ra_str):
    hr, ramin, rasec = np.asarray(ra_str.split(' ')).astype(np.float)
    ra = hr*15 + (ramin*15/60) + (rasec*15/3600)
    return ra

def get_dec_deg(dec_str):
    deg, arcmin, arcsec = np.asarray(dec_str.split(' ')).astype(np.float)
    if deg < 0.0:
        dec = deg - (arcmin/60) - (arcsec/3600)
    elif deg >= 0.0:
        dec = deg + (arcmin/60) + (arcsec/3600)
    return dec

if __name__ == '__main__':
    
    # read in goods images which will have the pointings overlaid on them
    goodsn = fits.open(figs_dir + 'goodsn_3dhst_v4.0_f606w/goodsn_3dhst.v4.0.F606w_orig_sci.fits')
    goodss = fits.open(figs_dir + 'goodss_3dhst_v4.0_f606w/goodss_3dhst.v4.0.F606w_orig_sci.fits')

    # read in WCS
    wcs_goodsn = WCS(goodsn[0].header)

    # plot image
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection=wcs_goodsn)

    norm = ImageNormalize(goodsn[0].data, interval=ZScaleInterval(), stretch=LogStretch())
    orig_cmap = mpl.cm.Greys
    shifted_cmap = vcm.shiftedColorMap(orig_cmap, midpoint=0.6, name='shifted')
    im = ax.imshow(goodsn[0].data, origin='lower', cmap=shifted_cmap, vmin=-0.005, vmax=0.15, norm=norm)

    # Set tick labels on/off
    ax.set_autoscale_on(False)

    lon = ax.coords[0]
    lat = ax.coords[1]

    lon.set_ticks_visible(True)
    lon.set_ticklabel_visible(True)
    lat.set_ticks_visible(True)
    lat.set_ticklabel_visible(True)
    lon.set_axislabel('')
    lat.set_axislabel('')

    ax.coords.frame.set_color('k')

    # Overlay coord grid
    overlay = ax.get_coords_overlay('fk5')
    overlay.grid(color='white')

    # Plot the pointings by looping over all positions
    # read in PEARS pointings information
    # this file has a wierd format so I'll 
    # have to read it line by line. I simply 
    # copied the info from the readme file 
    # htat came with the master catalogs and 
    # pasted it into the file I'm reading here
    acs_wfc_fov_side = 202/3600  # arcseconds converted to degrees

    with open(figs_dir + 'massive-galaxies/grismz_pipeline/pears_pointings.txt') as f:
        lines = f.readlines()
        for line in lines[4:]:
            current_pointing_set = []
            if 'GOODS-N-PEARS' in line:
                # then its a line indicating the start of 
                # a new set of pointings with the same PA in GOODS-N
                continue
            if 'Position Angle' in line:
                # this indicates the PA
                # Store it and continue
                print line.split(' ')
                pa = float(line.split(' ')[-1])
                continue

            current_ra_str = line.split(' ')[1:4]
            current_dec_str = line.split(' ')[4:7]
            current_ra = get_ra_deg(current_ra_str)
            current_dec = get_dec_deg(current_dec_str)

            print current_ra, current_dec, current_pa

            #r = Rectangle((current_ra, current_dec), \
            #    width=acs_wfc_fov_side / np.cos(current_dec * np.pi/180.0), height=acs_wfc_fov_side, \
            #    angle=current_pa, edgecolor='red', facecolor='red', transform=ax.get_transform('fk5'), alpha=0.1)
            #ax.add_patch(r)

    # Add text for field name

    plt.show()
    plt.cla()
    plt.clf()
    plt.close()

    # Close files
    goodsn.close()
    goodss.close()

    sys.exit(0)