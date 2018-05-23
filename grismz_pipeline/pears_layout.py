from __future__ import division

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ManualInterval, ZScaleInterval, LogStretch, ImageNormalize

import sys
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

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
    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white')

    # Plot the pointings by looping over all positions
    # read in PEARS pointings information
    # this file has a wierd format so I'll 
    # have to read it line by line. I simply 
    # copied the info from the readme file 
    # htat came with the master catalogs and 
    # pasted it into the file I'm reading here
    acs_wfc_fov_side = 202/3600  # arcseconds converted to degrees

    ax.set_aspect('equal')

    with open(figs_dir + 'massive-galaxies/grismz_pipeline/pears_goodsn_pointings.txt') as f:
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
                current_pa = float(line.split(' ')[-1])
                continue

            if 'G800L' in line:
                current_ra_str = " ".join(line.split(' ')[2:5])
                current_dec_str = " ".join(line.split(' ')[5:8])

                current_ra = get_ra_deg(current_ra_str)
                current_dec = get_dec_deg(current_dec_str)

                print current_ra_str, current_dec_str, current_ra, current_dec, current_pa

                # Coordinates of all four corners
                tl = (current_ra + acs_wfc_fov_side/2 , current_dec + acs_wfc_fov_side/2)
                tr = (current_ra - acs_wfc_fov_side/2 , current_dec + acs_wfc_fov_side/2)
                br = (current_ra - acs_wfc_fov_side/2 , current_dec - acs_wfc_fov_side/2)
                bl = (current_ra + acs_wfc_fov_side/2 , current_dec - acs_wfc_fov_side/2)

                fovpoints = [tl, tr, br, bl]
                new_fovpoints = []

                rot_angle = current_pa

                ax.scatter(current_ra, current_dec, s=8, color='green', transform=ax.get_transform('icrs'))

                # Loop over all points defining FoV
                for i in range(4):

                    # Define vector for point
                    # I'm calling this vector 'A'
                    Ara = fovpoints[i][0] * np.pi/180  # numpy cos and sin expect args in radians
                    Adec = fovpoints[i][1] * np.pi/180
                    Ax = np.cos(Adec) * np.cos(Ara)
                    Ay = np.cos(Adec) * np.sin(Ara)
                    Az = np.sin(Adec)

                    # Define rotation axis vector i.e. vector for center of rectangle
                    Kra = current_ra * np.pi/180  # numpy cos and sin expect args in radians
                    Kdec = current_dec * np.pi/180
                    Kx = np.cos(Kdec) * np.cos(Kra)
                    Ky = np.cos(Kdec) * np.sin(Kra)
                    Kz = np.sin(Kdec)

                    # Compute cross product
                    A = np.array([Ax, Ay, Az])
                    K = np.array([Kx, Ky, Kz])

                    # First get teh vector in the FoV plane
                    # Because the older vector defined as 'A' above is 
                    # from the origin to the point.
                    # You want a vector drawn from teh FoV center to the point
                    A_fovplane = A - K

                    crossprod = np.cross(K, A_fovplane)

                    Ax_fovplane = A_fovplane[0]
                    Ay_fovplane = A_fovplane[1]
                    Az_fovplane = A_fovplane[2]

                    # Get new vector in Cartesian
                    newx = Ax_fovplane * np.cos(rot_angle * np.pi/180) + crossprod[0] * np.sin(rot_angle * np.pi/180)
                    newy = Ay_fovplane * np.cos(rot_angle * np.pi/180) + crossprod[1] * np.sin(rot_angle * np.pi/180)
                    newz = Az_fovplane * np.cos(rot_angle * np.pi/180) + crossprod[2] * np.sin(rot_angle * np.pi/180)

                    new_vector_fovplane = np.array([newx, newy, newz])
                    new_vector = new_vector_fovplane + K

                    newx = new_vector[0]
                    newy = new_vector[1]
                    newz = new_vector[2]

                    # Now convert new vector to spherical/astronomical
                    new_ra = np.arctan2(newy, newx) * 180/np.pi
                    new_dec = np.arctan2(newz, np.sqrt(newx*newx + newy*newy)) * 180/np.pi

                    # Save in list to plot polygon later
                    new_fovpoints.append([new_ra, new_dec])

                # Now plot new FoV polygon
                print new_fovpoints
                pnew = Polygon(np.array(new_fovpoints), facecolor='red', closed=True, transform=ax.get_transform('icrs'), alpha=0.05)
                ax.add_patch(pnew)

    # Add text for field name

    plt.show()
    plt.cla()
    plt.clf()
    plt.close()

    # Close files
    goodsn.close()
    goodss.close()

    sys.exit(0)