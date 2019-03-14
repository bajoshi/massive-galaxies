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
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = home + "/Desktop/FIGS/massive-galaxies-figures/"

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

def get_vertices(ra_ctr, dec_ctr, box_size, fieldname):

    # ---------------------------------------------------------------------------------------------- #
    # ------------------------------------ NEW COORD CALCULATION ----------------------------------- #
    # ---------------------------------------------------------------------------------------------- #

    # ------------------------------------ OLD COORD CALCULATION ----------------------------------- #
    #tl = (ra_ctr + box_size/2 , dec_ctr + box_size/2)
    #tr = (ra_ctr - box_size/2 , dec_ctr + box_size/2)
    #br = (ra_ctr - box_size/2 , dec_ctr - box_size/2)
    #bl = (ra_ctr + box_size/2 , dec_ctr - box_size/2)
    # ---------------------------------------------------------------------------------------------- #

    # Get the vertices 
    dec_top = (dec_ctr + box_size/2)

    mu = np.cos(dec_top * np.pi/180) * np.cos(dec_ctr * np.pi/180) * np.cos(ra_ctr * np.pi/180)
    nu = np.cos(dec_top * np.pi/180) * np.cos(dec_ctr * np.pi/180) * np.sin(ra_ctr * np.pi/180)
    sigma = np.sin(dec_top* np.pi/180) * np.sin(dec_ctr * np.pi/180)
    lam = np.cos(box_size * np.pi / (180*np.sqrt(2))) - sigma

    if fieldname == 'n':
        tr = (360 - np.arccos((mu*lam + np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_top)
        tl = (360 - np.arccos((mu*lam - np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_top)
    elif fieldname == 's':  
        # in principle these formulae here should also work for north but they don't.
        # I think because np.arccos() is a multivalued function and it is only giving 
        # me the first valid answer.
        tr = (np.arccos((mu*lam + np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_top)
        tl = (np.arccos((mu*lam - np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_top)

    # -------------------------------------------------------------------------------------- #
    dec_bottom = (dec_ctr - box_size/2)

    mu = np.cos(dec_bottom * np.pi/180) * np.cos(dec_ctr * np.pi/180) * np.cos(ra_ctr * np.pi/180)
    nu = np.cos(dec_bottom * np.pi/180) * np.cos(dec_ctr * np.pi/180) * np.sin(ra_ctr * np.pi/180)
    sigma = np.sin(dec_bottom* np.pi/180) * np.sin(dec_ctr * np.pi/180)
    lam = np.cos(box_size * np.pi / (180*np.sqrt(2))) - sigma

    if fieldname == 'n':
        br = (360 - np.arccos((mu*lam + np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_bottom)
        bl = (360 - np.arccos((mu*lam - np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_bottom)
    elif fieldname == 's':  
        # in principle these formulae here should also work for north but they don't.
        # I think because np.arccos() is a multivalued function and it is only giving 
        # me the first valid answer.
        br = (np.arccos((mu*lam + np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_bottom)
        bl = (np.arccos((mu*lam - np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_bottom)

    # ----------- Additional helpful debugging info ----------- #
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    print "Center:", ra_ctr, dec_ctr

    ccen = SkyCoord(str(ra_ctr), str(dec_ctr), frame='icrs', unit='deg')
    c1_orig = SkyCoord(str(tl[0]), str(tl[1]), frame='icrs', unit='deg')
    c2_orig = SkyCoord(str(tr[0]), str(tr[1]), frame='icrs', unit='deg')
    c3_orig = SkyCoord(str(br[0]), str(br[1]), frame='icrs', unit='deg')
    c4_orig = SkyCoord(str(bl[0]), str(bl[1]), frame='icrs', unit='deg')

    print "Original points at Top:" 
    print c1_orig
    print c2_orig
    print "Original points at Bottom:" 
    print c3_orig
    print c4_orig

    print "Size of Box (arcseconds):", box_size * 3600
    print "Size of Box * sqrt(2) (arcseconds):", box_size * 3600 * np.sqrt(2)
    print "Size of Box / sqrt(2) (arcseconds):", box_size * 3600 / np.sqrt(2)

    print "Separation from center for A:", ccen.separation(c1_orig).arcsecond
    print "Separation from center for B:", ccen.separation(c2_orig).arcsecond
    print "Separation from center for C:", ccen.separation(c3_orig).arcsecond
    print "Separation from center for D:", ccen.separation(c4_orig).arcsecond

    print "Distance AB:", c1_orig.separation(c2_orig).arcsecond
    print "Distance CD:", c3_orig.separation(c4_orig).arcsecond
    print "Distance AD:", c1_orig.separation(c4_orig).arcsecond
    print "Distance BC:", c2_orig.separation(c3_orig).arcsecond

    print "Distance AC (diagonal):", c1_orig.separation(c3_orig).arcsecond
    print "Distance BD (diagonal):", c2_orig.separation(c4_orig).arcsecond
    """

    return tl, tr, br, bl

def do_goods_layout(goods_fitsfile, fieldname):

    # read in WCS
    wcs_goods = WCS(goods_fitsfile[0].header)

    # plot image
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection=wcs_goods)

    norm = ImageNormalize(goods_fitsfile[0].data, interval=ZScaleInterval(), stretch=LogStretch())
    orig_cmap = mpl.cm.Greys
    shifted_cmap = vcm.shiftedColorMap(orig_cmap, midpoint=0.6, name='shifted')

    # NaN out exact zeros before plotting
    idx = np.where(goods_fitsfile[0].data == 0.0)
    goods_fitsfile[0].data[idx] = np.nan
    im = ax.imshow(goods_fitsfile[0].data, origin='lower', cmap=shifted_cmap, vmin=-0.005, vmax=0.3, norm=norm)

    # Set tick labels on/off and other param changes
    ax.set_autoscale_on(False)

    lon = ax.coords[0]
    lat = ax.coords[1]

    lon.set_ticks_visible(True)
    lon.set_ticklabel_visible(True)
    lat.set_ticks_visible(True)
    lat.set_ticklabel_visible(True)
    
    lon.set_axislabel('')
    lat.set_axislabel('')

    lon.set_ticklabel_position('all')
    lat.set_ticklabel_position('all')

    lon.display_minor_ticks(True)
    lat.display_minor_ticks(True)

    lon.set_ticklabel(size=15)
    lat.set_ticklabel(size=15)

    ax.coords.frame.set_color('k')

    # Overlay coord grid
    #overlay = ax.get_coords_overlay('icrs')
    #overlay.grid(color='white')

    # Plot the pointings by looping over all positions
    # read in PEARS pointings information
    # this file has a wierd format so I'll 
    # have to read it line by line. I simply 
    # copied the info from the readme file 
    # htat came with the master catalogs and 
    # pasted it into the file I'm reading here
    acs_wfc_fov_side = 202/3600  # arcseconds converted to degrees

    ax.set_aspect('equal')

    if fieldname == 'n':
        grism_pointing_file = figs_dir + 'massive-galaxies/grismz_pipeline/pears_goodsn_pointings.txt'
    elif fieldname == 's':
        grism_pointing_file = figs_dir + 'massive-galaxies/grismz_pipeline/pears_goodss_pointings.txt'

    with open(grism_pointing_file) as f:
        lines = f.readlines()
        for line in lines[4:]:
            current_pointing_set = []
            if ('GOODS-N-PEARS' in line) or ('GOODS-S-PEARS' in line):
                # then its a line indicating the start of 
                # a new set of pointings with the same PA 
                # in GOODS-N or GOODS-S
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

                tl, tr, br, bl = get_vertices(current_ra, current_dec, acs_wfc_fov_side, fieldname)

                fovpoints = [tl, tr, br, bl]
                new_fovpoints = []
                print current_ra_str, current_dec_str, current_ra, current_dec, current_pa

                rot_angle = current_pa

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
                pnew = Polygon(np.array(new_fovpoints), facecolor='darkred', closed=True, \
                    transform=ax.get_transform('icrs'), alpha=0.02, edgecolor='darkred', linewidth=1.5)
                ax.add_patch(pnew)

    # Add text for field name
    if fieldname == 'n':
        ax.text(0.03, 0.08, r'$\mathrm{GOODS-N}$', \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=17, zorder=10)
    elif fieldname == 's':
        ax.text(0.03, 0.08, r'$\mathrm{GOODS-S}$', \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=17, zorder=10)
    
    # Save figure
    fig.savefig(massive_figures_dir + 'pears_goods' + fieldname + '_layout.pdf', dpi=300, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    return None

if __name__ == '__main__':
    
    # read in goods images which will have the pointings overlaid on them
    goodsn = fits.open(figs_dir + 'goodsn_3dhst.v4.0.F606w_orig_sci.fits')
    do_goods_layout(goodsn, 'n')

    goodss = fits.open(figs_dir + 'goodss_3dhst.v4.0.F606w_orig_sci.fits')
    do_goods_layout(goodss, 's')

    # Close files
    goodsn.close()
    goodss.close()

    sys.exit(0)