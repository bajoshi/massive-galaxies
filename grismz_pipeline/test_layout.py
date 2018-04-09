from __future__ import division

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import ManualInterval, ZScaleInterval, LogStretch, ImageNormalize
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import SkyOffsetFrame

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

home = os.getenv('HOME')

if __name__ == '__main__':

    # read taffy image which is much faster for testing purposes
    hdu = fits.open(home + '/Desktop/ipac/taffy_lzifu/taffy_xliners_figs_misc_data/taffy/SDSS/sdss_i_cutout.fits')
    wcs_sdss = WCS(hdu[0].header)

    # Plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection=wcs_sdss)
    
    norm = ImageNormalize(hdu[0].data, interval=ZScaleInterval(), stretch=LogStretch())
    orig_cmap = mpl.cm.Greys
    #im = ax.imshow(hdu[0].data, origin='lower', cmap=orig_cmap, vmin=0.0, vmax=3.0, norm=norm)

    # Set grid etc.
    #ax.set_autoscale_on(False)
    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white')

    # Show patch unrotated and rotated
    box_size = 30/3600  # arcseconds to degreees
    ra_ctr = 0.425
    dec_ctr = 23.497222
    rot_angle = 45  # degrees

    # I think it expects bottom left coordinates 
    # but the RA axis on plots on the sky is usually 
    # flipped so convert it to the bottom right coords.
    ra_br = ra_ctr - box_size/2
    dec_br = dec_ctr - box_size/2

    r0 = Rectangle((ra_br, dec_br), width=box_size, height=box_size, edgecolor='red', facecolor='red', \
        transform=ax.get_transform('icrs'), alpha=0.3)
    #ax.add_patch(r0)

    r1 = Rectangle((ra_ctr, dec_ctr), width=box_size, height=box_size, edgecolor='green', facecolor='green', \
        transform=ax.get_transform('icrs'), angle=45, alpha=0.2)
    #ax.add_patch(r1)

    """
    Tried all kinds of shit within matplotlib, astropy, and aplpy but nothing worked to properly 
    rotate patch around its center. Moving on to doing it by hand. See below.
    """
    r2 = Rectangle((ra_ctr, dec_ctr), width=box_size, height=box_size, edgecolor='blue', facecolor='blue', \
        transform=ax.get_transform('icrs'), alpha=0.2)
    ts = ax.transData
    rc = ts.transform([ra_ctr, dec_ctr])
    tr = mpl.transforms.Affine2D().rotate_deg_around(rc[0], rc[1], 60)
    t = ts + tr
    r2.set_transform(t)
    #ax.add_patch(r2)
    
    # ----------------------
    # Figure out the coords of all corners of the new rotated rectangle
    # by letting Astropy do the math here and then just plot the polygon patch.

    # Trying SkyOffsetFrame from astropy....
    center = SkyCoord(ra=0.425*u.degree, dec=23.497222*u.degree, frame='icrs')

    tl = SkyCoord(ra=(center.ra.degree + box_size/2)*u.degree, dec=(center.dec.degree + box_size/2)*u.degree, frame='icrs')
    tr = SkyCoord(ra=(center.ra.degree - box_size/2)*u.degree, dec=(center.dec.degree + box_size/2)*u.degree, frame='icrs')
    br = SkyCoord(ra=(center.ra.degree - box_size/2)*u.degree, dec=(center.dec.degree - box_size/2)*u.degree, frame='icrs')
    bl = SkyCoord(ra=(center.ra.degree + box_size/2)*u.degree, dec=(center.dec.degree - box_size/2)*u.degree, frame='icrs')

    # Plot all points to check that it knows where the starting points are
    # Plot top left seperately to make sure that top left is exactly the point I think it is.
    all_ra_tocheck = [tr.ra.degree, br.ra.degree, bl.ra.degree]
    all_dec_tocheck = [tr.dec.degree, br.dec.degree, bl.dec.degree]

    #ax.scatter(center.ra.degree, center.dec.degree, s=15, color='red', transform=ax.get_transform('icrs'))
    #ax.scatter(tl.ra.degree, tl.dec.degree, s=10, color='green', transform=ax.get_transform('icrs'))
    #ax.scatter(all_ra_tocheck, all_dec_tocheck, s=10, transform=ax.get_transform('icrs'))

    # Set up a rotated SkyOffsetFrame
    rot_frame = SkyOffsetFrame(representation=None, origin=center, rotation=45*u.degree)

    # rotate and get new coords for the rectangle
    tl_rot = tl.transform_to(rot_frame)
    tr_rot = tr.transform_to(rot_frame)
    br_rot = br.transform_to(rot_frame)
    bl_rot = bl.transform_to(rot_frame)

    print tl.ra, tl.dec
    print tr.ra, tr.dec
    print br.ra, br.dec
    print bl.ra, bl.dec

    print tl_rot.lon, tl_rot.lat
    print tr_rot.lon, tr_rot.lat
    print br_rot.lon, br_rot.lat
    print bl_rot.lon, bl_rot.lat

    # Now plot them as before
    all_rot_ra_tocheck = [tr_rot.lon.degree, br_rot.lon.degree, bl_rot.lon.degree]
    all_rot_dec_tocheck = [tr_rot.lat.degree, br_rot.lat.degree, bl_rot.lat.degree]

    center_rot = center.transform_to(rot_frame)

    print center.position_angle(tl)
    print center.position_angle(tl_rot)

    print center.separation(tl)
    print center.separation(tl_rot)

    print tr.separation(tl)
    print tr_rot.separation(tl_rot)

    ax.scatter(center_rot.lon.degree, center_rot.lat.degree, s=12, color='yellow', transform=ax.get_transform('icrs'))
    ax.scatter(tl_rot.lon.degree, tl_rot.lat.degree, s=10, color='green', transform=ax.get_transform('icrs'))
    ax.scatter(all_rot_ra_tocheck, all_rot_dec_tocheck, s=10, transform=ax.get_transform('icrs'))

    #sys.exit(0)

    # Also check by plotting the polygon patch

    plt.show()
