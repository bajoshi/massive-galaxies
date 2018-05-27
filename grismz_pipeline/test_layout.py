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
from matplotlib.patches import Rectangle, Polygon

home = os.getenv('HOME')

if __name__ == '__main__':

    # read taffy image which is much faster for testing purposes
    hdu = fits.open(home + '/Desktop/ipac/taffy_lzifu/taffy_xliners_figs_misc_data/taffy/SDSS/sdss_i_cutout.fits')
    wcs_sdss = WCS(hdu[0].header)

    # --------------------------------- Plot basic image --------------------------------- #
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection=wcs_sdss)
    
    norm = ImageNormalize(hdu[0].data, interval=ZScaleInterval(), stretch=LogStretch())
    orig_cmap = mpl.cm.Greys
    im = ax.imshow(hdu[0].data, origin='lower', cmap=orig_cmap, vmin=0.0, vmax=3.0, norm=norm)

    # Set grid etc.
    #ax.set_autoscale_on(False)
    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white')

    # --------------------------------- Basic info for patch --------------------------------- #
    # Show patch unrotated and rotated
    box_size = 30/3600  # arcseconds to degreees
    ra_ctr = 0.425
    dec_ctr = 23.497222
    rot_angle = 65  # degrees

    # --------------------------------- Start adding patches --------------------------------- #
    # I think it expects bottom left coordinates 
    # but the RA axis on plots on the sky is usually 
    # flipped so convert it to the bottom right coords.
    ra_br = ra_ctr - box_size/2
    dec_br = dec_ctr - box_size/2

    r0 = Rectangle((ra_br, dec_br), width=box_size, height=box_size, edgecolor='red', facecolor='red', \
        transform=ax.get_transform('icrs'), alpha=0.3)
    ax.add_patch(r0)

    r1 = Rectangle((ra_ctr, dec_ctr), width=box_size, height=box_size, edgecolor='green', facecolor='green', \
        transform=ax.get_transform('icrs'), angle=45, alpha=0.2)
    ax.add_patch(r1)

    """
    Tried all kinds of shit within matplotlib, astropy, and aplpy but nothing worked to properly 
    rotate patch around its center. Moving on to doing it by hand. See below.
    """
    r2 = Rectangle((ra_ctr, dec_ctr), width=box_size, height=box_size, edgecolor='blue', facecolor='blue', \
        transform=ax.get_transform('icrs'), alpha=0.2)
    ts = ax.transData
    rc = ts.transform([ra_ctr, dec_ctr])
    #print ra_ctr, dec_ctr
    #print rc
    inv = ax.transData.inverted()
    #print inv.transform((rc[0], rc[1]))
    tr = mpl.transforms.Affine2D().rotate_deg(30)
    t = ts + tr
    r2.set_transform(t)
    ax.add_patch(r2)

    # Check polygon patch
    p = Polygon(np.array([[0.425, 23.502778], [0.4180556, 23.4966667], \
        [0.425, 23.49166667], [0.431388, 23.4966667]]), closed=True, transform=ax.get_transform('icrs'), alpha=0.3)
    ax.add_patch(p)
    # So computing the polygon by hand and then plotting it will work for sure

    ax.set_aspect('equal')

    # ---------------------------------------------------------------------------------------------- #
    # ------------------------------------ NEW COORD CALCULATION ----------------------------------- #
    # ---------------------------------------------------------------------------------------------- #
    # Now compute new FoV veritces using Rodrigues's formula
    # You will need the cneter and the rotation angle
    #tl = (ra_ctr + box_size/2 , dec_ctr + box_size/2)
    #tr = (ra_ctr - box_size/2 , dec_ctr + box_size/2)
    #br = (ra_ctr - box_size/2 , dec_ctr - box_size/2)
    #bl = (ra_ctr + box_size/2 , dec_ctr - box_size/2)
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    print "Center:", ra_ctr, dec_ctr

    # Get the vertices first 
    # First vertex: Top Left
    dec_top = (dec_ctr + box_size/2)

    mu = np.cos(dec_top * np.pi/180) * np.cos(dec_ctr * np.pi/180) * np.cos(ra_ctr * np.pi/180)
    nu = np.cos(dec_top * np.pi/180) * np.cos(dec_ctr * np.pi/180) * np.sin(ra_ctr * np.pi/180)
    sigma = np.sin(dec_top* np.pi/180) * np.sin(dec_ctr * np.pi/180)
    lam = np.cos(box_size * np.pi / (180*np.sqrt(2))) - sigma
    print mu*lam
    print np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4)

    tr = (np.arccos((mu*lam + np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_top)
    tl = (np.arccos((mu*lam - np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_top)

    # -------------------------------------------------------------------------------------- #
    dec_bottom = (dec_ctr - box_size/2)

    mu = np.cos(dec_bottom * np.pi/180) * np.cos(dec_ctr * np.pi/180) * np.cos(ra_ctr * np.pi/180)
    nu = np.cos(dec_bottom * np.pi/180) * np.cos(dec_ctr * np.pi/180) * np.sin(ra_ctr * np.pi/180)
    sigma = np.sin(dec_bottom* np.pi/180) * np.sin(dec_ctr * np.pi/180)
    lam = np.cos(box_size * np.pi / (180*np.sqrt(2))) - sigma

    br = (np.arccos((mu*lam + np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_bottom)
    bl = (np.arccos((mu*lam - np.sqrt(mu**2*nu**2 - nu**2*lam**2 + nu**4))/(mu**2 + nu**2)) * 180/np.pi, dec_bottom)

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

    sys.exit(0)

    # plot center point
    ax.scatter(ra_ctr, dec_ctr, s=12, color='blue', transform=ax.get_transform('icrs'))

    fovpoints = [tl, tr, br, bl]
    new_fovpoints = []

    from astropy.coordinates import SkyCoord
    from astropy import units as u

    c1_orig = SkyCoord(str(fovpoints[0][0]), str(fovpoints[0][1]), frame='icrs', unit='deg')
    c2_orig = SkyCoord(str(fovpoints[1][0]), str(fovpoints[1][1]), frame='icrs', unit='deg')
    sep_orig = c1_orig.separation(c2_orig)
    print "Original points:", c1_orig, c2_orig
    print "Separation in arcseconds for original points:", sep_orig.arcsecond
    sys.exit(0)

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
        Kra = ra_ctr * np.pi/180  # numpy cos and sin expect args in radians
        Kdec = dec_ctr * np.pi/180
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

        #print "Numpy dot product:", np.dot(A_fovplane, K)  # This should be almost zero because they are almost perpendicular

        Ax_fovplane = A_fovplane[0]
        Ay_fovplane = A_fovplane[1]
        Az_fovplane = A_fovplane[2]

        #crossx = np.cos(Kdec) * np.sin(Kra) * np.sin(Adec) - np.sin(Kdec) * np.cos(Adec) * np.sin(Ara)
        #crossy = np.sin(Kdec) * np.cos(Adec) * np.cos(Ara) - np.sin(Adec) * np.cos(Kdec) * np.cos(Kra)
        #crossz = np.cos(Kdec) * np.cos(Kra) * np.cos(Adec) * np.sin(Ara) - np.cos(Kdec) * np.sin(Kra) * np.cos(Adec) * np.cos(Ara)
        #print "Hand cross product :", np.array([crossx, crossy, crossz])

        # Get new vector in Cartesian
        newx = Ax_fovplane * np.cos(rot_angle * np.pi/180) + crossprod[0] * np.sin(rot_angle * np.pi/180)
        newy = Ay_fovplane * np.cos(rot_angle * np.pi/180) + crossprod[1] * np.sin(rot_angle * np.pi/180)
        newz = Az_fovplane * np.cos(rot_angle * np.pi/180) + crossprod[2] * np.sin(rot_angle * np.pi/180)

        new_vector_fovplane = np.array([newx, newy, newz])

        #print "Numpy dot product:", np.dot(new_vector_fovplane, K)  # This should be almost zero because they are almost perpendicular
        new_vector = new_vector_fovplane + K
        #print "Numpy dot product:", np.dot(new_vector, K)  # This should be 1 because they are almost in the same direction

        newx = new_vector[0]
        newy = new_vector[1]
        newz = new_vector[2]

        # Now convert new vector to spherical/astronomical
        new_ra = np.arctan(newy/newx) * 180/np.pi
        new_dec = np.arctan(newz/np.sqrt(newx*newx + newy*newy)) * 180/np.pi

        # Plot old and new points
        ax.scatter(fovpoints[i][0], fovpoints[i][1], s=12, color='magenta', transform=ax.get_transform('icrs'))
        ax.scatter(new_ra, new_dec, s=12, color='green', transform=ax.get_transform('icrs'))

        # Save in list to plot polygon later
        new_fovpoints.append([new_ra, new_dec])

    # Now plot new FoV polygon
    print new_fovpoints
    pnew = Polygon(np.array(new_fovpoints), facecolor='darkslategray', closed=True, transform=ax.get_transform('icrs'), alpha=0.7)
    ax.add_patch(pnew)

    plt.show()
    sys.exit(0)
    
    # ------------------- THIS STUFF BELOW DIDN'T WORK -------------------- #
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

    plt.show()
    #sys.exit(0)
