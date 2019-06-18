img=readfits('grid2.fits',hdr)

alphastep=sxpar(hdr,'CDELT1')
alpha0=sxpar(hdr,'CRVAL1')

mstarstep=sxpar(hdr,'CDELT2')
mstar0=sxpar(hdr,'CRVAL2')

sxaddpar,hdr,'CRPIX1',1.0
sxaddpar,hdr,'CRPIX2',1.0
sxaddpar,hdr,'CRVAL1',alpha0+alphastep/2.0
sxaddpar,hdr,'CRVAL2',mstar0+mstarstep/2.0


writefits,'test.fits',img,hdr

end
