img=readfits('grid2.fits',hdr)
;img=readfits('deeper1mag.fits',hdr)


;img=img(0:9,*)
x0=sxpar(hdr,'CRVAL1')
dx=sxpar(hdr,'CDELT1')
y0=sxpar(hdr,'CRVAL2')
dy=sxpar(hdr,'CDELT2')
x0=x0-dx/2.
y0=y0-dy/2.



s=size(img,/dim)
sx=s(0) & sy=s(1)
xmax=x0+dx*(sx+0)
ymax=y0+dy*(sy+0)

scl=1
x=findgen(s(0))*dx/scl+x0
y=findgen(s(1))*dy/scl+y0


;erase,0
startps
!P.multi=[0,2,1,0,0]
;loadct,39,/silent
device,file='z11gridv2.ps',xsize=7.5,ysize=6.,/inch,/color,yoffset=2
;device,file='z11grid_deep1mag.ps',xsize=7.5,ysize=6.,/inch,/color,yoffset=2
myct=39
loadct,myct,/silent
scl=2
; so imdisp hates odd sized arrays
; and frebin will not interpolate for integer rebinning
img=frebin(img,sx*scl,sy*scl)

imdisp,(bytscl(alog10(img),min=alog10(0.03),max=alog10(2000))),/axis,xstyle=13,ystyle=13,position=[0.14,0.12,0.8,0.99],/usepos,bottom=10,/noscale
;loadct,0,/silent
gcol=0
axis,xaxis=0,xr=[x0,xmax],xtitle='Alpha',color=gcol,xtick_get=xtick
axis,xaxis=1,xr=[x0,xmax],color=gcol,xtickname=replicate(' ',n_elements(xtick))
axis,yaxis=0,yr=[y0,ymax],ytitle='M!u*!n (1500 !3' + STRING(197B) + '!X)',color=gcol,ytick_get=ytick
axis,yaxis=1,yr=[y0,ymax],ytickname=replicate(' ',n_elements(ytick)),color=gcol
;loadct,0,/sil
;plotsym,0,5.0,/fill,color=0
;oplot,[-1.73],[-20.29+0.33*(10.7-6.)],psym=8


loadct,myct,/silent
xx=findgen(256)
zz=fltarr(16,256)
for j=0,15 do zz(j,*)=xx

imdisp,(zz),/axis,xstyle=13,ystyle=13,position=[0.82,0.12,0.88,0.99],/zlog,/usepos,bottom=10,/noscale

axis,yaxis=1,yr=[0.03,2000],/ylog,yticklen=0.2,ytick_get=ytick
axis,yaxis=0,yr=[0.03,2000],/ylog,yticklen=0.2,ytickname=replicate(' ',n_elements(ytick))

axis,xaxis=0,xr=[0,15],xtickname=replicate(' ',14),xticklen=0.00001
axis,xaxis=1,xr=[0,15],xtickname=replicate(' ',14),xticklen=0.00001



endps
spawn,'gv z11gridv2.ps &'
;spawn,'gv z11grid_deep1mag.ps &'

end
