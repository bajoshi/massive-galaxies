pro plotgrid,fitsname,psname=psname,zmax=zmax

if ~keyword_set(psname) then psname=fitsname+'.ps'


img=readfits(fitsname,hdr)

if ~keyword_set(zmax) then zmax=max(img)


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
device,file=psname,xsize=7.5,ysize=6.,/inch,/color,yoffset=2

myct=39
loadct,myct,/silent

scl=2  ; could modify to test sx for oddness but rebinning by 2 seems harmless
; so imdisp hates odd sized arrays
; and frebin will not interpolate for integer rebinning
img=frebin(img,sx*scl,sy*scl)

imdisp,(bytscl(alog10(img),min=alog10(0.03),max=alog10(zmax))),/axis,xstyle=13,ystyle=13,position=[0.14,0.12,0.8,0.99],/usepos,bottom=10,/noscale
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

;create 16 by 256 array where each of the 16 rows are 0..255 -r
zz=replicate(1b,16)#indgen(256)

imdisp,(zz),/axis,xstyle=13,ystyle=13,position=[0.82,0.12,0.88,0.99],/zlog,/usepos,bottom=10,/noscale

axis,yaxis=1,yr=[0.03,zmax],/ylog,yticklen=0.2,ytick_get=ytick
axis,yaxis=0,yr=[0.03,zmax],/ylog,yticklen=0.2,ytickname=replicate(' ',n_elements(ytick))

axis,xaxis=0,xr=[0,15],xtickname=replicate(' ',14),xticklen=0.00001
axis,xaxis=1,xr=[0,15],xtickname=replicate(' ',14),xticklen=0.00001



endps
spawn,'gv '+psname+' &'


!P.multi=[0,0,0,0,0]

end
