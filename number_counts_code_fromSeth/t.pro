;goto,skip
;!P.multi=[0,1,2,0,0]
;window,0,ysize=700
magstep=0.5
multiplot,/reset
startps
device,file='z2cts.ps',ysize=7,xsize=7,/inch,/color,yoffset=1
multiplot,[1,2],gap=0.001,mxtitle='Observed Magnitude B!dF450W!n [1500 @ z=2]',$
    mytitle='Number per 50 sq. arcmin per 1.0 mag',mxtitoffset=0.1,$
    mytitoffset=0.1

;goto,oesch
dz=0.4d0
z1=2.1d0-dz*3.
z2=2.1d0+dz*3.
zselpari=dblarr(3)
zselpari[0]=1.0 ; self normalize in zselect
zselpari[1]=0.5*(z1+z2)
zselpari[2]=dz ; seems more correct
phistari=1.57d-3 ; nph
mstari=-20.39d0 ;nph
alphai=-1.17d0
betai=0.0d0
gammai=0.0d0
h0i=70.5d0
omega_mi=0.274d0
lambda0i=1.0d0-omega_mi
ki=0d0
other=0d0
areai=50.0d0
t0=systime(1)
ncounter,phistari,mstari,alphai,betai,gammai,h0i,omega_mi,lambda0i,ki,z1,z2,areai,zselpari,nofm,mm,nhist,nhistbin
t1=systime(1)
print,'time',t1-t0
plot,[0],xr=[20,35],yr=[.1,1d5],/ylog
oplot,mm,nofm,line=2
oplot,nhistbin,nhist,psym=10
readcol,'redropouts/u2_catalog.txt',num,ra,dec,u1,e1,u2,eu2,u3,eu3,b450,eb450
realhist,b450,0,30,magstep,h,bin
plotsym,0,1,/fill
oplot,bin,(h)/magstep,psym=8
oplot,[1,1]*appmag(zselpari[1],mstari),[1d-10,1d10],line=1,thick=0.5
print,'Magnitude',appmag(zselpari[1],mstari)
al_legend,['Hathi LF/Hathi Data']

multiplot

oesch:
dz=0.4d0/2.0   ; paper says dz=0.4 but that looks like 2*dz
z1=1.9d0-dz*3.
z2=1.9d0+dz*3.
zselpari=dblarr(3)
zselpari[0]=1.0 ; self normalize
zselpari[1]=1.9d0
zselpari[2]=dz

phistari=10^(-2.66d0) ; OEsch
mstari=-20.16d0
alphai=-1.60d0
betai=0.0d0
gammai=0.0d0
h0i=70.0d0
omega_mi=0.3d0
lambda0i=1.0d0-omega_mi
ki=0d0
other=0d0
areai=50.0d0
ncounter,phistari,mstari,alphai,betai,gammai,h0i,omega_mi,lambda0i,ki,z1,z2,areai,zselpari,nofm,mm,nhist,nhistbin
plot,[0],xr=[20,35],yr=[0.1,1d5],/ylog
oplot,mm,nofm,line=2
oplot,nhistbin,nhist,psym=10
readcol,'redropouts/u2_catalog.txt',num,ra,dec,u1,e1,u2,eu2,u3,eu3,b450,eb450
realhist,b450,0,30,magstep,h,bin
plotsym,0,1,/fill
oplot,bin,(h)/magstep,psym=8
oplot,[1,1]*appmag(zselpari[1],mstari),[1d-10,1d10],line=1,thick=0.5
print,'Magnitude',appmag(zselpari[1],mstari)

al_legend,['Oesch LF/Hathi Data']

endps
spawn,'gv z2cts.ps &'
multiplot,/reset
stop

skip:
phistari=1.14d-3 ;Oesch
z1=7.2d0
z2=8.5d0
zz=0.5d0*(z1+z2)
mstari=-20.29d0+0.33*(zz-6.0d0)
alphai=-1.73d0
betai=0.0d0
gammai=0.0d0
h0i=71.0d0
omega_mi=0.27d0
lambda0i=1.0d0-omega_mi
ki=0d0
other=0d0
areai=4.7
ncounter,phistari,mstari,alphai,betai,gammai,h0i,omega_mi,lambda0i,ki,z1,z2,areai
end
