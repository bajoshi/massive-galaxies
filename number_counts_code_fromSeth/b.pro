;goto,skip
!P.multi=[0,1,3,0,0]
;window,0,ysize=700
magstep=0.5
multiplot,/reset
startps
device,file='b07cts.ps',ysize=7,xsize=7,/inch,/color,yoffset=1

multiplot,[1,3],gap=0.003,mxtitle='Observed Magnitude at redshifted 1500',$
    mytitle='Number per sq. arcmin per 1.0 mag',mxtitoffset=0.5,mytitoffset=0.8


dz=0.6d0/2.
z1=5.9d0-dz*3.
z2=5.9d0+dz*3.
zselpari=dblarr(3)
zselpari[0]=1.0 ; self normalize
zselpari[1]=0.5*(z1+z2)
zselpari[2]=0.5*(z2-z1)
zselpari[2]=dz
phistari=1.4d-3 ; 
mstari=-20.24d0 ;
alphai=-1.74d0

betai=0.0d0
gammai=0.0d0
h0i=70.0d0
omega_mi=0.30d0
lambda0i=1.0d0-omega_mi
ki=0d0
other=0d0
areai=1.0d0
t0=systime(1)
ncounter,phistari,mstari,alphai,betai,gammai,h0i,omega_mi,lambda0i,ki,z1,z2,areai,zselpari,nofm,mm,nhist,nhistbin
t1=systime(1)
print,'time',t1-t0
plot,[0],xr=[20,35],yr=[.0001,1d4],/ylog
oplot,mm,nofm,line=2
oplot,nhistbin,nhist,psym=10
readcol,'bouwens07.dat',zl,zh,nc,enc
zmid=(zh+zl)/2.0
plotsym,0,1,/fill
oploterror,zmid,nc/magstep,enc/magstep,psym=8,/nohat
oplot,[1,1]*appmag(zselpari[1],mstari),[1d-10,1d10],line=1,thick=0.5
al_legend,['B07 z=5.9 [z850]']
multiplot
;stop

dz=0.7d0/2.
z1=5.0-dz*3.
z2=5.0d0+dz*3.
zselpari=dblarr(3)
zselpari[0]=1.0 ; self normalize
zselpari[1]=0.5*(z1+z2)
zselpari[2]=0.5*(z2-z1)
zselpari[2]=dz

phistari=1.0d-3 ; 
mstari=-20.64d0 ;
alphai=-1.66d0
betai=0.0d0
gammai=0.0d0
h0i=70.0d0
omega_mi=0.30d0
lambda0i=1.0d0-omega_mi
ki=0d0
other=0d0
areai=1.0d0
t0=systime(1)
ncounter,phistari,mstari,alphai,betai,gammai,h0i,omega_mi,lambda0i,ki,z1,z2,areai,zselpari,nofm,mm,nhist,nhistbin
t1=systime(1)
print,'time',t1-t0
plot,[0],xr=[20,35],yr=[.0001,1d4],/ylog
oplot,mm,nofm,line=2
oplot,nhistbin,nhist,psym=10
readcol,'bouwens07_z5.dat',zl,zh,nc,enc
zmid=(zh+zl)/2.0
plotsym,0,1,/fill
oploterror,zmid,nc/magstep,enc/magstep,psym=8,/nohat
oplot,[1,1]*appmag(zselpari[1],mstari),[1d-10,1d10],line=1,thick=0.5
al_legend,['B07 z=5.0 [z850]']
multiplot


dz=0.7d0/2.
z1=3.8-dz*3.
z2=3.8d0+dz*3.
zselpari=dblarr(3)
zselpari[0]=1.0 ; self normalize
zselpari[1]=0.5*(z1+z2)
zselpari[2]=0.5*(z2-z1)
zselpari[2]=dz

phistari=1.3d-3 ; 
mstari=-20.98d0 ;
alphai=-1.73d0
betai=0.0d0
gammai=0.0d0
h0i=70.0d0
omega_mi=0.30d0
lambda0i=1.0d0-omega_mi
ki=0d0
other=0d0
areai=1.0d0
t0=systime(1)
ncounter,phistari,mstari,alphai,betai,gammai,h0i,omega_mi,lambda0i,ki,z1,z2,areai,zselpari,nofm,mm,nhist,nhistbin
t1=systime(1)
print,'time',t1-t0
plot,[0],xr=[20,35],yr=[.0001,1d4],/ylog
oplot,mm,nofm,line=2
oplot,nhistbin,nhist,psym=10
readcol,'bouwens07_z4.dat',zl,zh,nc,enc
zmid=(zh+zl)/2.0
plotsym,0,1,/fill
oploterror,zmid,nc/magstep,enc/magstep,psym=8,/nohat
oplot,[1,1]*appmag(zselpari[1],mstari),[1d-10,1d10],line=1,thick=0.5
al_legend,['B07 z=3.8 [i775]']


endps
spawn,'gv b07cts.ps &'
multiplot,/reset
end
