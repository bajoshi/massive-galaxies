magstep=0.5
goto,wind

phistari=1.14d-3 ;Oesch
z1=7.2d0+2
z2=8.5d0+2
zz=0.5d0*(z1+z2)
zselpari=dblarr(3)
zselpari[0]=1.0 ; self normalize
zselpari[1]=0.5*(z1+z2)
zselpari[2]=0.5*(z2-z1)
zselpari[2]=0.7/2
mstari=-20.29d0+0.33*(zz-6.0d0)
mstari=-18.
alphai=-1.73d0
betai=0.0d0
gammai=0.0d0
h0i=71.0d0
omega_mi=0.27d0
lambda0i=1.0d0-omega_mi
ki=0d0
other=0d0
areai=4.7
ncounter,phistari,mstari,alphai,betai,gammai,h0i,omega_mi,lambda0i,ki,z1,z2,areai,zselpari,nofm,mm,nhist,nhistbin,compcurve='fivesigmalimit.idl'

wind:
phistari=1.00d-3 ;Windhorst approved avg of planck and WMAP
;10.7
;9-12.3, 10-11.5
z1=9.0
z2=12.3
zz=10.7
zselpari=dblarr(3)
zselpari[0]=1.0 ; self normalize
zselpari[1]=10.7
zselpari[2]=0.5*(z2-z1)
zselpari[2]=0.7
mstari=-20.29d0+0.33*(zz-6.0d0)
mstari=-18.
alphai=-1.73d0
betai=0.0d0
gammai=0.0d0
h0i=68.5d0
omega_mi=0.30d0
lambda0i=1.0d0-omega_mi
ki=0d0
other=0d0
areai=4.7
ncounter,phistari,mstari,alphai,betai,gammai,h0i,omega_mi,lambda0i,ki,z1,z2,areai,zselpari,nofm,mm,nhist,nhistbin,compcurve='fivesigmalimit.idl'

plot,[0],xr=[20,35],yr=[.01,1d4],/ylog
oplot,mm,nofm,line=2
oplot,nhistbin,nhist,psym=10
oplot,nhistbin,total(nhist,/cum),psym=10,line=3
oplot,[1,1]*appmag(zselpari[1],mstari),[1d-10,1d10],line=1,thick=0.5
oplot,[0,100],[1,1],line=1, thick=0.5

stop

phistari=1.d-3
mstari=[-21,-20.5,-20,-19.5,-19,-18.5,-18,-17.5,-17,-16.5,-16]*1d0
alphai=-1.0d0*(findgen(11)/10.d0+1.5d0)



end
