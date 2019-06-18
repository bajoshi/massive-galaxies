pro ncounter,phistari,mstari,alphai,betai,gammai,h0i,omega_mi,lambda0i,ki,$
      z1,z2,areai,zselpari,nofm,mm,nhist,nhistbin,compcurve=compcurve

; set binned keyword to not bin up the data-faster

; This is a new version based on the old version that did not work
; well.
; essentially started from scratch
; phistar in units of?
; mstar in units of standard Absolute magnitude
; alpha is faint end slope of Schechter function
; beta is power of (1+z)^beta for Luminoisty Evolution
; gamma is power of (1+z)^gamma for number (phi) evolution
; h0 is hubble constant in units of km/s/Mpc
; omega_m is matter density (nominally 0.27 or so)
; lambda0 is cosmological constant (1-omega_m is best?)
;forward_function nm,dnmdz,psi,abmag,dvdz
common const1,h0,omega_m,lambda0,k,c,phistar,alpha,mstar,unit,mag,beta,gamma,$
          zpzp,zselpar
h0=h0i & omega_m=omega_mi & lambda0=lambda0i & k=ki
phistar=phistari & alpha=alphai & mstar=mstari & mag=25d0
zpzp=0d0 & area=areai & zselpar=zselpari

;set some constants
c=3d5
unit=c/h0
sqdeg=3.05d-4 ; steradians per sq deg


areafrac=area/3600.0d0 ; convert to fraction of sq deg

if omega_m+lambda0 ne 1d0 then begin
  print,'Your universe is not flat-'
  print,'Lets fix that'
  k=1d0-omega_m-lambda0
endif

print,z1,z2
minmag=20.0d0
maxmag=40.0d0
magstep=0.5d0
nsubstep=10.
magsubstep=magstep/nsubstep
npt=(maxmag-minmag)/magsubstep
cts=dblarr(npt)
mm=dblarr(npt)
plotsym,0,0.5,/fill
;plot,[0],xr=[20,35],yr=[0.01,1d5]/10,/ylog
for j=0,npt-1 do begin
 mm(j)=minmag+j*magsubstep
 mag=mm(j)
; unit is for dv/dz (2 powers for DLUM^2 included already)
 cts(j)=nm(z1,z2)*unit*sqdeg*areafrac
; print,mm(j),cts(j),alog10(cts(j))
; oplot,[mag],[cts/2.0],psym=8
endfor
; try lazy integration
nbin=npt/nsubstep
;evalstep=magstep*nsubstep
mag0=mm(0)+magstep/2.0
hcts=dblarr(nbin)
bin=dblarr(nbin)
wt=dblarr(n_elements(mm))+1.0d0
if keyword_set(compcurve) then begin
  restore,compcurve
  compstep,mm,climit,newwt
  wt=newwt
endif
for j=0,nbin-1 do begin
  bin(j)=mag0+magstep*j
  use=where(mm lt bin(j)+magstep/2. and mm ge bin(j)-magstep/2.)
  hcts(j)=int_tabulated(mm(use),cts(use)*wt(use),/double,/sort)
endfor


;oplot,mm,cts,psym=3
;oplot,bin,hcts/magstep,psym=10

nofm=cts
nhist=hcts/magstep
nhistbin=bin

;forprint,mm,cts
end ;ncounter










