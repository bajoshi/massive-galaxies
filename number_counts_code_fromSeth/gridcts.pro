magstep=0.5
outroot='grid_z_107'
!except=0  ; turn off underflow
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
phistari=1.d-3
mstarv=[-21,-20.5,-20,-19.5,-19,-18.5,-18,-17.5,-17,-16.5,-16]*1d0
mstarv=reverse(mstarv)
alphav=-1.0d0*(findgen(11)/10.d0+1.5d0)

nalpha=n_elements(alphav)
nmstar=n_elements(mstarv)
alphastep=(alphav[1]-alphav[0])
mstarstep=mstarv[1]-mstarv[0]
alpha0=alphav[0]    ; looks like it uses center of pixel
mstar0=mstarv[0]    ;
gridct=fltarr(nalpha,nmstar)+!VALUES.F_NAN
mkhdr,hdr,gridct,/image
make_astr,astr,crpix=[1,1],crval=[alpha0,mstar0],delt=[alphastep,mstarstep],$
      ctype=['alpha','mstar']
putast,hdr,astr,CD_TYPE=0
; add other parameters here
sxaddpar,hdr,'HUBBLE0',h0i,'Hubble Constant'
sxaddpar,hdr,'OMEGA_M',omega_mi,'Matter density'
sxaddpar,hdr,'LAMBDA0',lambda0i,'Cosmological Constant'
sxaddpar,hdr,'K_CURVE',omega_mi+lambda0i,'Curvature'
sxaddpar,hdr,'PHISTAR',phistari,'Number per mag per Mpc^3'
sxaddpar,hdr,'AREAOBS',areai,'Area in sq. arcmin'
sxaddpar,hdr,'REDSHFT',zselpari[1],'Average Redshift'
sxaddpar,hdr,'SIGMA_Z',zselpari[2],'Sigma of z-selection funct.'
sxaddpar,hdr,'BETA_EV',betai,'Luminosity evolution-not used'
sxaddpar,hdr,'GAMMAEV',gammai,'Density evolution-not used'

runnum=1.0
for j=0,nalpha-1 do begin
 for k=0,nmstar-1 do begin
  print,'Sim# ',strcompress(runnum,/rem),' Percent:',runnum/(nalpha*nmstar),'%'
  runnum=runnum+1.0
  alphai=alphav[j]
  mstari=mstarv[k]
  ncounter,phistari,mstari,alphai,betai,gammai,h0i,omega_mi,lambda0i,ki,$
   z1,z2,areai,zselpari,nofm,mm,nhist,nhistbin,compcurve='fivesigmalimit.idl'
  if j eq 0 and k eq 0 then begin
    nofm_a=fltarr(nalpha,nmstar,n_elements(nofm))
    mm_a=fltarr(nalpha,nmstar,n_elements(mm))
    nhist_a=fltarr(nalpha,nmstar,n_elements(nhist))
    nhistbin_a=fltarr(nalpha,nmstar,n_elements(nhistbin))
  endif
  nofm_a(j,k,*)=nofm
  mm_a(j,k,*)=mm
  nhist_a(j,k,*)=nhist
  nhistbin_a(j,k,*)=nhistbin
  gridct[j,k]=max(total(nhist,/cum))
  save,file=outroot+'_save.idl',nofm_a,mm_a,nhist_a,nhistbin_a,alphav,mstarv,$
     phistari,h0i,omega_mi,lambda0i,ki,z1,z2,areai,zselpari,gridct,hdr
  writefits,outroot+'.fits',gridct,hdr 

  plot,[0],xr=[20,35],yr=[.01,1d4],/ylog,xtitle='F160W Magnitude',$
     ytitle='Number per bin per area'
  oplot,mm,nofm,line=2
  oplot,nhistbin,nhist,psym=10
  oplot,nhistbin,total(nhist,/cum),psym=10,line=3
  oplot,[1,1]*appmag(zselpari[1],mstari),[1d-10,1d10],line=1,thick=0.5
  oplot,[0,100],[1,1],line=1, thick=0.5
  al_legend,['M!u*!n='+string(mstari,F='(F6.2)'),$
     '!7a!6='+string(alphai,F='(F6.3)')],box=0,/bot,/left
  al_legend,['N(m)dm','N(m,binned)','N(<m,binned)'],linestyle=[2,0,3],box=0,$
     /top,/left
 
endfor

endfor

end
