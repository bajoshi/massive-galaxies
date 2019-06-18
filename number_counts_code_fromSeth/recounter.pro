pro recounter,nofm,mm,newlimit,nhist,nhistbin,compcurve=compcurve




magstep=0.5d0
npt=float(n_elements(mm))
nsubstep=10.0d0
cts=nofm


; try lazy integration
nbin=npt/nsubstep
;evalstep=magstep*nsubstep
mag0=mm(0)+magstep/2.0
hcts=dblarr(nbin)
bin=dblarr(nbin)
wt=dblarr(n_elements(mm))+1.0d0
if keyword_set(compcurve) then begin
  restore,compcurve
  climit[1]=newlimit
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


nhist=hcts/magstep
nhistbin=bin

;forprint,mm,cts
end ;recounter










