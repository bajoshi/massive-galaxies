restore,'sedcube.idl'
ageidx=114 ;101 Myr
tauidx=7   ;100 gyr-want big for CSF
flux=flux[*,ageidx,tauidx]
age=age[ageidx]
tau=tau[tauidx]
lamcen=1500.
wid=100
q=where(wave le lamcen+wid and wave gt lamcen-wid,nq)
tran=fltarr(nq)+1.0d0
nu=1./wave
f1500=int_tabulated(nu(q),flux(q)/nu(q),/sort,/double)/int_tabulated(nu(q),tran/nu(q),/sort,/double)
;f1500=flux at 1500 of unredshifted spectrum
;f160=flux at H-band of redshifted spectrum
; generalize to fobs
restore,'F336W_WFC3.idl'
fnu=flux
zz=8.0
zz=2.07
lam=wave*(1.d0+zz)
fnu2=interpol(fnu,lam,wav)
nu2=1./wav
qq=where(fnu2 gt 0.0 and finite(fnu2) eq 1 and thrt gt 1.0e-4 )
yy=fnu2(qq)*thrt(qq)/nu2(qq)
fobs=int_tabulated(nu2(qq),yy,/sort,/double)/int_tabulated(nu2(qq),thrt(qq)/nu2(qq),/sort,/double)

; note this is using hogg eqn 8, but the 1+z term needs to be in
;   denominator because the BC03 is F almbda at this point)
kc=f1500/fobs
kc=-2.5*alog10(kc)
print,kc
ktot=2.5*alog10(zz+1.)+kc
print,ktot


end
