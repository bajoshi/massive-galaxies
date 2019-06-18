function psi,z
; compute schechter function in mag units
common const1
zz=z
t1=0d0
t2=0d0
t3=0d0
a=alog(10.0d0)*0.4d0
m=abmag(zz)
;evolve mstar
mstarz=mstar
;mstarz=mstar-2.5d0*beta*alog10(1d0+zz)

t1=-1.0d0*a*(m-mstarz)
t2= (t1*(alpha+1.0d0))-exp(t1) 

lf=a*phistar*exp(t2)
return,lf
end
