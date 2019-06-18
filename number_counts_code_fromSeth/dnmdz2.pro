function dnmdz2,z
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

; compute dvdz

one=1d0
z1=one+zz
dist=lumdist(zz,omega_m=omega_m,h0=h0,lambda0=lambda0,/silent)
; using eqn 6 of Gardner 
;factor of 4pi will cancel eqn 8
; omitted 4*pi*c/Ho
top=dist*dist
k=0d0
bottom=z1*z1*sqrt(omega_m*z1*z1*z1+lambda0+k*z1*z1)
ans=top/bottom

;compute selection function

P=zselpar
temp=((z-P[1])/P[2])
P[0]=1.0d0/(P[2]*sqrt(2.0d0*!pi))
temp=P[0]*exp(-0.5d0*temp*temp)


return,lf*ans*temp


end
