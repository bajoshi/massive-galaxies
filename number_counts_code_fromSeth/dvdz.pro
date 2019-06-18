function dvdz,z
; differential volume element
common const1
zz=z
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
return,ans
end
