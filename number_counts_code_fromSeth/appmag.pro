function appmag,z,absolutemag
; evaluate absolute Mag as a function of z (appmag is fixed)
; uses lumdist
; need to work some factors out
common const1
zz=z
one=1d0
twop5=-2.5d0
five=5d0
; distance needed in PC
dist=lumdist(zz,omega_m=omega_m,h0=h0,lambda0=lambda0,/silent)*1d6
band=twop5*alog10(one+zz) ; see Hogg eqn 13 (BC03 units-see mkfiltcube.pro)
                                ; verified Oct 2013 - sign is positive
                                ;                     and subtract below
                                ; and now I think it gets minus sign like eqn 8    
kcorr=-1.7 ; check all my 1+z's still needed
kcorr=0.0d0
;band=0d0 ; 2.5*alog10(1_z) term? --checked this, I set
;E(z)=2.5*log(1+z)
                                ; and assume filter curve is same at
                                ; both redshifts-observed and rest
; when I did polynomials I put this term in the polynomial
ans=absolutemag+band+kcorr+five*alog10(dist)-five-zpzp
return,ans
end
