function nm,zi,zf
; for a given apparent magnitude integrate n(m,z) over z
; lets make zi<zf and assume nothing exists above zf
; for total counts zi=0d0
 tol=1d-12 ; default for /double
 z0=zi
 z1=zf
 tempnm=qromo('dnmdz2',z0,z1,/double,/midpnt)
 return,tempnm
end
