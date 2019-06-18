function dnmdz,z
;for a given apparent mag find n(m,z)
common const1
tempdnmdz=psi(z)*dvdz(z)*zselect(z)
return,tempdnmdz
end
