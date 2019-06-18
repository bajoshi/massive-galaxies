common const1,h0,omega_m,lambda0,k,c,phistar,alpha,mstar,unit,mag,beta,gamma,$
          zpzp,zselpar

qq=findgen(100)/10.
dz=0.4d0
z1=2.1d0-dz*3.
z2=2.1d0+dz*3.
zselpar=dblarr(3)
zselpar[0]=1.0 ; self normalize
zselpar[1]=0.5*(z1+z2)
zselpar[2]=0.5*(z2-z1)

plot,qq,zselect(qq),yr=[0,1]
print,total(zselect(qq))
zselpar[2]=dz
oplot,qq,zselect(qq)
print,total(zselect(qq))

end
