function zselect,z
common const1
P=zselpar
temp=((z-P[1])/P[2])
P[0]=1.0d0/(P[2]*sqrt(2.0d0*!pi))
temp=P[0]*exp(-0.5d0*temp*temp)
return,temp

end
