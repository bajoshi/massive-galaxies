restore,'/Users/shc/HUDF/XDF/complete/NEW/test1_comptest.idl'
avgcomp,comp,acomp,ecomp,climit,bins=bin,yfit=yfit,/allpar

idx=4
print,climit[idx,*]
xx=findgen(400)/10.+10.
;fudge max to 1 and min to 0
climit[*,2]=1.0d0
climit[*,3]=0.0d0
compstep,xx,climit[idx,*],yy
plot,xx,yy,yr=[-0.1,1.1],xr=[25,35]
climit=reform(climit[idx,*])
save,file='fivesigmalimit.idl',climit

oplot,bin,acomp(*,idx)/max(acomp(*,idx)),psym=5
end
