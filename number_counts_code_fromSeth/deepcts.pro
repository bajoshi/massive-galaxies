restore,'grid2_saveall.idl'
magstep=0.5
newlimit=30.1994d0+1.
s=size(mm_a,/dim)
nmstar=s(1)
nalpha=s(0)
npts=s(2)
hdr=headfits('grid2.fits')


for j=0,nalpha-1 do begin
 for k=0,nmstar-1 do begin
    nofm=reform(nofm_a[j,k,*])
    mm=reform(mm_a[j,k,*])
    recounter,nofm,mm,newlimit,nhist,nhistbin,compcurve='fivesigmalimit.idl'
    nhist_a[j,k,*]=nhist
    nhistbin_a[j,k,*]=nhistbin
    gridct[j,k]=max(total(nhist,/cum))
 endfor
endfor
outroot='deeper1mag'
writefits,outroot+'.fits',gridct,hdr

end
