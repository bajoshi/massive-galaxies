pro magcat2fluxcat,magcat,outname=outname,parfile=parfile,resfile=resfile,tweakzp=tweakzp

if not keyword_set(outname) then outname='fluxcat.cat'
if not keyword_set(parfile) then parname='filters.param'
fmt='(E12.5)'
;data=read_ascii(magcat,template=ascii_template(magcat))
readfmt,magcat,'(a10000)',line
temp=strsplit(line(0),/extract)
readcol,parfile,filtcol,/silent
filtcol=fix(filtcol)
if keyword_set(resfile) then begin
   zpoff=vega2ab(resfile) 
   zpoff=zpoff(filtcol-1) ; this only uses selected filters
endif else zpoff=0.0
if keyword_set(tweakzp) then begin
  if n_elements(tweakzp) eq n_elements(filtcol) then begin
          zpoff=zpoff+tweakzp
  endif

endif
nmag=(n_elements(temp)-1)/2
if n_elements(filtcol) ne nmag then begin
  stop,"--->Number of filters don't Match in param and cat files"
endif
nobj=n_elements(line)
openw,22,outname
magloc=strarr(nmag)+'F'
magloc=magloc+strcompress(filtcol,/rem)
emagloc=strarr(nmag)+'E'
;emagloc=emagloc+strcompress(indgen(nmag)+1,/rem)
emagloc=emagloc+strcompress(filtcol,/rem)
tok=' '
topline='# id'+tok+strjoin(magloc,tok)+tok+strjoin(emagloc,tok)+tok+'zspec'
printf,22,topline
for j=0,nobj-1 do begin
  temp=strsplit(line(j),/extract)
  id=temp(0)
  mags=temp(1:nmag)+zpoff ; zpoff converts to ab
  mags=float(mags)
  emags=temp(nmag+1:2*nmag)
  emags=float(emags)
  gd= where(mags ge -50 and mags le 50,ngd,complement=bad)
  flux=findgen(nmag)
  eflux=findgen(nmag)
  flux(gd)=10^(-0.4*(mags(gd)+48.6))*1e-7*1e4*1e26*1e6 ;1e7 is erg to joule, 1e4 is cm to m, to JY
  eflux(gd)=(2.5/alog(10.))*emags(gd)*flux(gd)
  if (ngd lt nmag) then begin
   flux(bad)=-99.0
   eflux(bad)=-99.0
  endif 
  newline=id+tok+strjoin(string(flux,f=fmt),tok)+tok+strjoin(string(eflux,f=fmt),tok)+' -1.00'
  printf,22,newline
endfor
close,/all
end
