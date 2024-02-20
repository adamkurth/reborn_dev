;Sabine -Eiger for CXLS simulations

; EIGER 4M (2162, 2068) 

; Camera length (in m) and photon energy (eV)
clen = 0.10 
photon_energy = 7000

; adu_per_photon needs a relatively recent CrystFEL version.  If your version is
; older, change it to adu_per_eV and set it to one over the photon energy in eV

adu_per_photon = 1
res = 13333.33   ; 75 micron pixel size

; These lines describe the data layout for the Eiger native multi-event files
dim0 = %
dim1 = ss
dim2 = fs
data = /entry/data/data


; Mask out strips between panels

; Uncomment these lines if you have a separate bad pixel map (recommended!)
; mask_file = mask-march.h5
; mask = /data/data
; mask_good = 1
; mask_bad = 0

; corner_{x,y} set the position of the corner of the detector (in pixels)
; relative to the beam
panel0/min_fs = 0
panel0/max_fs = 2068
panel0/min_ss = 0
panel0/max_ss = 2162
panel0/corner_x = -1034
panel0/corner_y = -1080
panel0/fs = +1.000000x +0.000000y
panel0/ss = +0.000000x +1.000000y

