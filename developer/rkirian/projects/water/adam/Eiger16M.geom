; 08/09/18 zatsepin 
; FMX @ NSLS II August 2018
; Eiger 16M, 12 keV. 

; Camera length (in m) and photon energy (eV)
clen = 0.135
photon_energy = 5000

; adu_per_photon needs a relatively recent CrystFEL version.  If your version is
; older, change it to adu_per_eV and set it to one over the photon energy in eV

adu_per_photon = 1
res = 13333.3   ; 75 micron pixel size

; These lines describe the data layout for the Eiger native multi-event files
dim0 = %
dim1 = ss
dim2 = fs
data = /entry/data/data


; Mask out strips between panels
bad_v0/min_fs = 1030
bad_v0/max_fs = 1039
bad_v0/min_ss = 0
bad_v0/max_ss = 4370

bad_v1/min_fs = 2070
bad_v1/max_fs = 2079
bad_v1/min_ss = 0
bad_v1/max_ss = 4370

bad_v2/min_fs = 3110
bad_v2/max_fs = 3119 
bad_v2/min_ss = 0
bad_v2/max_ss = 4370

bad_h0/min_fs = 0
bad_h0/max_fs = 4149
bad_h0/min_ss = 514
bad_h0/max_ss = 550

bad_h1/min_fs = 0
bad_h1/max_fs = 4149
bad_h1/min_ss = 1065
bad_h1/max_ss = 1101

bad_h2/min_fs = 0
bad_h2/max_fs = 4149
bad_h2/min_ss = 1616
bad_h2/max_ss = 1652

bad_h3/min_fs = 0
bad_h3/max_fs = 4149
bad_h3/min_ss = 2167
bad_h3/max_ss = 2203

bad_h4/min_fs = 0
bad_h4/max_fs = 4149
bad_h4/min_ss = 2718
bad_h4/max_ss = 2754

bad_h5/min_fs = 0
bad_h5/max_fs = 4149
bad_h5/min_ss = 3269
bad_h5/max_ss = 3305

bad_h6/min_fs = 0
bad_h6/max_fs = 4149
bad_h6/min_ss = 3820
bad_h6/max_ss = 3856

bad_bs/min_fs = 1880
bad_bs/max_fs = 2100
bad_bs/min_ss = 2200
bad_bs/max_ss = 2350

bad_bp1/min_fs = 1963
bad_bp1/max_fs = 1964
bad_bp1/min_ss = 600
bad_bp1/max_ss = 601

bad_bp2/min_fs = 3931
bad_bp2/max_fs = 3932  
bad_bp2/min_ss = 1201   
bad_bp2/max_ss = 1202

bad_bp3/min_fs = 1606
bad_bp3/max_fs = 1607
bad_bp3/min_ss = 1740
bad_bp3/max_ss = 1741

bad_bp4/min_fs = 1180
bad_bp4/max_fs = 1181
bad_bp4/min_ss = 1884
bad_bp4/max_ss = 1885

bad_bp5/min_fs = 1542
bad_bp5/max_fs = 1543
bad_bp5/min_ss = 2664
bad_bp5/max_ss = 2665

bad_bp6/min_fs = 1934
bad_bp6/max_fs = 1935
bad_bp6/min_ss = 2889
bad_bp6/max_ss = 2890

bad_bp7/min_fs = 1656
bad_bp7/max_fs = 1657
bad_bp7/min_ss = 3239
bad_bp7/max_ss = 3240

bad_bp8/min_fs = 1200
bad_bp8/max_fs = 1201
bad_bp8/min_ss = 3382
bad_bp8/max_ss = 3383

bad_bp9/min_fs = 1692
bad_bp9/max_fs = 1693
bad_bp9/min_ss = 4085
bad_bp9/max_ss = 4086


; Uncomment these lines if you have a separate bad pixel map (recommended!)
mask_file = mask_comb_V2.h5
mask = /data/data
mask_good = 1
mask_bad = 0

; corner_{x,y} set the position of the corner of the detector (in pixels)
; relative to the beam
panel0/min_fs = 0
panel0/max_fs = 4149
panel0/min_ss = 0
panel0/max_ss = 4370
panel0/corner_x = -2000.919900
panel0/corner_y = -2249.374056
panel0/fs = +1.000000x +0.000000y
panel0/ss = +0.000000x +1.000000y

