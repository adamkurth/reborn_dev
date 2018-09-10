; Twice geoptimized by O.Y. using KP indexed data
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Copied from cxil2316-oy-v4-predrefine.geom (which was 
; shifted with lys from Cherezov beamtime, Aug 2016.)
; Refined with phyco using only 95mm distance data (O.Y.)
; Resolution changed back to 9097.52, geoptimized by O.Y.
; Resolution changed back to 9090.9, distance adjusted based on 2 measurements
; Twice optimized using merge3d by O.Y.
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Starting geometry for CSPAD in DS1 for cxil2316 - Fromme PS2 August 2016.
; Optimized panel offsets can be found at the end of the file
; Refined using intor by O.Y.
; Optimized panel offsets can be found at the end of the file
; Copied refined-cxil0216.geom without changes, to cxim8916-nz1.geom for zatsepin16. june 17 2016.
; 0.582 showed best indexing rate & tightest, symmetric unit cell distributions closest to 
; PDB values (using 0.578 gave a = 75 A which is too small); and smallest detector-shift.
; this geometry.
; optimized using intor by O.Y.
; Manually optimized with hdfsee
; Optimized by O.Y., corrected distance between panels in ASICs
; Optimized panel offsets can be found at the end of the file
; Manually optimized with hdfsee
; geoptimized again using intor - O.Y.
; geoptimized using intor - O.Y.
; Manually optimized with hdfsee (only quadrants) - O.Y.
; Automatically generated from calibration data

clen =  /LCLS/detector_1/EncoderValue
coffset = 0.580
res = 9097.52

photon_energy = /LCLS/photon_energy_eV
adu_per_eV = 0.00338

data = /entry_1/data_1/data
;mask = /entry_1/data_1/mask
mask_good = 0x0000
;mask_bad = 0xffff
dim0 = %
dim1 = ss
dim2 = fs

; construction of the detector.  This is used when refining the detector
; geometry.

rigid_group_q0 = q0a0,q0a1,q0a2,q0a3,q0a4,q0a5,q0a6,q0a7,q0a8,q0a9,q0a10,q0a11,q0a12,q0a13,q0a14,q0a15
rigid_group_q1 = q1a0,q1a1,q1a2,q1a3,q1a4,q1a5,q1a6,q1a7,q1a8,q1a9,q1a10,q1a11,q1a12,q1a13,q1a14,q1a15
rigid_group_q2 = q2a0,q2a1,q2a2,q2a3,q2a4,q2a5,q2a6,q2a7,q2a8,q2a9,q2a10,q2a11,q2a12,q2a13,q2a14,q2a15
rigid_group_q3 = q3a0,q3a1,q3a2,q3a3,q3a4,q3a5,q3a6,q3a7,q3a8,q3a9,q3a10,q3a11,q3a12,q3a13,q3a14,q3a15

rigid_group_a0 = q0a0,q0a1
rigid_group_a1 = q0a2,q0a3
rigid_group_a2 = q0a4,q0a5
rigid_group_a3 = q0a6,q0a7
rigid_group_a4 = q0a8,q0a9
rigid_group_a5 = q0a10,q0a11
rigid_group_a6 = q0a12,q0a13
rigid_group_a7 = q0a14,q0a15
rigid_group_a8 = q1a0,q1a1
rigid_group_a9 = q1a2,q1a3
rigid_group_a10 = q1a4,q1a5
rigid_group_a11 = q1a6,q1a7
rigid_group_a12 = q1a8,q1a9
rigid_group_a13 = q1a10,q1a11
rigid_group_a14 = q1a12,q1a13
rigid_group_a15 = q1a14,q1a15
rigid_group_a16 = q2a0,q2a1
rigid_group_a17 = q2a2,q2a3
rigid_group_a18 = q2a4,q2a5
rigid_group_a19 = q2a6,q2a7
rigid_group_a20 = q2a8,q2a9
rigid_group_a21 = q2a10,q2a11
rigid_group_a22 = q2a12,q2a13
rigid_group_a23 = q2a14,q2a15
rigid_group_a24 = q3a0,q3a1
rigid_group_a25 = q3a2,q3a3
rigid_group_a26 = q3a4,q3a5
rigid_group_a27 = q3a6,q3a7
rigid_group_a28 = q3a8,q3a9
rigid_group_a29 = q3a10,q3a11
rigid_group_a30 = q3a12,q3a13
rigid_group_a31 = q3a14,q3a15

rigid_group_collection_quadrants = q0,q1,q2,q3
rigid_group_collection_asics = a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31

q0a0/min_fs = 0
q0a0/min_ss = 0
q0a0/max_fs = 193
q0a0/max_ss = 184
q0a0/res = 9097.52
q0a0/fs = +0.006578x +0.999978y
q0a0/ss = -0.999978x +0.006578y
q0a0/corner_x = 441.206
q0a0/corner_y = -24.8431

q0a1/min_fs = 194
q0a1/min_ss = 0
q0a1/max_fs = 387
q0a1/max_ss = 184
q0a1/res = 9097.52
q0a1/fs = +0.006578x +0.999978y
q0a1/ss = -0.999978x +0.006578y
q0a1/corner_x = 442.502
q0a1/corner_y = 172.153

q0a2/min_fs = 0
q0a2/min_ss = 185
q0a2/max_fs = 193
q0a2/max_ss = 369
q0a2/res = 9097.52
q0a2/fs = +0.004803x +0.999989y
q0a2/ss = -0.999989x +0.004803y
q0a2/corner_x = 234.838
q0a2/corner_y = -24.225

q0a3/min_fs = 194
q0a3/min_ss = 185
q0a3/max_fs = 387
q0a3/max_ss = 369
q0a3/res = 9097.52
q0a3/fs = +0.004803x +0.999989y
q0a3/ss = -0.999989x +0.004803y
q0a3/corner_x = 235.784
q0a3/corner_y = 172.773

q0a4/min_fs = 0
q0a4/min_ss = 370
q0a4/max_fs = 193
q0a4/max_ss = 554
q0a4/res = 9097.52
q0a4/fs = -0.999989x +0.004897y
q0a4/ss = -0.004897x -0.999989y
q0a4/corner_x = 867.032
q0a4/corner_y = 368.103

q0a5/min_fs = 194
q0a5/min_ss = 370
q0a5/max_fs = 387
q0a5/max_ss = 554
q0a5/res = 9097.52
q0a5/fs = -0.999989x +0.004897y
q0a5/ss = -0.004897x -0.999989y
q0a5/corner_x = 670.034
q0a5/corner_y = 369.067

q0a6/min_fs = 0
q0a6/min_ss = 555
q0a6/max_fs = 193
q0a6/max_ss = 739
q0a6/res = 9097.52
q0a6/fs = -0.999993x +0.003676y
q0a6/ss = -0.003676x -0.999993y
q0a6/corner_x = 866.084
q0a6/corner_y = 162.183

q0a7/min_fs = 194
q0a7/min_ss = 555
q0a7/max_fs = 387
q0a7/max_ss = 739
q0a7/res = 9097.52
q0a7/fs = -0.999993x +0.003676y
q0a7/ss = -0.003676x -0.999993y
q0a7/corner_x = 669.086
q0a7/corner_y = 162.907

q0a8/min_fs = 0
q0a8/min_ss = 740
q0a8/max_fs = 193
q0a8/max_ss = 924
q0a8/res = 9097.52
q0a8/fs = -0.006197x -0.999981y
q0a8/ss = +0.999981x -0.006197y
q0a8/corner_x = 476.976
q0a8/corner_y = 792.343

q0a9/min_fs = 194
q0a9/min_ss = 740
q0a9/max_fs = 387
q0a9/max_ss = 924
q0a9/res = 9097.52
q0a9/fs = -0.006197x -0.999981y
q0a9/ss = +0.999981x -0.006197y
q0a9/corner_x = 475.755
q0a9/corner_y = 595.346

q0a10/min_fs = 0
q0a10/min_ss = 925
q0a10/max_fs = 193
q0a10/max_ss = 1109
q0a10/res = 9097.52
q0a10/fs = -0.007141x -0.999975y
q0a10/ss = +0.999975x -0.007141y
q0a10/corner_x = 686.341
q0a10/corner_y = 792.429

q0a11/min_fs = 194
q0a11/min_ss = 925
q0a11/max_fs = 387
q0a11/max_ss = 1109
q0a11/res = 9097.52
q0a11/fs = -0.007141x -0.999975y
q0a11/ss = +0.999975x -0.007141y
q0a11/corner_x = 684.934
q0a11/corner_y = 595.434

q0a12/min_fs = 0
q0a12/min_ss = 1110
q0a12/max_fs = 193
q0a12/max_ss = 1294
q0a12/res = 9097.52
q0a12/fs = -1.000000x -0.001414y
q0a12/ss = +0.001414x -1.000000y
q0a12/corner_x = 445.431
q0a12/corner_y = 774.528

q0a13/min_fs = 194
q0a13/min_ss = 1110
q0a13/max_fs = 387
q0a13/max_ss = 1294
q0a13/res = 9097.52
q0a13/fs = -1.000000x -0.001414y
q0a13/ss = +0.001414x -1.000000y
q0a13/corner_x = 248.431
q0a13/corner_y = 774.249

q0a14/min_fs = 0
q0a14/min_ss = 1295
q0a14/max_fs = 193
q0a14/max_ss = 1479
q0a14/res = 9097.52
q0a14/fs = -0.999995x +0.003265y
q0a14/ss = -0.003265x -0.999995y
q0a14/corner_x = 442.379
q0a14/corner_y = 564.914

q0a15/min_fs = 194
q0a15/min_ss = 1295
q0a15/max_fs = 387
q0a15/max_ss = 1479
q0a15/res = 9097.52
q0a15/fs = -0.999995x +0.003265y
q0a15/ss = -0.003265x -0.999995y
q0a15/corner_x = 245.38
q0a15/corner_y = 565.557

q1a0/min_fs = 388
q1a0/min_ss = 0
q1a0/max_fs = 581
q1a0/max_ss = 184
q1a0/res = 9097.52
q1a0/fs = -0.999988x +0.004894y
q1a0/ss = -0.004894x -0.999988y
q1a0/corner_x = 35.6605
q1a0/corner_y = 448.56

q1a1/min_fs = 582
q1a1/min_ss = 0
q1a1/max_fs = 775
q1a1/max_ss = 184
q1a1/res = 9097.52
q1a1/fs = -0.999988x +0.004894y
q1a1/ss = -0.004894x -0.999988y
q1a1/corner_x = -161.337
q1a1/corner_y = 449.524

q1a2/min_fs = 388
q1a2/min_ss = 185
q1a2/max_fs = 581
q1a2/max_ss = 369
q1a2/res = 9097.52
q1a2/fs = -0.999996x +0.002256y
q1a2/ss = -0.002256x -0.999996y
q1a2/corner_x = 34.7429
q1a2/corner_y = 240.498

q1a3/min_fs = 582
q1a3/min_ss = 185
q1a3/max_fs = 775
q1a3/max_ss = 369
q1a3/res = 9097.52
q1a3/fs = -0.999996x +0.002256y
q1a3/ss = -0.002256x -0.999996y
q1a3/corner_x = -162.256
q1a3/corner_y = 240.943

q1a4/min_fs = 388
q1a4/min_ss = 370
q1a4/max_fs = 581
q1a4/max_ss = 554
q1a4/res = 9097.52
q1a4/fs = -0.004389x -0.999989y
q1a4/ss = +0.999989x -0.004389y
q1a4/corner_x = -356.774
q1a4/corner_y = 871.586

q1a5/min_fs = 582
q1a5/min_ss = 370
q1a5/max_fs = 775
q1a5/max_ss = 554
q1a5/res = 9097.52
q1a5/fs = -0.004389x -0.999989y
q1a5/ss = +0.999989x -0.004389y
q1a5/corner_x = -357.639
q1a5/corner_y = 674.589

q1a6/min_fs = 388
q1a6/min_ss = 555
q1a6/max_fs = 581
q1a6/max_ss = 739
q1a6/res = 9097.52
q1a6/fs = -0.003293x -0.999993y
q1a6/ss = +0.999993x -0.003293y
q1a6/corner_x = -149.106
q1a6/corner_y = 870.962

q1a7/min_fs = 582
q1a7/min_ss = 555
q1a7/max_fs = 775
q1a7/max_ss = 739
q1a7/res = 9097.52
q1a7/fs = -0.003293x -0.999993y
q1a7/ss = +0.999993x -0.003293y
q1a7/corner_x = -149.755
q1a7/corner_y = 673.964

q1a8/min_fs = 388
q1a8/min_ss = 740
q1a8/max_fs = 581
q1a8/max_ss = 924
q1a8/res = 9097.52
q1a8/fs = +0.999995x -0.002865y
q1a8/ss = +0.002865x +0.999995y
q1a8/corner_x = -780.324
q1a8/corner_y = 481.88

q1a9/min_fs = 582
q1a9/min_ss = 740
q1a9/max_fs = 775
q1a9/max_ss = 924
q1a9/res = 9097.52
q1a9/fs = +0.999995x -0.002865y
q1a9/ss = +0.002865x +0.999995y
q1a9/corner_x = -583.325
q1a9/corner_y = 481.316

q1a10/min_fs = 388
q1a10/min_ss = 925
q1a10/max_fs = 581
q1a10/max_ss = 1109
q1a10/res = 9097.52
q1a10/fs = +0.999994x -0.003560y
q1a10/ss = +0.003560x +0.999994y
q1a10/corner_x = -780.349
q1a10/corner_y = 690.994

q1a11/min_fs = 582
q1a11/min_ss = 925
q1a11/max_fs = 775
q1a11/max_ss = 1109
q1a11/res = 9097.52
q1a11/fs = +0.999994x -0.003560y
q1a11/ss = +0.003560x +0.999994y
q1a11/corner_x = -583.35
q1a11/corner_y = 690.293

q1a12/min_fs = 388
q1a12/min_ss = 1110
q1a12/max_fs = 581
q1a12/max_ss = 1294
q1a12/res = 9097.52
q1a12/fs = -0.002473x -0.999996y
q1a12/ss = +0.999996x -0.002473y
q1a12/corner_x = -759.862
q1a12/corner_y = 449.46

q1a13/min_fs = 582
q1a13/min_ss = 1110
q1a13/max_fs = 775
q1a13/max_ss = 1294
q1a13/res = 9097.52
q1a13/fs = -0.002473x -0.999996y
q1a13/ss = +0.999996x -0.002473y
q1a13/corner_x = -760.349
q1a13/corner_y = 252.461

q1a14/min_fs = 388
q1a14/min_ss = 1295
q1a14/max_fs = 581
q1a14/max_ss = 1479
q1a14/res = 9097.52
q1a14/fs = -0.004582x -0.999989y
q1a14/ss = +0.999989x -0.004582y
q1a14/corner_x = -552.043
q1a14/corner_y = 449.105

q1a15/min_fs = 582
q1a15/min_ss = 1295
q1a15/max_fs = 775
q1a15/max_ss = 1479
q1a15/res = 9097.52
q1a15/fs = -0.004582x -0.999989y
q1a15/ss = +0.999989x -0.004582y
q1a15/corner_x = -552.946
q1a15/corner_y = 252.107

q2a0/min_fs = 776
q2a0/min_ss = 0
q2a0/max_fs = 969
q2a0/max_ss = 184
q2a0/res = 9097.52
q2a0/fs = +0.002374x -0.999998y
q2a0/ss = +0.999998x +0.002374y
q2a0/corner_x = -439.413
q2a0/corner_y = 40.5578

q2a1/min_fs = 970
q2a1/min_ss = 0
q2a1/max_fs = 1163
q2a1/max_ss = 184
q2a1/res = 9097.52
q2a1/fs = +0.002374x -0.999998y
q2a1/ss = +0.999998x +0.002374y
q2a1/corner_x = -438.945
q2a1/corner_y = -156.442

q2a2/min_fs = 776
q2a2/min_ss = 185
q2a2/max_fs = 969
q2a2/max_ss = 369
q2a2/res = 9097.52
q2a2/fs = -0.001071x -0.999998y
q2a2/ss = +0.999998x -0.001071y
q2a2/corner_x = -232.02
q2a2/corner_y = 40.2853

q2a3/min_fs = 970
q2a3/min_ss = 185
q2a3/max_fs = 1163
q2a3/max_ss = 369
q2a3/res = 9097.52
q2a3/fs = -0.001071x -0.999998y
q2a3/ss = +0.999998x -0.001071y
q2a3/corner_x = -232.231
q2a3/corner_y = -156.714

q2a4/min_fs = 776
q2a4/min_ss = 370
q2a4/max_fs = 969
q2a4/max_ss = 554
q2a4/res = 9097.52
q2a4/fs = +1.000000x -0.000603y
q2a4/ss = +0.000603x +1.000000y
q2a4/corner_x = -863.648
q2a4/corner_y = -347.803

q2a5/min_fs = 970
q2a5/min_ss = 370
q2a5/max_fs = 1163
q2a5/max_ss = 554
q2a5/res = 9097.52
q2a5/fs = +1.000000x -0.000603y
q2a5/ss = +0.000603x +1.000000y
q2a5/corner_x = -666.648
q2a5/corner_y = -347.922

q2a6/min_fs = 776
q2a6/min_ss = 555
q2a6/max_fs = 969
q2a6/max_ss = 739
q2a6/res = 9097.52
q2a6/fs = +0.999998x +0.000681y
q2a6/ss = -0.000681x +0.999998y
q2a6/corner_x = -863.487
q2a6/corner_y = -144.548

q2a7/min_fs = 970
q2a7/min_ss = 555
q2a7/max_fs = 1163
q2a7/max_ss = 739
q2a7/res = 9097.52
q2a7/fs = +0.999998x +0.000681y
q2a7/ss = -0.000681x +0.999998y
q2a7/corner_x = -666.488
q2a7/corner_y = -144.413

q2a8/min_fs = 776
q2a8/min_ss = 740
q2a8/max_fs = 969
q2a8/max_ss = 924
q2a8/res = 9097.52
q2a8/fs = +0.001587x +0.999998y
q2a8/ss = -0.999998x +0.001587y
q2a8/corner_x = -472.553
q2a8/corner_y = -772.277

q2a9/min_fs = 970
q2a9/min_ss = 740
q2a9/max_fs = 1163
q2a9/max_ss = 924
q2a9/res = 9097.52
q2a9/fs = +0.001587x +0.999998y
q2a9/ss = -0.999998x +0.001587y
q2a9/corner_x = -472.24
q2a9/corner_y = -575.278

q2a10/min_fs = 776
q2a10/min_ss = 925
q2a10/max_fs = 969
q2a10/max_ss = 1109
q2a10/res = 9097.52
q2a10/fs = +0.002220x +0.999997y
q2a10/ss = -0.999997x +0.002220y
q2a10/corner_x = -677.744
q2a10/corner_y = -771.288

q2a11/min_fs = 970
q2a11/min_ss = 925
q2a11/max_fs = 1163
q2a11/max_ss = 1109
q2a11/res = 9097.52
q2a11/fs = +0.002220x +0.999997y
q2a11/ss = -0.999997x +0.002220y
q2a11/corner_x = -677.307
q2a11/corner_y = -574.288

q2a12/min_fs = 776
q2a12/min_ss = 1110
q2a12/max_fs = 969
q2a12/max_ss = 1294
q2a12/res = 9097.52
q2a12/fs = +1.000001x +0.001031y
q2a12/ss = -0.001031x +1.000001y
q2a12/corner_x = -435.239
q2a12/corner_y = -752.696

q2a13/min_fs = 970
q2a13/min_ss = 1110
q2a13/max_fs = 1163
q2a13/max_ss = 1294
q2a13/res = 9097.52
q2a13/fs = +1.000001x +0.001031y
q2a13/ss = -0.001031x +1.000001y
q2a13/corner_x = -238.239
q2a13/corner_y = -752.493

q2a14/min_fs = 776
q2a14/min_ss = 1295
q2a14/max_fs = 969
q2a14/max_ss = 1479
q2a14/res = 9097.52
q2a14/fs = +1.000002x -0.000144y
q2a14/ss = +0.000144x +1.000002y
q2a14/corner_x = -436.815
q2a14/corner_y = -546.655

q2a15/min_fs = 970
q2a15/min_ss = 1295
q2a15/max_fs = 1163
q2a15/max_ss = 1479
q2a15/res = 9097.52
q2a15/fs = +1.000002x -0.000144y
q2a15/ss = +0.000144x +1.000002y
q2a15/corner_x = -239.814
q2a15/corner_y = -546.683

q3a0/min_fs = 1164
q3a0/min_ss = 0
q3a0/max_fs = 1357
q3a0/max_ss = 184
q3a0/res = 9097.52
q3a0/fs = +1.000000x -0.000914y
q3a0/ss = +0.000914x +1.000000y
q3a0/corner_x = -33.6894
q3a0/corner_y = -432.482

q3a1/min_fs = 1358
q3a1/min_ss = 0
q3a1/max_fs = 1551
q3a1/max_ss = 184
q3a1/res = 9097.52
q3a1/fs = +1.000000x -0.000914y
q3a1/ss = +0.000914x +1.000000y
q3a1/corner_x = 163.31
q3a1/corner_y = -432.662

q3a2/min_fs = 1164
q3a2/min_ss = 185
q3a2/max_fs = 1357
q3a2/max_ss = 369
q3a2/res = 9097.52
q3a2/fs = +0.999999x -0.001691y
q3a2/ss = +0.001691x +0.999999y
q3a2/corner_x = -33.464
q3a2/corner_y = -227.243

q3a3/min_fs = 1358
q3a3/min_ss = 185
q3a3/max_fs = 1551
q3a3/max_ss = 369
q3a3/res = 9097.52
q3a3/fs = +0.999999x -0.001691y
q3a3/ss = +0.001691x +0.999999y
q3a3/corner_x = 163.536
q3a3/corner_y = -227.576

q3a4/min_fs = 1164
q3a4/min_ss = 370
q3a4/max_fs = 1357
q3a4/max_ss = 554
q3a4/res = 9097.52
q3a4/fs = +0.004965x +0.999988y
q3a4/ss = -0.999988x +0.004965y
q3a4/corner_x = 357.346
q3a4/corner_y = -856.194

q3a5/min_fs = 1358
q3a5/min_ss = 370
q3a5/max_fs = 1551
q3a5/max_ss = 554
q3a5/res = 9097.52
q3a5/fs = +0.004965x +0.999988y
q3a5/ss = -0.999988x +0.004965y
q3a5/corner_x = 358.324
q3a5/corner_y = -659.196

q3a6/min_fs = 1164
q3a6/min_ss = 555
q3a6/max_fs = 1357
q3a6/max_ss = 739
q3a6/res = 9097.52
q3a6/fs = -0.000189x +0.999999y
q3a6/ss = -0.999999x -0.000189y
q3a6/corner_x = 155.836
q3a6/corner_y = -855.238

q3a7/min_fs = 1358
q3a7/min_ss = 555
q3a7/max_fs = 1551
q3a7/max_ss = 739
q3a7/res = 9097.52
q3a7/fs = -0.000189x +0.999999y
q3a7/ss = -0.999999x -0.000189y
q3a7/corner_x = 155.798
q3a7/corner_y = -658.238

q3a8/min_fs = 1164
q3a8/min_ss = 740
q3a8/max_fs = 1357
q3a8/max_ss = 924
q3a8/res = 9097.52
q3a8/fs = -0.999979x +0.006193y
q3a8/ss = -0.006193x -0.999979y
q3a8/corner_x = 782.641
q3a8/corner_y = -465.548

q3a9/min_fs = 1358
q3a9/min_ss = 740
q3a9/max_fs = 1551
q3a9/max_ss = 924
q3a9/res = 9097.52
q3a9/fs = -0.999979x +0.006193y
q3a9/ss = -0.006193x -0.999979y
q3a9/corner_x = 585.646
q3a9/corner_y = -464.328

q3a10/min_fs = 1164
q3a10/min_ss = 925
q3a10/max_fs = 1357
q3a10/max_ss = 1109
q3a10/res = 9097.52
q3a10/fs = -0.999970x +0.007470y
q3a10/ss = -0.007470x -0.999970y
q3a10/corner_x = 781.027
q3a10/corner_y = -671.869

q3a11/min_fs = 1358
q3a11/min_ss = 925
q3a11/max_fs = 1551
q3a11/max_ss = 1109
q3a11/res = 9097.52
q3a11/fs = -0.999970x +0.007470y
q3a11/ss = -0.007470x -0.999970y
q3a11/corner_x = 584.033
q3a11/corner_y = -670.398

q3a12/min_fs = 1164
q3a12/min_ss = 1110
q3a12/max_fs = 1357
q3a12/max_ss = 1294
q3a12/res = 9097.52
q3a12/fs = +0.005700x +0.999984y
q3a12/ss = -0.999984x +0.005700y
q3a12/corner_x = 760.708
q3a12/corner_y = -432.511

q3a13/min_fs = 1358
q3a13/min_ss = 1110
q3a13/max_fs = 1551
q3a13/max_ss = 1294
q3a13/res = 9097.52
q3a13/fs = +0.005700x +0.999984y
q3a13/ss = -0.999984x +0.005700y
q3a13/corner_x = 761.83
q3a13/corner_y = -235.514

q3a14/min_fs = 1164
q3a14/min_ss = 1295
q3a14/max_fs = 1357
q3a14/max_ss = 1479
q3a14/res = 9097.52
q3a14/fs = +0.000638x +0.999999y
q3a14/ss = -0.999999x +0.000638y
q3a14/corner_x = 554.418
q3a14/corner_y = -432.332

q3a15/min_fs = 1358
q3a15/min_ss = 1295
q3a15/max_fs = 1551
q3a15/max_ss = 1479
q3a15/res = 9097.52
q3a15/fs = +0.000638x +0.999999y
q3a15/ss = -0.999999x +0.000638y
q3a15/corner_x = 554.544
q3a15/corner_y = -235.332
