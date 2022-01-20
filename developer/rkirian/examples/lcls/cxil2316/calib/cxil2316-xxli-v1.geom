; Optimized panel offsets can be found at the end of the file
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
; Optimized by O.Y. now with correct res
; Optimized by O.Y., corrected distance between panels in ASICs
; Optimized panel offsets can be found at the end of the file
; Manually optimized with hdfsee
; geoptimized again using intor - O.Y.
; geoptimized using intor - O.Y.
; Manually optimized with hdfsee (only quadrants) - O.Y.
; Automatically generated from calibration data

clen =  /LCLS/detector_1/EncoderValue

photon_energy = /LCLS/photon_energy_eV
res = 9097.52
adu_per_eV = 0.00338

data = /entry_1/data_1/data
;mask = /entry_1/data_1/mask
mask_good = 0x0000
;mask_bad = 0xffff
dim0 = %
dim1 = ss
dim2 = fs

; The following lines define "rigid groups" which express the physical
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
q0a0/res = 9090.91
q0a0/fs = +0.007614x +0.999972y
q0a0/ss = -0.999972x +0.007614y
q0a0/corner_x = 442.749
q0a0/corner_y = -28.698

q0a1/min_fs = 194
q0a1/min_ss = 0
q0a1/max_fs = 387
q0a1/max_ss = 184
q0a1/res = 9090.91
q0a1/fs = +0.007614x +0.999972y
q0a1/ss = -0.999972x +0.007614y
q0a1/corner_x = 444.236
q0a1/corner_y = 168.297

q0a2/min_fs = 0
q0a2/min_ss = 185
q0a2/max_fs = 193
q0a2/max_ss = 369
q0a2/res = 9090.91
q0a2/fs = +0.006126x +0.999982y
q0a2/ss = -0.999982x +0.006126y
q0a2/corner_x = 236.648
q0a2/corner_y = -27.2926

q0a3/min_fs = 194
q0a3/min_ss = 185
q0a3/max_fs = 387
q0a3/max_ss = 369
q0a3/res = 9090.91
q0a3/fs = +0.006126x +0.999982y
q0a3/ss = -0.999982x +0.006126y
q0a3/corner_x = 237.942
q0a3/corner_y = 169.703

q0a4/min_fs = 0
q0a4/min_ss = 370
q0a4/max_fs = 193
q0a4/max_ss = 554
q0a4/res = 9090.91
q0a4/fs = -0.999984x +0.005828y
q0a4/ss = -0.005828x -0.999984y
q0a4/corner_x = 869.442
q0a4/corner_y = 363.867

q0a5/min_fs = 194
q0a5/min_ss = 370
q0a5/max_fs = 387
q0a5/max_ss = 554
q0a5/res = 9090.91
q0a5/fs = -0.999984x +0.005828y
q0a5/ss = -0.005828x -0.999984y
q0a5/corner_x = 672.447
q0a5/corner_y = 365.33

q0a6/min_fs = 0
q0a6/min_ss = 555
q0a6/max_fs = 193
q0a6/max_ss = 739
q0a6/res = 9090.91
q0a6/fs = -0.999982x +0.005947y
q0a6/ss = -0.005947x -0.999982y
q0a6/corner_x = 868.24
q0a6/corner_y = 157.171

q0a7/min_fs = 194
q0a7/min_ss = 555
q0a7/max_fs = 387
q0a7/max_ss = 739
q0a7/res = 9090.91
q0a7/fs = -0.999982x +0.005947y
q0a7/ss = -0.005947x -0.999982y
q0a7/corner_x = 671.246
q0a7/corner_y = 158.616

q0a8/min_fs = 0
q0a8/min_ss = 740
q0a8/max_fs = 193
q0a8/max_ss = 924
q0a8/res = 9090.91
q0a8/fs = -0.007798x -0.999971y
q0a8/ss = +0.999971x -0.007798y
q0a8/corner_x = 480.494
q0a8/corner_y = 789.129

q0a9/min_fs = 194
q0a9/min_ss = 740
q0a9/max_fs = 387
q0a9/max_ss = 924
q0a9/res = 9090.91
q0a9/fs = -0.007798x -0.999971y
q0a9/ss = +0.999971x -0.007798y
q0a9/corner_x = 478.952
q0a9/corner_y = 592.135

q0a10/min_fs = 0
q0a10/min_ss = 925
q0a10/max_fs = 193
q0a10/max_ss = 1109
q0a10/res = 9090.91
q0a10/fs = -0.007237x -0.999973y
q0a10/ss = +0.999973x -0.007237y
q0a10/corner_x = 687.323
q0a10/corner_y = 787.244

q0a11/min_fs = 194
q0a11/min_ss = 925
q0a11/max_fs = 387
q0a11/max_ss = 1109
q0a11/res = 9090.91
q0a11/fs = -0.007237x -0.999973y
q0a11/ss = +0.999973x -0.007237y
q0a11/corner_x = 685.823
q0a11/corner_y = 590.25

q0a12/min_fs = 0
q0a12/min_ss = 1110
q0a12/max_fs = 193
q0a12/max_ss = 1294
q0a12/res = 9090.91
q0a12/fs = -1.000001x +0.000063y
q0a12/ss = -0.000063x -1.000001y
q0a12/corner_x = 447.773
q0a12/corner_y = 771.139

q0a13/min_fs = 194
q0a13/min_ss = 1110
q0a13/max_fs = 387
q0a13/max_ss = 1294
q0a13/res = 9090.91
q0a13/fs = -1.000001x +0.000063y
q0a13/ss = -0.000063x -1.000001y
q0a13/corner_x = 250.773
q0a13/corner_y = 771.152

q0a14/min_fs = 0
q0a14/min_ss = 1295
q0a14/max_fs = 193
q0a14/max_ss = 1479
q0a14/res = 9090.91
q0a14/fs = -0.999989x +0.004456y
q0a14/ss = -0.004456x -0.999989y
q0a14/corner_x = 444.603
q0a14/corner_y = 561.604

q0a15/min_fs = 194
q0a15/min_ss = 1295
q0a15/max_fs = 387
q0a15/max_ss = 1479
q0a15/res = 9090.91
q0a15/fs = -0.999989x +0.004456y
q0a15/ss = -0.004456x -0.999989y
q0a15/corner_x = 247.604
q0a15/corner_y = 562.335

q1a0/min_fs = 388
q1a0/min_ss = 0
q1a0/max_fs = 581
q1a0/max_ss = 184
q1a0/res = 9090.91
q1a0/fs = -0.999988x +0.005081y
q1a0/ss = -0.005081x -0.999988y
q1a0/corner_x = 38.1982
q1a0/corner_y = 445.882

q1a1/min_fs = 582
q1a1/min_ss = 0
q1a1/max_fs = 775
q1a1/max_ss = 184
q1a1/res = 9090.91
q1a1/fs = -0.999988x +0.005081y
q1a1/ss = -0.005081x -0.999988y
q1a1/corner_x = -158.799
q1a1/corner_y = 446.889

q1a2/min_fs = 388
q1a2/min_ss = 185
q1a2/max_fs = 581
q1a2/max_ss = 369
q1a2/res = 9090.91
q1a2/fs = -0.999994x +0.003362y
q1a2/ss = -0.003362x -0.999994y
q1a2/corner_x = 37.1274
q1a2/corner_y = 237.588

q1a3/min_fs = 582
q1a3/min_ss = 185
q1a3/max_fs = 775
q1a3/max_ss = 369
q1a3/res = 9090.91
q1a3/fs = -0.999994x +0.003362y
q1a3/ss = -0.003362x -0.999994y
q1a3/corner_x = -159.871
q1a3/corner_y = 238.316

q1a4/min_fs = 388
q1a4/min_ss = 370
q1a4/max_fs = 581
q1a4/max_ss = 554
q1a4/res = 9090.91
q1a4/fs = -0.004649x -0.999989y
q1a4/ss = +0.999989x -0.004649y
q1a4/corner_x = -354.371
q1a4/corner_y = 868.906

q1a5/min_fs = 582
q1a5/min_ss = 370
q1a5/max_fs = 775
q1a5/max_ss = 554
q1a5/res = 9090.91
q1a5/fs = -0.004649x -0.999989y
q1a5/ss = +0.999989x -0.004649y
q1a5/corner_x = -355.324
q1a5/corner_y = 671.908

q1a6/min_fs = 388
q1a6/min_ss = 555
q1a6/max_fs = 581
q1a6/max_ss = 739
q1a6/res = 9090.91
q1a6/fs = -0.005732x -0.999983y
q1a6/ss = +0.999983x -0.005732y
q1a6/corner_x = -146.436
q1a6/corner_y = 868.674

q1a7/min_fs = 582
q1a7/min_ss = 555
q1a7/max_fs = 775
q1a7/max_ss = 739
q1a7/res = 9090.91
q1a7/fs = -0.005732x -0.999983y
q1a7/ss = +0.999983x -0.005732y
q1a7/corner_x = -147.304
q1a7/corner_y = 671.677

q1a8/min_fs = 388
q1a8/min_ss = 740
q1a8/max_fs = 581
q1a8/max_ss = 924
q1a8/res = 9090.91
q1a8/fs = +0.999995x -0.003018y
q1a8/ss = +0.003018x +0.999995y
q1a8/corner_x = -777.422
q1a8/corner_y = 479.132

q1a9/min_fs = 582
q1a9/min_ss = 740
q1a9/max_fs = 775
q1a9/max_ss = 924
q1a9/res = 9090.91
q1a9/fs = +0.999995x -0.003018y
q1a9/ss = +0.003018x +0.999995y
q1a9/corner_x = -580.422
q1a9/corner_y = 478.765

q1a10/min_fs = 388
q1a10/min_ss = 925
q1a10/max_fs = 581
q1a10/max_ss = 1109
q1a10/res = 9090.91
q1a10/fs = +0.999997x -0.002926y
q1a10/ss = +0.002926x +0.999997y
q1a10/corner_x = -778.39
q1a10/corner_y = 686.608

q1a11/min_fs = 582
q1a11/min_ss = 925
q1a11/max_fs = 775
q1a11/max_ss = 1109
q1a11/res = 9090.91
q1a11/fs = +0.999997x -0.002926y
q1a11/ss = +0.002926x +0.999997y
q1a11/corner_x = -581.39
q1a11/corner_y = 686.216

q1a12/min_fs = 388
q1a12/min_ss = 1110
q1a12/max_fs = 581
q1a12/max_ss = 1294
q1a12/res = 9090.91
q1a12/fs = -0.004037x -0.999992y
q1a12/ss = +0.999992x -0.004037y
q1a12/corner_x = -756.631
q1a12/corner_y = 447.189

q1a13/min_fs = 582
q1a13/min_ss = 1110
q1a13/max_fs = 775
q1a13/max_ss = 1294
q1a13/res = 9090.91
q1a13/fs = -0.004037x -0.999992y
q1a13/ss = +0.999992x -0.004037y
q1a13/corner_x = -756.826
q1a13/corner_y = 250.189

q1a14/min_fs = 388
q1a14/min_ss = 1295
q1a14/max_fs = 581
q1a14/max_ss = 1479
q1a14/res = 9090.91
q1a14/fs = -0.004305x -0.999990y
q1a14/ss = +0.999990x -0.004305y
q1a14/corner_x = -549.872
q1a14/corner_y = 446.123

q1a15/min_fs = 582
q1a15/min_ss = 1295
q1a15/max_fs = 775
q1a15/max_ss = 1479
q1a15/res = 9090.91
q1a15/fs = -0.004305x -0.999990y
q1a15/ss = +0.999990x -0.004305y
q1a15/corner_x = -550.722
q1a15/corner_y = 249.125

q2a0/min_fs = 776
q2a0/min_ss = 0
q2a0/max_fs = 969
q2a0/max_ss = 184
q2a0/res = 9090.91
q2a0/fs = +0.001528x -0.999999y
q2a0/ss = +0.999999x +0.001528y
q2a0/corner_x = -437.241
q2a0/corner_y = 37.4285

q2a1/min_fs = 970
q2a1/min_ss = 0
q2a1/max_fs = 1163
q2a1/max_ss = 184
q2a1/res = 9090.91
q2a1/fs = +0.001528x -0.999999y
q2a1/ss = +0.999999x +0.001528y
q2a1/corner_x = -436.965
q2a1/corner_y = -159.572

q2a2/min_fs = 776
q2a2/min_ss = 185
q2a2/max_fs = 969
q2a2/max_ss = 369
q2a2/res = 9090.91
q2a2/fs = -0.002112x -0.999996y
q2a2/ss = +0.999996x -0.002112y
q2a2/corner_x = -229.7
q2a2/corner_y = 37.6912

q2a3/min_fs = 970
q2a3/min_ss = 185
q2a3/max_fs = 1163
q2a3/max_ss = 369
q2a3/res = 9090.91
q2a3/fs = -0.002112x -0.999996y
q2a3/ss = +0.999996x -0.002112y
q2a3/corner_x = -229.97
q2a3/corner_y = -159.309

q2a4/min_fs = 776
q2a4/min_ss = 370
q2a4/max_fs = 969
q2a4/max_ss = 554
q2a4/res = 9090.91
q2a4/fs = +1.000000x -0.000134y
q2a4/ss = +0.000134x +1.000000y
q2a4/corner_x = -861.291
q2a4/corner_y = -351.59

q2a5/min_fs = 970
q2a5/min_ss = 370
q2a5/max_fs = 1163
q2a5/max_ss = 554
q2a5/res = 9090.91
q2a5/fs = +1.000000x -0.000134y
q2a5/ss = +0.000134x +1.000000y
q2a5/corner_x = -664.291
q2a5/corner_y = -351.452

q2a6/min_fs = 776
q2a6/min_ss = 555
q2a6/max_fs = 969
q2a6/max_ss = 739
q2a6/res = 9090.91
q2a6/fs = +0.999999x +0.000168y
q2a6/ss = -0.000168x +0.999999y
q2a6/corner_x = -861.049
q2a6/corner_y = -147.634

q2a7/min_fs = 970
q2a7/min_ss = 555
q2a7/max_fs = 1163
q2a7/max_ss = 739
q2a7/res = 9090.91
q2a7/fs = +0.999999x +0.000168y
q2a7/ss = -0.000168x +0.999999y
q2a7/corner_x = -664.049
q2a7/corner_y = -147.392

q2a8/min_fs = 776
q2a8/min_ss = 740
q2a8/max_fs = 969
q2a8/max_ss = 924
q2a8/res = 9090.91
q2a8/fs = +0.002180x +0.999998y
q2a8/ss = -0.999998x +0.002180y
q2a8/corner_x = -469.894
q2a8/corner_y = -775.506

q2a9/min_fs = 970
q2a9/min_ss = 740
q2a9/max_fs = 1163
q2a9/max_ss = 924
q2a9/res = 9090.91
q2a9/fs = +0.002180x +0.999998y
q2a9/ss = -0.999998x +0.002180y
q2a9/corner_x = -469.585
q2a9/corner_y = -578.506

q2a10/min_fs = 776
q2a10/min_ss = 925
q2a10/max_fs = 969
q2a10/max_ss = 1109
q2a10/res = 9090.91
q2a10/fs = -0.000739x +0.999999y
q2a10/ss = -0.999999x -0.000739y
q2a10/corner_x = -674.087
q2a10/corner_y = -774.794

q2a11/min_fs = 970
q2a11/min_ss = 925
q2a11/max_fs = 1163
q2a11/max_ss = 1109
q2a11/res = 9090.91
q2a11/fs = -0.000739x +0.999999y
q2a11/ss = -0.999999x -0.000739y
q2a11/corner_x = -674.288
q2a11/corner_y = -577.794

q2a12/min_fs = 776
q2a12/min_ss = 1110
q2a12/max_fs = 969
q2a12/max_ss = 1294
q2a12/res = 9090.91
q2a12/fs = +1.000001x +0.000800y
q2a12/ss = -0.000800x +1.000001y
q2a12/corner_x = -434.155
q2a12/corner_y = -755.967

q2a13/min_fs = 970
q2a13/min_ss = 1110
q2a13/max_fs = 1163
q2a13/max_ss = 1294
q2a13/res = 9090.91
q2a13/fs = +1.000001x +0.000800y
q2a13/ss = -0.000800x +1.000001y
q2a13/corner_x = -237.155
q2a13/corner_y = -755.983

q2a14/min_fs = 776
q2a14/min_ss = 1295
q2a14/max_fs = 969
q2a14/max_ss = 1479
q2a14/res = 9090.91
q2a14/fs = +1.000001x -0.000750y
q2a14/ss = +0.000750x +1.000001y
q2a14/corner_x = -435.229
q2a14/corner_y = -549.582

q2a15/min_fs = 970
q2a15/min_ss = 1295
q2a15/max_fs = 1163
q2a15/max_ss = 1479
q2a15/res = 9090.91
q2a15/fs = +1.000001x -0.000750y
q2a15/ss = +0.000750x +1.000001y
q2a15/corner_x = -238.229
q2a15/corner_y = -549.783

q3a0/min_fs = 1164
q3a0/min_ss = 0
q3a0/max_fs = 1357
q3a0/max_ss = 184
q3a0/res = 9090.91
q3a0/fs = +0.999993x -0.003831y
q3a0/ss = +0.003831x +0.999993y
q3a0/corner_x = -32.3436
q3a0/corner_y = -434.921

q3a1/min_fs = 1358
q3a1/min_ss = 0
q3a1/max_fs = 1551
q3a1/max_ss = 184
q3a1/res = 9090.91
q3a1/fs = +0.999993x -0.003831y
q3a1/ss = +0.003831x +0.999993y
q3a1/corner_x = 164.656
q3a1/corner_y = -435.576

q3a2/min_fs = 1164
q3a2/min_ss = 185
q3a2/max_fs = 1357
q3a2/max_ss = 369
q3a2/res = 9090.91
q3a2/fs = +0.999995x -0.003148y
q3a2/ss = +0.003148x +0.999995y
q3a2/corner_x = -31.8413
q3a2/corner_y = -229.822

q3a3/min_fs = 1358
q3a3/min_ss = 185
q3a3/max_fs = 1551
q3a3/max_ss = 369
q3a3/res = 9090.91
q3a3/fs = +0.999995x -0.003148y
q3a3/ss = +0.003148x +0.999995y
q3a3/corner_x = 165.157
q3a3/corner_y = -230.455

q3a4/min_fs = 1164
q3a4/min_ss = 370
q3a4/max_fs = 1357
q3a4/max_ss = 554
q3a4/res = 9090.91
q3a4/fs = +0.005506x +0.999985y
q3a4/ss = -0.999985x +0.005506y
q3a4/corner_x = 359.069
q3a4/corner_y = -860.323

q3a5/min_fs = 1358
q3a5/min_ss = 370
q3a5/max_fs = 1551
q3a5/max_ss = 554
q3a5/res = 9090.91
q3a5/fs = +0.005506x +0.999985y
q3a5/ss = -0.999985x +0.005506y
q3a5/corner_x = 359.59
q3a5/corner_y = -663.324

q3a6/min_fs = 1164
q3a6/min_ss = 555
q3a6/max_fs = 1357
q3a6/max_ss = 739
q3a6/res = 9090.91
q3a6/fs = -0.001559x +0.999997y
q3a6/ss = -0.999997x -0.001559y
q3a6/corner_x = 157.486
q3a6/corner_y = -858.755

q3a7/min_fs = 1358
q3a7/min_ss = 555
q3a7/max_fs = 1551
q3a7/max_ss = 739
q3a7/res = 9090.91
q3a7/fs = -0.001559x +0.999997y
q3a7/ss = -0.999997x -0.001559y
q3a7/corner_x = 156.957
q3a7/corner_y = -661.756

q3a8/min_fs = 1164
q3a8/min_ss = 740
q3a8/max_fs = 1357
q3a8/max_ss = 924
q3a8/res = 9090.91
q3a8/fs = -0.999974x +0.006976y
q3a8/ss = -0.006976x -0.999974y
q3a8/corner_x = 784.933
q3a8/corner_y = -469.71

q3a9/min_fs = 1358
q3a9/min_ss = 740
q3a9/max_fs = 1551
q3a9/max_ss = 924
q3a9/res = 9090.91
q3a9/fs = -0.999974x +0.006976y
q3a9/ss = -0.006976x -0.999974y
q3a9/corner_x = 587.938
q3a9/corner_y = -468.384

q3a10/min_fs = 1164
q3a10/min_ss = 925
q3a10/max_fs = 1357
q3a10/max_ss = 1109
q3a10/res = 9090.91
q3a10/fs = -0.999971x +0.007504y
q3a10/ss = -0.007504x -0.999971y
q3a10/corner_x = 782.205
q3a10/corner_y = -676.988

q3a11/min_fs = 1358
q3a11/min_ss = 925
q3a11/max_fs = 1551
q3a11/max_ss = 1109
q3a11/res = 9090.91
q3a11/fs = -0.999971x +0.007504y
q3a11/ss = -0.007504x -0.999971y
q3a11/corner_x = 585.209
q3a11/corner_y = -675.68

q3a12/min_fs = 1164
q3a12/min_ss = 1110
q3a12/max_fs = 1357
q3a12/max_ss = 1294
q3a12/res = 9090.91
q3a12/fs = -0.000271x +0.999999y
q3a12/ss = -0.999999x -0.000271y
q3a12/corner_x = 763.637
q3a12/corner_y = -436.008

q3a13/min_fs = 1358
q3a13/min_ss = 1110
q3a13/max_fs = 1551
q3a13/max_ss = 1294
q3a13/res = 9090.91
q3a13/fs = -0.000271x +0.999999y
q3a13/ss = -0.999999x -0.000271y
q3a13/corner_x = 763.507
q3a13/corner_y = -239.008

q3a14/min_fs = 1164
q3a14/min_ss = 1295
q3a14/max_fs = 1357
q3a14/max_ss = 1479
q3a14/res = 9090.91
q3a14/fs = +0.002039x +0.999997y
q3a14/ss = -0.999997x +0.002039y
q3a14/corner_x = 555.642
q3a14/corner_y = -436.508

q3a15/min_fs = 1358
q3a15/min_ss = 1295
q3a15/max_fs = 1551
q3a15/max_ss = 1479
q3a15/res = 9090.91
q3a15/fs = +0.002039x +0.999997y
q3a15/ss = -0.999997x +0.002039y
q3a15/corner_x = 556.16
q3a15/corner_y = -239.509




q0a0/coffset = 0.579451
q0a1/coffset = 0.579451
q0a2/coffset = 0.579451
q0a3/coffset = 0.579451
q0a4/coffset = 0.579451
q0a5/coffset = 0.579451
q0a6/coffset = 0.579451
q0a7/coffset = 0.579451
q0a8/coffset = 0.579451
q0a9/coffset = 0.579451
q0a10/coffset = 0.579451
q0a11/coffset = 0.579451
q0a12/coffset = 0.579451
q0a13/coffset = 0.579451
q0a14/coffset = 0.579451
q0a15/coffset = 0.579451
q1a0/coffset = 0.579451
q1a1/coffset = 0.579451
q1a2/coffset = 0.579451
q1a3/coffset = 0.579451
q1a4/coffset = 0.579451
q1a5/coffset = 0.579451
q1a6/coffset = 0.579451
q1a7/coffset = 0.579451
q1a8/coffset = 0.579451
q1a9/coffset = 0.579451
q1a10/coffset = 0.579451
q1a11/coffset = 0.579451
q1a12/coffset = 0.579451
q1a13/coffset = 0.579451
q1a14/coffset = 0.579451
q1a15/coffset = 0.579451
q2a0/coffset = 0.579451
q2a1/coffset = 0.579451
q2a2/coffset = 0.579451
q2a3/coffset = 0.579451
q2a4/coffset = 0.579451
q2a5/coffset = 0.579451
q2a6/coffset = 0.579451
q2a7/coffset = 0.579451
q2a8/coffset = 0.579451
q2a9/coffset = 0.579451
q2a10/coffset = 0.579451
q2a11/coffset = 0.579451
q2a12/coffset = 0.579451
q2a13/coffset = 0.579451
q2a14/coffset = 0.579451
q2a15/coffset = 0.579451
q3a0/coffset = 0.579451
q3a1/coffset = 0.579451
q3a2/coffset = 0.579451
q3a3/coffset = 0.579451
q3a4/coffset = 0.579451
q3a5/coffset = 0.579451
q3a6/coffset = 0.579451
q3a7/coffset = 0.579451
q3a8/coffset = 0.579451
q3a9/coffset = 0.579451
q3a10/coffset = 0.579451
q3a11/coffset = 0.579451
q3a12/coffset = 0.579451
q3a13/coffset = 0.579451
q3a14/coffset = 0.579451
q3a15/coffset = 0.579451
