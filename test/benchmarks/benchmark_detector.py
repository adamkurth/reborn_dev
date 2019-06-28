import bornagain
from time import time
pad = bornagain.detector.PADGeometry(pixel_size=100e-6, distance=0.05, n_pixels=1000)

n_iter = 10

t0 = time()
for i in range(n_iter):
    a = pad.solid_angles1()
t1 = (time()-t0)/n_iter
print('t1 (solid_angles1):', t1)

t0 = time()
for i in range(n_iter):
    a = pad.solid_angles2()
t2 = (time()-t0)/n_iter
print('t2 (solid_angles2):', t2)

print('t2/t1:', t2/t1)
