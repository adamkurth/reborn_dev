import time
import numpy as np
from reborn.target import density

np.random.seed(0)

shape = np.array([10, 10, 10], dtype=int)
densities = np.random.random(shape).astype(np.float64)
x_min = np.array([0, 0, 0], dtype=np.float64)
x_max = np.array([1, 1, 1], dtype=np.float64)
vecs = np.random.rand(80000, 3)
vals = np.zeros(80000, dtype=np.float64)

t = time.time()
for i in range(100):
    density.trilinear_interpolation(densities, vecs, x_min=x_min, x_max=x_max, out=vals)
print(time.time()-t)
print(vals[0])
