import numpy as np
import peaker
print peaker.peaker.__doc__

nx = 10
ny = 10
p = np.ones((nx,ny),dtype=np.float64,order='F')
print p
peaker.peaker.squarediff(p,3,3,2,2)
print p
