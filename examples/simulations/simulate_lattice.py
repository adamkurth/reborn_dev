import sys
import time

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

sys.path.append("../..")
import bornagain as ba
import bornagain.simulate.clcore as clcore

# Create a detector
pl = ba.detector.PanelList()
nPix = 1000;
pl.simple_setup(nPix,nPix,100e-6,0.1,1.5e-10)

# Scattering vectors
q = pl.Q
qmag = ba.utils.vecMag(q)

# Atomic coordinates
N = 2
x = np.arange(0,N)*10e-10
[xx,yy,zz] = np.meshgrid(x,x,x,indexing='ij')
r = np.zeros([N**3,3])
r[:,0] = zz.flatten()
r[:,1] = yy.flatten()
r[:,2] = xx.flatten()


# Scattering factors
f = np.ones([N**3], dtype=np.complex64)

# Compute diffraction amplitudes
t = time.time()
A = clcore.phase_factor_qrf(q,r,f)
print(time.time() - t)

# Display diffraction intensities
I = np.abs(A)**2
# Multiply by polarization factor
#I *= pl.polarization_factor

dispim = np.reshape(I,[nPix,nPix])
dispim = np.log10(dispim+0.1)
plt.imshow(dispim,interpolation='nearest',cmap='gray',origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(r[:,0], r[:,1], r[:,2])
# plt.show()
