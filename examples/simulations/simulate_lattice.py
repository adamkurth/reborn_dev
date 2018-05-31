import sys
import time

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

sys.path.append("../..")
import bornagain as ba
import bornagain.simulate.clcore as clcore

sim = clcore.ClCore()

# Create a detector
pad = ba.detector.PADGeometry()
n_pixels = 1000
pad.simple_setup(n_pixels=n_pixels, pixel_size=100e-6, distance=0.05)

# Scattering vectors
q = pad.q_vecs(beam_vec=[0, 0, 1], wavelength=1e-10)
qmag = ba.utils.vec_mag(q)

# Atomic coordinates
N = 2
x = np.arange(0, N) * 10e-10
[xx, yy, zz] = np.meshgrid(x, x, x, indexing='ij')
r = np.zeros([N ** 3, 3])
r[:, 0] = zz.flatten()
r[:, 1] = yy.flatten()
r[:, 2] = xx.flatten()

# Scattering factors
f = np.ones([N ** 3])

# Compute diffraction amplitudes
t = time.time()
A = sim.phase_factor_qrf(q, r, f)
print(time.time() - t)

# Display diffraction intensities
I = np.abs(A) ** 2

dispim = np.reshape(I, pad.shape())
dispim = np.log10(dispim + 0.1)
plt.imshow(dispim, interpolation='nearest', cmap='gray', origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(r[:,0], r[:,1], r[:,2])
# plt.show()
