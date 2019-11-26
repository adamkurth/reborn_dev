import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# sys.path.append("../..")
import bornagain as ba
import bornagain.simulate.clcore as clcore

show = True
if 'noplots' in sys.argv:
    show = False

sim = clcore.ClCore(group_size=32, double_precision=False)

# Create a detector
pad = ba.detector.PADGeometry()
pad.simple_setup(shape=(1000, 1000), pixel_size=100e-6, distance=0.05)

beam_vec = [0, 0, 1]

# Scattering vectors
q = pad.q_vecs(beam_vec=beam_vec, wavelength=1e-10)

# Atomic coordinates
r = np.zeros([2, 3])
r[1, 0] = 20e-10

# Scattering factors
f = np.ones([2])

# Compute diffraction amplitudes
t = time.time()
A = sim.phase_factor_qrf(q, r, f)
print(time.time() - t)

# Display diffraction intensities
I = np.abs(A)**2
# Multiply by polarization factor
I *= pad.polarization_factors(beam_vec=beam_vec, polarization_vec=[1, 0, 0])

I = np.reshape(I, pad.shape())

if show:
    plt.imshow(I, interpolation='nearest', cmap='gray', origin='lower')
    plt.title('y: up, x: right, z: beam (towards you)')
    plt.show()
