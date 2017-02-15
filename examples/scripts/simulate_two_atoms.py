import sys
sys.path.append("../..")
import time
import numpy as np
import matplotlib.pyplot as plt

import bornagain as ba
import bornagain.simulate.clcore as clcore

# Create a detector
pl = ba.detector.PanelList()
pl.simple_setup(1000,1000,100e-6,0.1,1.5e-10)

# Scattering vectors
Q = pl.Q

# Atomic coordinates
N = 2
r = np.zeros([N,3])
r[1,0] = 20e-10

# Scattering factors
f = np.ones([N])

# Compute diffraction amplitudes
t = time.time()
A = clcore.phaseFactor(Q,r,f)
print(time.time() - t)

# Display diffraction intensities
I = np.abs(A)**2
# Multiply by polarization factor
I *= pl.polarization_factor

I = np.reshape(I,[1000,1000])



plt.imshow(I,interpolation='nearest',cmap='gray',origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()
