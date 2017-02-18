import sys
sys.path.append("..")
import time
import numpy as np
import matplotlib.pyplot as plt

import bornagain as ba
import bornagain.simulate.clcore as clcore
import bornagain.target.crystal as crystal

# Create a detector
pl = ba.detector.PanelList()
pl.simple_setup(1000,1000,100e-6,0.05,1.5e-10)

# Scattering vectors
Q = pl.Q

# Load a crystal structure from pdb file
cryst = crystal.structure('../examples/data/pdb/2LYZ.pdb')

# These are atomic coordinates
r = cryst.r

# Look up atomic scattering factors
f = ba.simulate.utils.atomicScatteringFactors(cryst.Z, pl.beam.wavelength)

# Run the simulation
p = pl[0]
t = time.time()
A = clcore.phase_factor_pad(r, f, p.T, p.F, p.S, p.B, p.nF, p.nS, p.beam.wavelength)
print('Ran simulation in %f seconds' % (time.time() - t))

# Display pattern
I = np.abs(A)**2
I = I.reshape((pl[0].nS, pl[0].nF))
plt.imshow(np.log(I+1),interpolation='nearest',cmap='gray',origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()