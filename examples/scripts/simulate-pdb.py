import sys
sys.path.append("../..")
import time
import numpy as np
import matplotlib.pyplot as plt

import bornagain as ba
import bornagain.simulate.clcore as clcore
import bornagain.target.crystal as crystal

# Create a detector
pl = ba.detector.panelList()
pl.simpleSetup(1000,1000,100e-6,0.05,1.5e-10)

# Load a crystal structure from pdb file
pdbFile = '../../examples/data/pdb/2LYZ.pdb'  # Lysozyme
pdbFile = '../../examples/data/pdb/1jb0.pdb'  # Photosystem I
cryst = crystal.structure(pdbFile)

# These are atomic coordinates (Nx3 array)
r = cryst.r

# Look up atomic scattering factors (they are complex numbers)
f = ba.simulate.utils.atomicScatteringFactors(cryst.Z, pl.beam.wavelength)

# This method computes the q vectors on the fly.  Slight speed increase.

p = pl[0]  # p is the first panel in the panelList (there is only one)

for i in range(0,10):
	t = time.time()
	A = clcore.phaseFactorPAD(r, f, p.T, p.F, p.S, p.B, p.nF, p.nS, p.beam.wavelength)
	tf = time.time() - t
	print('phaseFactorPAD: %f seconds' % (tf))

# This method uses any q vectors that you supply.  Here we grab the q vectors from the 
# detector panelList class instance.

q = pl.Q

for i in range(0,10):
	t = time.time()
	A = clcore.phaseFactorQRF(q,r,f)
	tf = time.time() - t
	print('phaseFactorQRF: %f seconds' % (tf))

# Display pattern
I = np.abs(A)**2
I = I.reshape((pl[0].nS, pl[0].nF))
plt.imshow(np.log(I+0.1),interpolation='nearest',cmap='gray',origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()