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
cryst = crystal.structure('../../examples/data/pdb/2LYZ.pdb')

# These are atomic coordinates (Nx3 array)
r = cryst.r

# N = 4
# x = np.arange(0,N)*10e-10
# [xx,yy,zz] = np.meshgrid(x,x,x,indexing='ij')
# r = np.zeros([N**3,3])
# r[:,0] = xx.ravel()
# r[:,1] = yy.ravel()
# r[:,2] = zz.ravel()
# print(r[N,:])

# Look up atomic scattering factors (they are complex numbers)
f = ba.simulate.utils.atomicScatteringFactors(cryst.Z, pl.beam.wavelength)

# f = np.ones([N**3])

# Run the simulation
p = pl[0]
for i in range(0,10):
	t = time.time()
	A = clcore.phaseFactorPAD(r, f, p.T, p.F, p.S, p.B, p.nF, p.nS, p.beam.wavelength)
	tf = time.time() - t
	print('%f seconds' % (tf))

# Display pattern
I = np.abs(A)**2
I = I.reshape((pl[0].nS, pl[0].nF))
plt.imshow(np.log(I+0.1),interpolation='nearest',cmap='gray',origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()