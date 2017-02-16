import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl

sys.path.append("../..")
import bornagain as ba
import bornagain.simulate.clcore as clcore
import bornagain.target.crystal as crystal

# Create a detector
pl = ba.detector.PanelList()
nPixels = 1001
pixelSize = 100e-6
detectorDistance = 0.05
wavelength = 1.5e-10
pl.simple_setup(nPixels, nPixels, pixelSize, detectorDistance, wavelength)

# Load a crystal structure from pdb file
pdbFile = '../../examples/data/pdb/2LYZ.pdb'  # Lysozyme
# pdbFile = '../../examples/data/pdb/1jb0.pdb'  # Photosystem I
cryst = crystal.structure(pdbFile)

# These are atomic coordinates (Nx3 array)
r = cryst.r

# Look up atomic scattering factors (they are complex numbers)
f = ba.simulate.utils.atomicScatteringFactors(cryst.Z, pl.beam.wavelength)

# Create an opencl context
context = cl.create_some_context()

# This method computes the q vectors on the fly.  Slight speed increase.
if 1:
    p = pl[0]  # p is the first panel in the PanelList (there is only one)
    for i in range(0, 10):
        t = time.time()
        A = clcore.phaseFactorPAD(
            r, f, p.T, p.F, p.S, p.B, p.nF, p.nS, p.beam.wavelength, context=context)
        tf = time.time() - t
        print('phaseFactorPAD: %0.3g seconds/atom/pixel' %
              (tf / p.nF / p.nS / r.shape[0]))
    imdisp = np.abs(A)**2
    imdisp = imdisp.reshape((pl[0].nS, pl[0].nF))
    imdisp = np.log(imdisp + 0.1)

# This method uses any q vectors that you supply.  Here we grab the q vectors from the
# detector.PanelList class.
if 1:
    q = pl.Q  # These are the scattering vectors, Nx3 array.
    for i in range(0, 10):
        t = time.time()
        A = clcore.phase_factor_qrf(q, r, f, context=context)
        tf = time.time() - t
        print('phase_factor_qrf: %0.3g seconds/atom/pixel' %
              (tf / q.shape[0] / r.shape[0]))
    imdisp = np.abs(A)**2
    imdisp = imdisp.reshape((pl[0].nS, pl[0].nF))
    imdisp = np.log(imdisp + 0.1)

# This method involves first making a 3D map of reciprocal-space amplitudes.  We will
# interpolate individual patterns from this map.
if 1:
    res = 1e-10  # Resolution
    qmax = 2 * np.pi / (res)
    qmin = -qmax
    N = 201  # Number of samples
    for i in range(0, 10):
        t = time.time()
        A = clcore.phaseFactor3DM(r, f, N, qmin, qmax, context=context)
        tf = time.time() - t
        print('phaseFactor3DM: %0.3g seconds/atom/pixel' %
              (tf / N**3 / r.shape[0]))
    A = A.reshape([N, N, N])
    imdisp = A[(N - 1) / 2, :, :].reshape([N, N])
    imdisp = np.abs(imdisp)**2
    imdisp = np.log(imdisp + 0.1)


# Display pattern
plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()
