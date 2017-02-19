import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import pyopencl.array

sys.path.append("../../..")
import bornagain as ba
import bornagain.simulate.clcore as clcore
import bornagain.target.crystal as crystal

# Create a detector
pl = ba.detector.PanelList()
nPixels = 1001
pixelSize = 100e-6
detectorDistance = 0.05
wavelength = 1.5e-10
pl.simple_setup(nPixels, nPixels+1, pixelSize, detectorDistance, wavelength)
q = pl[0].Q

# Load a crystal structure from pdb file
pdbFile = '../../data/pdb/2LYZ.pdb'  # Lysozyme
#pdbFile = '../../data/pdb/1jb0.pdb'  # Photosystem I
cryst = crystal.structure(pdbFile)

# These are atomic coordinates (Nx3 array)
r = cryst.r

# Look up atomic scattering factors (they are complex numbers)
#f = ba.simulate.utils.atomicScatteringFactors(cryst.Z, pl.beam.wavelength)
f = ba.simulate.atoms.get_scattering_factors(cryst.Z,ba.units.hc/pl.beam.wavelength)

# Create an opencl context and queue
context = cl.create_some_context()
queue = cl.CommandQueue(context)


n_trials = 3
show = True
show_all = True
#plt.ion()

# This method computes the q vectors on the fly.  Slight speed increase.
if 1:
    p = pl[0]  # p is the first panel in the PanelList (there is only one)
    for i in range(0, n_trials):
        t = time.time()
        A = clcore.phase_factor_pad(
            r, f, p.T, p.F, p.S, p.B, p.nF, p.nS, p.beam.wavelength, context=context, queue=queue)
        tf = time.time() - t
        print('phase_factor_pad: %0.3g seconds/atom/pixel' %
              (tf / p.nF / p.nS / r.shape[0]))
    imdisp = np.abs(A)**2
    imdisp = imdisp.reshape((pl[0].nS, pl[0].nF))
    imdisp = np.log(imdisp + 0.1)
    # Display pattern
    if show_all and show:
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")


# This method uses any q vectors that you supply.  Here we grab the q vectors from the
# detector.PanelList class.
if 1:
    q = pl.Q  # These are the scattering vectors, Nx3 array.
    for i in range(0, n_trials):
        t = time.time()
        A = clcore.phase_factor_qrf(q, r, f, context=context, queue=queue)
        tf = time.time() - t
        print('phase_factor_qrf: %0.3g seconds/atom/pixel' %
              (tf / q.shape[0] / r.shape[0]))
    imdisp = np.abs(A)**2
    imdisp = imdisp.reshape((pl[0].nS, pl[0].nF))
    imdisp = np.log(imdisp + 0.1)
    if show_all and show:
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")

# This method uses any q vectors that you supply.  Here we grab the q vectors from the
# detector.PanelList class. This time we make device memory explicitly.
if 1:
    t = time.time()
    q_dev = clcore.to_device(queue, q)
    r_dev = clcore.to_device(queue, r)
    f_dev = clcore.to_device(queue, f)
    a_dev = clcore.to_device(queue, np.zeros([q_dev.shape[0]],dtype=np.complex64))
    tf = time.time() - t
    print('move to device memory: %0.3g seconds' % (tf))
    for i in range(0, n_trials):
        t = time.time()
        a = clcore.phase_factor_qrf(q_dev, r_dev, f_dev, a_dev)
        tf = time.time() - t
        print('phase_factor_qrf: %0.3g seconds (%0.3g/atom/pixel)' %
              (tf, tf / q.shape[0] / r.shape[0]))
    imdisp = np.abs(a.get())**2
    imdisp = imdisp.reshape((pl[0].nS, pl[0].nF))
    imdisp = np.log(imdisp + 0.1)
    if show_all and show:
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")
    del q_dev
    del r_dev
    del f_dev
    del a_dev


# This method involves first making a 3D map of reciprocal-space amplitudes.  We will
# interpolate individual patterns from this map.
if 1:
    res = 1e-10  # Resolution
    qmax = 2 * np.pi / (res)
    qmin = -qmax
    N = 200  # Number of samples
    for i in range(0, n_trials):
        t = time.time()
        A = clcore.phase_factor_mesh(r, f, N, qmin, qmax, context=context, queue=queue)
        tf = time.time() - t
        print('phase_factor_mesh: %0.3g seconds/atom/pixel' %
              (tf / N**3 / r.shape[0]))
    imdisp = A.reshape([N, N, N])
    imdisp = imdisp[(N - 1) / 2, :, :].reshape([N, N])
    imdisp = np.abs(imdisp)**2
    imdisp = np.log(imdisp + 0.1)
    if show_all and show:
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")
    
# This method involves first making a 3D map of reciprocal-space amplitudes.  We will
# interpolate individual patterns from this map.
if 1:
    res = 2e-10  # Resolution
    qmax = 2 * np.pi / (res)
    qmin = -qmax
    N = 200  # Number of samples
    for i in range(0,n_trials):
        t = time.time()
        A = clcore.phase_factor_mesh(r, f, N, qmin, qmax, 
                                        context=context, queue=queue, get=False)
        tf = time.time() - t
        print('phase_factor_mesh: %0.3g seconds/atom/pixel' %
                  (tf / N**3 / r.shape[0]))
    for i in range(0,n_trials):
        t = time.time()
        AA = clcore.buffer_mesh_lookup(A, N, qmin, qmax, pl.Q, 
                                           context=context, queue=queue)
        tf = time.time() - t
        print('buffer_mesh_lookup: %0.3g seconds/atom/pixel' %
                  (tf / pl.Q.shape[0] / r.shape[0]))
    imdisp = AA.reshape(pl[0].nS,pl[0].nF) 
    imdisp = np.abs(imdisp)**2
    imdisp = np.log(imdisp + 0.1)
    if show_all and show:
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")


# Display pattern
if ~show_all and show:
    plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
    plt.title('y: up, x: right, z: beam (towards you)')
    plt.show()
