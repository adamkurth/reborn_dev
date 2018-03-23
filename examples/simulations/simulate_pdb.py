import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")
import bornagain as ba
import bornagain.target.crystal as crystal
import bornagain.simulate.clcore as core


show = False  # Display the simulated patterns
double = False  # Use double precision if available
rotate = False  # Check if rotation matrices work
if 'view' in sys.argv: show = True
if 'double' in sys.argv: double = True
if 'rotate' in sys.argv: rotate = True

show = True

clcore = core.ClCore(group_size=32, double_precision=double)

lin = '='*78
print('')
print(lin)
print('Initializations')
print(lin)


# Create a detector
print('Setting up the detector')
tstart = time.time()
pl = ba.detector.PADGeometry()
nPixels = 1001
pixelSize = 100e-6
detectorDistance = 0.05
wavelength = 1.5e-10
beam_vec = np.array([0, 0, 1])
pl.simple_setup(n_pixels=nPixels, pixel_size=pixelSize, distance=detectorDistance)
# pl.simple_setup(nPixels, nPixels + 1, pixelSize, detectorDistance, wavelength)
# q = pl[0].Q
q_vecs = pl.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
n_pixels = q_vecs.shape[0]
tdif = time.time() - tstart
print('CPU q array created in %7.03f ms' % (tdif * 1e3))

# Load a crystal structure from pdb file
pdbFile = '../data/pdb/2LYZ.pdb'  # Lysozyme
print('Loading pdb file (%s)' % pdbFile)
cryst = crystal.structure(pdbFile)
r = cryst.r # These are atomic coordinates (Nx3 array)
n_atoms = r.shape[0]

# Look up atomic scattering factors (they are complex numbers)
print('Getting scattering factors')
f = ba.simulate.atoms.get_scattering_factors(cryst.Z, ba.units.hc / wavelength)

n_trials = 3
show_all = show
# plt.ion()

print('Generate rotation matrix')
if rotate:
    phi = 0.5
    R = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
else:
    R = np.eye(3)


n_trials = 5
show_all = show
print('Will run %d trials' % (n_trials))
print('Molecule has %d atoms' % (n_atoms))
print('Detector has %d pixels' % (n_pixels))
print('')

if 1:

    print(lin)
    print("Simplest on-the-fly amplitude calculation")
    print("Arrays are passed in/out as CPU arrays")
    print(lin)

    q = q_vecs  # These are the scattering vectors, Nx3 array.
    n_pixels = q.shape[0]
    n_atoms = r.shape[0]

    for i in range(0, n_trials):
        t = time.time()
        A = clcore.phase_factor_qrf(q, r, f, R)
        tf = time.time() - t
        print('phase_factor_qrf: %7.03f ms' % (tf * 1e3))

    if show_all and show:
        imdisp = np.abs(A) ** 2
        imdisp = imdisp.reshape((pl.n_ss, pl.n_fs))
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")

if 1:

    print(lin)
    print("Simplest on-the-fly amplitude calculation")
    print("Arrays manually transferred to GPU device")
    print(lin)

    t = time.time()
    n_pixels = q.shape[0]
    n_atoms = r.shape[0]
    q_dev = clcore.to_device(q, dtype=clcore.real_t)
    r_dev = clcore.to_device(r, dtype=clcore.real_t)
    f_dev = clcore.to_device(f, dtype=clcore.complex_t)
    a_dev = clcore.to_device(shape=(q_dev.shape[0]), dtype=clcore.complex_t)
    tf = time.time() - t
    print('Move to GPU memory: %7.03f ms' % (tf * 1e3))

    for i in range(0, n_trials):
        t = time.time()
        clcore.phase_factor_qrf(q_dev, r_dev, f_dev, None, a_dev, False)
        tf = time.time() - t
        print('phase_factor_qrf: %7.03f ms' % (tf * 1e3))

    t = time.time()
    a = a_dev.get()
    tt = time.time() - t
    print("Moving amplitudes back to CPU memory in %7.03f ms"%(tt*1e3))

    if show_all and show:
        imdisp = np.abs(a) ** 2
        imdisp = imdisp.reshape((pl.n_ss, pl.n_fs))
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")
    del q_dev
    del r_dev
    del f_dev
    del a_dev

if 1:

    print(lin)
    print("Compute q vectors directly on GPU instead of CPU")
    print("Amplitudes passed in/out as GPU array")
    print(lin)

    n_pixels = pl.n_fs * pl.n_ss
    n_atoms = r.shape[0]
    a_dev = clcore.to_device(shape=(n_pixels), dtype=clcore.complex_t)

    for i in range(0, n_trials):
        t = time.time()
        clcore.phase_factor_pad(r, f, pl.t_vec, pl.fs_vec, pl.ss_vec, beam_vec, pl.n_fs, pl.n_ss, wavelength, R=R, a=a_dev)
        tf = time.time() - t
        print('phase_factor_pad: %7.03f ms' % (tf * 1e3))

    t = time.time()
    a = a_dev.get()
    tt = time.time() - t
    print("Moving amplitudes back to CPU memory in %7.03f ms"%(tt*1e3))

    # Display pattern
    if show_all and show:
        imdisp = np.abs(a) ** 2
        imdisp = imdisp.reshape((pl.n_ss, pl.n_fs))
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")

if 1:

    print(lin)
    print("Use a 3D lookup table on the GPU")
    print(lin)

    res = 1e-10  # Resolution
    qmax = 2 * np.pi / (res)
    qmin = -qmax
    N = 200  # Number of samples
    n_atoms = r.shape[0]
    n_pixels = N ** 3

    a_map_dev = clcore.to_device(shape=(n_pixels), dtype=clcore.complex_t)

    for i in range(0, 1):
        t = time.time()
        clcore.phase_factor_mesh(r, f, N, qmin, qmax, a_map_dev)
        tf = time.time() - t
        print('phase_factor_mesh: %7.03f ms (%d atoms; %d pixels)' %
              (tf * 1e3, n_atoms, n_pixels))

    print("Interpolate patterns from GPU lookup table")
    print("Amplitudes passed as GPU array")

    # t = time.time()
    q_dev = clcore.to_device(q, dtype=clcore.real_t)
    # a_map_dev = clcore.to_device(A, dtype=clcore.complex_t)
    a_out_dev = clcore.to_device(dtype=clcore.complex_t, shape=(pl.n_fs*pl.n_ss))
    # tf = time.time() - t
    # print('As above, but first move arrays to device memory (%7.03f ms)' % (tf * 1e3))
    n_atoms = 0
    n_pixels = q.shape[0]
    for i in range(0, n_trials):
        t = time.time()
        clcore.buffer_mesh_lookup(a_map_dev, N, qmin, qmax, q_dev, R, a_out_dev)
        tf = time.time() - t
        print('buffer_mesh_lookup: %7.03f ms' % (tf * 1e3))

    t = time.time()
    a = a_out_dev.get()
    tt = time.time() - t
    print("Moving amplitudes back to CPU memory in %7.03f ms"%(tt*1e3))

    if show_all and show:
        imdisp = a.reshape(pl.n_ss, pl.n_fs)
        imdisp = np.abs(imdisp) ** 2
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")

