import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")
import bornagain as ba
import bornagain.target.crystal as crystal

try:
    import bornagain.simulate.clcore as core
except:
    print('Cannot import clcore; check that pyopencl is installed')

show = False  # Display the simulated patterns
double = False  # Use double precision if available
rotate = False  # Check if rotation matrices work
if 'view' in sys.argv: show = True
if 'double' in sys.argv: double = True
if 'rotate' in sys.argv: rotate = True

clcore = core.ClCore(group_size=32, double_precision=double)

# Create a detector
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

# Load a crystal structure from pdb file
pdbFile = '../data/pdb/2LYZ.pdb'  # Lysozyme
cryst = crystal.structure(pdbFile)
print('')
print('Loading pdb file: %s' % pdbFile)
print('')

# These are atomic coordinates (Nx3 array)
r = cryst.r

# Look up atomic scattering factors (they are complex numbers)
f = ba.simulate.atoms.get_scattering_factors(cryst.Z, ba.units.hc / wavelength)

n_trials = 3
show_all = show
# plt.ion()

if rotate:
    phi = 0.5
    R = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
else:
    R = np.eye(3)

if 1:
    print("[clcore] Access q vectors from memory (i.e. compute them on cpu first)")
    tstart = time.time()
    q = q_vecs  # These are the scattering vectors, Nx3 array.

    n_pixels = q.shape[0]
    n_atoms = r.shape[0]
    for i in range(0, n_trials):
        t = time.time()
        A = clcore.phase_factor_qrf(q, r, f, R)
        tf = time.time() - t
        print('phase_factor_qrf: %7.03f ms (%d atoms; %d pixels)' %
              (tf * 1e3, n_atoms, n_pixels))
    imdisp = np.abs(A) ** 2
    
   
    tstop = (time.time() - tstart) / n_trials
    print("\n========= TOTAL TIME =======")
    print('phase_factor_qrf: %7.03f ms per trial (%d atoms; %d pixels)' %
        (tstop * 1e3, n_atoms, n_pixels))
    print("============================\n")
    imdisp = imdisp.reshape((pl.n_ss, pl.n_fs))
    imdisp = np.log(imdisp + 0.1)
    if show_all and show:
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")

if 1:

    print("[clcore] Compute q vectors on cpu, but load into GPU memory only once (at the beginning)")
    tstart = time.time()
    t = time.time()
    n_pixels = q.shape[0]
    n_atoms = r.shape[0]
    q_dev = clcore.to_device(q, dtype=clcore.real_t)
    r_dev = clcore.to_device(r, dtype=clcore.real_t)
    f_dev = clcore.to_device(f, dtype=clcore.complex_t)
    a_dev = clcore.to_device(shape=(q_dev.shape[0]), dtype=clcore.complex_t)
    tf = time.time() - t
    print('Move to device memory: %7.03f ms' % (tf * 1e3))
    for i in range(0, n_trials):
        t = time.time()
        clcore.phase_factor_qrf(q_dev, r_dev, f_dev, None, a_dev, False)
        tf = time.time() - t
        print('phase_factor_qrf: %7.03f ms (%d atoms; %d pixels)' %
              (tf * 1e3, n_atoms, n_pixels))
   
    print("\n  ==================================")
    print("  Getting amplitude back from device")
    tt = time.time()
    imdisp = np.abs(a_dev.get()) ** 2
    print("  --> took %7.03f ms"%( (time.time()-tt)*1e3))
#   this above bit is what slows things down, seems non-intuitive, but this time required
#   to copy back from device scales with n_trials
    
    tstop = (time.time() - tstart) / n_trials
    print("\n========= TOTAL TIME =======")
    print('phase_factor_qrf: %7.03f ms per trial (%d atoms; %d pixels)' %
        (tstop * 1e3, n_atoms, n_pixels))
    print("============================\n")
    imdisp = imdisp.reshape((pl.n_ss, pl.n_fs))
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

if 1:

    print("[clcore] Compute q vectors on the fly.  Faster than accessing q's from memory?")
    n_pixels = pl.n_fs * pl.n_ss
    n_atoms = r.shape[0]
    for i in range(0, n_trials):
        t = time.time()
        A = clcore.phase_factor_pad(
            r, f, pl.t_vec, pl.fs_vec, pl.ss_vec, beam_vec, pl.n_fs, pl.n_ss, wavelength, R)
        tf = time.time() - t
        print('phase_factor_pad: %7.03f ms (%d atoms; %d pixels)' %
              (tf * 1e3, n_atoms, n_pixels))
    imdisp = np.abs(A) ** 2
    imdisp = imdisp.reshape((pl.n_ss, pl.n_fs))
    imdisp = np.log(imdisp + 0.1)
    # Display pattern
    if show_all and show:
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")

if 0:

    print("[clcore] Make a full 3D diffraction amplitude map...")
    res = 1e-10  # Resolution
    qmax = 2 * np.pi / (res)
    qmin = -qmax
    N = 128  # Number of samples
    n_atoms = r.shape[0]
    n_pixels = N ** 3
    for i in range(0, 1):
        t = time.time()
        A = clcore.phase_factor_mesh(r, f, N, qmin, qmax)
        tf = time.time() - t
        print('phase_factor_mesh: %7.03f ms (%d atoms; %d pixels)' %
              (tf * 1e3, n_atoms, n_pixels))
    imdisp = A.reshape([N, N, N])
    imdisp = imdisp[(N - 1) / 2, :, :].reshape([N, N])
    imdisp = np.abs(imdisp) ** 2
    imdisp = np.log(imdisp + 0.1)
    if show_all and show:
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")

if 1:

    res = 1e-10  # Resolution
    qmax = 2 * np.pi / (res)
    qmin = -qmax
    N = 200  # Number of samples
    n_atoms = r.shape[0]
    n_pixels = N ** 3
    print("[clcore] First compute a 3D diffraction amplitude map...")
    for i in range(0, 1):
        t = time.time()
        A = clcore.phase_factor_mesh(r, f, N, qmin, qmax)
        tf = time.time() - t
        print('phase_factor_mesh: %7.03f ms (%d atoms; %d pixels)' %
              (tf * 1e3, n_atoms, n_pixels))
    print('')

    # pl = ba.detector.PanelList()
    # nPixels = 1000
    # pixelSize = 100e-6
    # detectorDistance = 0.05
    # wavelength = 1.5e-10
    # pl.simple_setup(nPixels, nPixels + 1, pixelSize, detectorDistance, wavelength)
    # q = pl[0].Q
    # n_atoms = 0
    # n_pixels = q.shape[0]
    print("[clcore] Now look up amplitudes for a set of q vectors, using the 3D map")
    for i in range(0, n_trials):
        t = time.time()
        AA = clcore.buffer_mesh_lookup(A, N, qmin, qmax, q, R)
        tf = time.time() - t
        print('buffer_mesh_lookup: %7.03f ms (%d atoms; %d pixels)' %
              (tf * 1e3, n_atoms, n_pixels))
    if show_all and show:
        imdisp = AA.reshape(pl.n_ss, pl.n_fs)
        imdisp = np.abs(imdisp) ** 2
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")

    t = time.time()
    q_dev = clcore.to_device(q, dtype=clcore.real_t)
    a_map_dev = clcore.to_device(A, dtype=clcore.complex_t)
    a_out_dev = clcore.to_device(dtype=clcore.complex_t, shape=(pl.n_fs*pl.n_ss))
    tf = time.time() - t
    print('[clcore] As above, but first move arrays to device memory (%7.03f ms)' % (tf * 1e3))
    n_atoms = 0
    n_pixels = q.shape[0]
    for i in range(0, n_trials):
        t = time.time()
        clcore.buffer_mesh_lookup(a_map_dev, N, qmin, qmax, q_dev, R, a_out_dev)
        tf = time.time() - t
        print('buffer_mesh_lookup: %7.03f ms (%d atoms; %d pixels)' %
              (tf * 1e3, n_atoms, n_pixels))
    if show_all and show:
        imdisp = a_out_dev.get().reshape(pl.n_ss, pl.n_fs)
        imdisp = np.abs(imdisp) ** 2
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")



# if 1:
#     print('[cycore] First test')
#     # Create a detector
#     pl = ba.detector.PanelList()
#     nPixels = 100
#     pixelSize = 100e-6
#     detectorDistance = 0.01
#     wavelength = 1.5e-10
#     pl.simple_setup(nPixels, nPixels+1, pixelSize, detectorDistance, wavelength)
#     q = pl[0].Q
#     #r = r[0:5,:]
#     n_atoms = r.shape[0]
#     n_pixels = q.shape[0]
#     for i in range(0,n_trials):
#         t = time.time()
#         A = cycore.molecularFormFactor(q.astype(clcore.real_t),r.astype(clcore.real_t),f.astype(clcore.complex_t))
#         tf = time.time() - t
#         print('phase_factor_qrf: %7.03f ms (%d atoms; %d pixels)' %
#               (tf*1e3,n_atoms,n_pixels))
#     if show_all and show:
#         imdisp = A.reshape(pl.n_ss,pl.n_fs)
#         imdisp = np.abs(imdisp)**2
#         imdisp = np.log(imdisp + 0.1)
#         plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
#         plt.title('y: up, x: right, z: beam (towards you)')
#         plt.show()
