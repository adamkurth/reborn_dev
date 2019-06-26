import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import bornagain as ba
import bornagain.target.crystal as crystal
import bornagain.simulate.clcore as core
from bornagain.simulate.examples import lysozyme_pdb_file
from bornagain.utils import rotation_about_axis
from scipy import constants as const

hc = const.h*const.c

show = True     # Display the simulated patterns
double = False  # Use double precision if available
dorotate = False  # Check if rotation matrices work
if 'view' in sys.argv:
    show = True
if 'double' in sys.argv:
    double = True
if 'dorotate' in sys.argv:
    dorotate = True
if 'noplots' in sys.argv:
    show = False

clcore = core.ClCore(group_size=32, double_precision=double)

lin = '='*78
print('')
print(lin)
print('Initializations')
print(lin)


# Create a detector
print('Setting up the detector')
tstart = time.time()
nPixels = 1001
pixelSize = 100e-6
detectorDistance = 0.5
wavelength = 1.5e-10
beam_vec = np.array([0, 0, 1])
pad = ba.detector.PADGeometry(n_pixels=nPixels, pixel_size=pixelSize, distance=detectorDistance)
q_vecs = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
n_pixels = q_vecs.shape[0]
tdif = time.time() - tstart
print('CPU q array created in %7.03f ms' % (tdif * 1e3))

# Load a crystal structure from pdb file
pdbFile = lysozyme_pdb_file  # Lysozyme
print('Loading pdb file (%s)' % pdbFile)
cryst = crystal.CrystalStructure(pdbFile)
r = cryst.molecule.coordinates  # These are atomic coordinates (Nx3 array)
n_atoms = r.shape[0]

# Look up atomic scattering factors (they are complex numbers)
print('Getting scattering factors')
f = ba.simulate.atoms.get_scattering_factors(cryst.molecule.atomic_numbers, hc / wavelength)

n_trials = 3

print('Generate rotation matrix')
if dorotate:
    phi = 0.5
    R = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
else:
    R = np.eye(3)


n_trials = 5
print('Will run %d trials' % (n_trials,))
print('Molecule has %d atoms' % (n_atoms,))
print('Detector has %d pixels' % (n_pixels,))
print('')

if 1:

    print(lin)
    print("Simplest on-the-fly amplitude calculation")
    print("Arrays are passed in/out as CPU arrays")
    print(lin)

    q = q_vecs  # These are the scattering vectors, Nx3 array.
    n_pixels = q.shape[0]
    n_atoms = r.shape[0]
    A = 0

    for i in range(0, n_trials):
        t = time.time()
        A = clcore.phase_factor_qrf(q, r, f, R=R)
        tf = time.time() - t
        print('phase_factor_qrf: %7.03f ms' % (tf * 1e3))

    if show:
        imdisp = np.abs(A) ** 2
        imdisp = imdisp.reshape((pad.n_ss, pad.n_fs))
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
        clcore.phase_factor_qrf(q_dev, r_dev, f_dev, R=None, a=a_dev, add=False)
        tf = time.time() - t
        print('phase_factor_qrf: %7.03f ms' % (tf * 1e3))

    t = time.time()
    a = a_dev.get()
    tt = time.time() - t
    print("Moving amplitudes back to CPU memory in %7.03f ms" % (tt*1e3,))

    if show:
        imdisp = np.abs(a) ** 2
        imdisp = imdisp.reshape((pad.n_ss, pad.n_fs))
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

    n_pixels = pad.n_fs * pad.n_ss
    n_atoms = r.shape[0]
    a_dev = clcore.to_device(shape=(n_pixels,), dtype=clcore.complex_t)

    for i in range(0, n_trials):
        t = time.time()
        clcore.phase_factor_pad(r, f, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R=R, a=a_dev)
        tf = time.time() - t
        print('phase_factor_pad: %7.03f ms' % (tf * 1e3))

    t = time.time()
    a = a_dev.get()
    tt = time.time() - t
    print("Moving amplitudes back to CPU memory in %7.03f ms" % (tt*1e3,))

    # Display pattern
    if show:
        imdisp = np.abs(a) ** 2
        imdisp = imdisp.reshape((pad.n_ss, pad.n_fs))
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")

if 1:

    print(lin)
    print("Compute q vectors directly on GPU instead of CPU")
    print("Amplitudes passed in/out as GPU array")
    print("Include two interfering pairs of molecules, shifted in space.")
    print(lin)

    n_pixels = pad.n_fs * pad.n_ss
    n_atoms = r.shape[0]
    a_dev = clcore.to_device(shape=(n_pixels,), dtype=clcore.complex_t)
    U = np.array([1, 0, 0])*5e-9

    for i in range(0, n_trials):
        t = time.time()
        clcore.phase_factor_pad(r, f, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R=R,
                                a=a_dev, U=None, add=False)
        clcore.phase_factor_pad(r, f, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R=R,
                                a=a_dev, U=U, add=True)
        tf = time.time() - t
        print('phase_factor_pad: %7.03f ms' % (tf * 1e3,))

    t = time.time()
    a = a_dev.get()
    tt = time.time() - t
    print("Moving amplitudes back to CPU memory in %7.03f ms" % (tt*1e3,))

    # Display pattern
    if show:
        imdisp = np.abs(a) ** 2
        imdisp = imdisp.reshape((pad.n_ss, pad.n_fs))
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")

if 1:

    print(lin)
    print("Compute q vectors directly on GPU instead of CPU")
    print("Amplitudes passed in/out as GPU array")
    print("Include two interfering pairs of molecules, shifted and rotated.")
    print(lin)

    n_pixels = pad.n_fs * pad.n_ss
    n_atoms = r.shape[0]
    a_dev = clcore.to_device(shape=(n_pixels,), dtype=clcore.complex_t)
    U = np.array([1, 0, 0]) * 5e-9
    RR = rotation_about_axis(20*np.pi/180.0, [1, 1, 0])

    print("Rotation and translation on CPU.")
    r0 = r
    clcore.phase_factor_pad(r0, f, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R=None,
                            a=a_dev, U=None, add=False)
    r0 = np.dot(r, RR.T) + U
    clcore.phase_factor_pad(r0, f, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R=None,
                            a=a_dev, U=None, add=True)
    if show:
        imdisp = np.abs(a_dev.get()) ** 2
        imdisp = imdisp.reshape((pad.n_ss, pad.n_fs))
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()

    print("Rotation and translation on GPU.")
    r0 = r
    clcore.phase_factor_pad(r0, f, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R=None,
                            a=a_dev, U=None, add=False)
    r0 = r
    clcore.phase_factor_pad(r0, f, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R=RR,
                            a=a_dev, U=U, add=True)
    if show:
        imdisp = np.abs(a_dev.get()) ** 2
        imdisp = imdisp.reshape((pad.n_ss, pad.n_fs))
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()

    print("Rotation on CPU, translation on GPU.")
    r0 = r.copy()
    clcore.phase_factor_pad(r0, f, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R=None,
                            a=a_dev, U=None, add=False)
    r0 = np.dot(r.copy(), RR.T)
    clcore.phase_factor_pad(r0, f, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R=None,
                            a=a_dev, U=U, add=True)
    if show:
        imdisp = np.abs(a_dev.get()) ** 2
        imdisp = imdisp.reshape((pad.n_ss, pad.n_fs))
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()

    print("")

if 1:

    print(lin)
    print("Use a 3D lookup table on the GPU")
    print(lin)

    res = 10e-10  # Resolution
    qmax = 2 * np.pi / res
    qmin = -qmax
    N = 200  # Number of samples
    n_atoms = r.shape[0]
    n_pixels = N ** 3

    a_map_dev = clcore.to_device(shape=(n_pixels,), dtype=clcore.complex_t)

    for i in range(0, 1):
        t = time.time()
        clcore.phase_factor_mesh(r, f, N=N, q_min=qmin, q_max=qmax, a=a_map_dev)
        tf = time.time() - t
        print('phase_factor_mesh: %7.03f ms (%d atoms; %d pixels)' % (tf * 1e3, n_atoms, n_pixels))

    print("Interpolate patterns from GPU lookup table")
    print("Amplitudes passed as GPU array")

    q_dev = clcore.to_device(q, dtype=clcore.real_t)
    a_out_dev = clcore.to_device(dtype=clcore.complex_t, shape=(pad.n_fs * pad.n_ss))
    n_atoms = 0
    n_pixels = q.shape[0]
    for i in range(0, n_trials):
        t = time.time()
        clcore.mesh_interpolation(a_map_dev, q_dev, N=N, q_min=qmin, q_max=qmax, R=R, a=a_out_dev)
        tf = time.time() - t
        print('mesh_interpolation: %7.03f ms' % (tf * 1e3))

    t = time.time()
    a = a_out_dev.get()
    tt = time.time() - t
    print("Moving amplitudes back to CPU memory in %7.03f ms" % (tt*1e3))

    if show:
        imdisp = a.reshape(pad.n_ss, pad.n_fs)
        imdisp = np.abs(imdisp) ** 2
        imdisp = np.log(imdisp + 0.1)
        plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
        plt.title('y: up, x: right, z: beam (towards you)')
        plt.show()
    print("")
