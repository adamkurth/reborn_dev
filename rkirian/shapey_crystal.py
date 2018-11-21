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
show_all = True

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
detectorDistance = 0.2
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
pdbFile = '../examples/data/pdb/2LYZ.pdb'  # Lysozyme
print('Loading pdb file (%s)' % pdbFile)
cryst = crystal.Structure(pdbFile)
r = cryst.r # These are atomic coordinates (Nx3 array)
n_atoms = r.shape[0]

# Look up atomic scattering factors (they are complex numbers)
print('Getting scattering factors')
f = ba.simulate.atoms.get_scattering_factors(cryst.Z, ba.units.hc / wavelength)



# Atomic coordinates
N_latt = 20
tmp = np.arange(0, N) * 10e-10
[xx, yy, zz] = np.meshgrid(tmp, tmp, tmp, indexing='ij')
r_latt = np.zeros([N ** 3, 3])
r_latt[:, 0] = zz.flatten()
r_latt[:, 1] = yy.flatten()
r_latt[:, 2] = xx.flatten()

# Scattering factors
f_latt = np.ones([N ** 3])





print('Generate rotation matrix')
if rotate:
    phi = 0.5
    R = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
else:
    R = np.eye(3)

# print(lin)
# print("Compute q vectors directly on GPU instead of CPU")
# print("Amplitudes passed in/out as GPU array")
# print(lin)
#
# n_pixels = pl.n_fs * pl.n_ss
# n_atoms = r.shape[0]
# a_dev = clcore.to_device(shape=(n_pixels), dtype=clcore.complex_t)
#
# t = time.time()
# clcore.phase_factor_pad(r, f, pl.t_vec, pl.fs_vec, pl.ss_vec, beam_vec, pl.n_fs, pl.n_ss, wavelength, R=R, a=a_dev)
#
# tf = time.time() - t
# print('phase_factor_pad: %7.03f ms' % (tf * 1e3))
#
# t = time.time()
# a = a_dev.get()
# tt = time.time() - t
# print("Moving amplitudes back to CPU memory in %7.03f ms"%(tt*1e3))





# # Compute diffraction amplitudes
# t = time.time()
# a_dev_latt = clcore.to_device(shape=(n_pixels), dtype=clcore.complex_t)
# clcore.phase_factor_pad(r_latt, f_latt, pl.t_vec, pl.fs_vec, pl.ss_vec, beam_vec, pl.n_fs, pl.n_ss, wavelength, R=R, a=a_dev_latt)
# print(time.time() - t)
#
#
# I = np.abs( a_dev.get() * a_dev_latt.get() )**2
# I = pl.reshape(I)
#
#
#
# # Display pattern
# if False:
#     # imdisp = np.abs(a) ** 2
#     # imdisp = imdisp.reshape((pl.n_ss, pl.n_fs))
#     # imdisp = np.log(imdisp + 0.1)
#     plt.imshow(np.log(I + 0.1), interpolation='nearest', cmap='gray', origin='lower')
#     plt.title('y: up, x: right, z: beam (towards you)')
#     plt.show()
# print("")










# ==============================================================
# 3d mesh




res = 5e-10  # Resolution
qmax = 2 * np.pi / (res)
qmin = -qmax
N = 128  # Number of samples
n_atoms = r.shape[0]
n_pixels = N ** 3

a_map_dev = clcore.to_device(shape=(n_pixels), dtype=clcore.complex_t)

t = time.time()
clcore.phase_factor_mesh(r, f, N, qmin, qmax, a_map_dev)
tf = time.time() - t
#print('phase_factor_mesh: %7.03f ms (%d atoms; %d pixels)' % (tf * 1e3, n_atoms, n_pixels))


a_map_latt = clcore.to_device(shape=(n_pixels), dtype=clcore.complex_t)

t = time.time()
clcore.phase_factor_mesh(r_latt, f_latt, N, qmin, qmax, a_map_latt)
tf = time.time() - t
#print('phase_factor_mesh: %7.03f ms (%d atoms; %d pixels)' % (tf * 1e3, n_atoms, n_pixels))

I = np.abs(a_map_dev.get() * a_map_latt.get() )**2
I = I.reshape((N_latt)*3)

if show_all and show:
    # imdisp = np.abs(a) ** 2
    # imdisp = imdisp.reshape((pl.n_ss, pl.n_fs))
    # imdisp = np.log(imdisp + 0.1)
    plt.imshow(np.log(I[int(np.floor(N_latt/2)), :, :] + 0.1), interpolation='nearest', cmap='gray', origin='lower')
    plt.title('y: up, x: right, z: beam (towards you)')
    plt.show()
print("")












