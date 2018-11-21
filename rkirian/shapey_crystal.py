import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")
import bornagain as ba
import bornagain.target.crystal as crystal
import bornagain.simulate.clcore as core
import bornagain.viewers.qtviews.qtviews as qtviews

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
N_latt = 5
latt_len = 20e-10
tmp = np.arange(0, N_latt) * latt_len
[xx, yy, zz] = np.meshgrid(tmp, tmp, tmp, indexing='ij')
r_latt = np.zeros([N_latt ** 3, 3])
r_latt[:, 0] = zz.flatten()
r_latt[:, 1] = yy.flatten()
r_latt[:, 2] = xx.flatten() + np.random.rand(len(xx.flatten()))*latt_len*0.2





# Scattering factors
f_latt = np.ones([N_latt ** 3])





print('Generate rotation matrix')
if rotate:
    phi = 0.5
    R = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
else:
    R = np.eye(3)



res = 5e-10  # Resolution
qmax = 2 * np.pi / (res)
qmin = -qmax
N_mesh = 128  # Number of samples
# n_atoms = r.shape[0]
n_pixels = N_mesh ** 3

a_map_dev = clcore.to_device(shape=(n_pixels), dtype=clcore.complex_t)

t = time.time()
clcore.phase_factor_mesh(r, f, N_mesh, qmin, qmax, a_map_dev)
tf = time.time() - t
print('phase_factor_mesh (molecule): %7.03f ms (%d atoms; %d pixels)' % (tf * 1e3, n_atoms, n_pixels))


a_map_latt = clcore.to_device(shape=(n_pixels), dtype=clcore.complex_t)

t = time.time()
clcore.phase_factor_mesh(r_latt, f_latt, N_mesh, qmin, qmax, a_map_latt)
tf = time.time() - t
print('phase_factor_mesh (lattice; %d points): %7.03f ms (%d atoms; %d pixels)' % (n_pixels, tf * 1e3, n_atoms, n_pixels))

I = np.abs(a_map_dev.get() * a_map_latt.get() )**2
I = I.reshape((N_mesh, N_mesh, N_mesh))

if show_all and show:
    # imdisp = np.abs(a) ** 2
    # imdisp = imdisp.reshape((pl.n_ss, pl.n_fs))
    # imdisp = np.log(imdisp + 0.1)
    qtviews.MapSlices(np.log(I + 0.01))
    scat = qtviews.Scatter3D()
    scat.add_points(r_latt, size=5)
    scat.show()
    # plt.imshow(np.log(I[int(np.floor(N_latt/2)), :, :] + 0.1), interpolation='nearest', cmap='gray', origin='lower')
    # plt.title('y: up, x: right, z: beam (towards you)')
    plt.show()
print("")












