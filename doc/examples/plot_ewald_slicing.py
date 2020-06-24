r"""
.. _ewald_slice:

Ewald slices through 3D maps
============================

How to make 3D intensity/amplitude map and slice 2D sections on the Ewald sphere.

Contributed by Richard A. Kirian.

Here we build a 3D map of diffraction amplitudes on a GPU.  Then we generate a list of :math:`\vec{q}`
samples corresponding to a pixel-array detector (PAD).  Those vectors naturally lie on the Ewald
sphere.  We "slice" through the 3D map in order to generate simulated diffraction intensities on the
PAD.  By "slice", we mean "trilinear interpolate" in this case.

Imports, constants:
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from reborn.target.density import trilinear_interpolation
import reborn.target.crystal as crystal
import reborn.simulate.clcore
from scipy import constants as const
eV = const.value('electron volt')  # 1.602e-19
# %%
# Detector and x-ray beam
pad = reborn.detector.PADGeometry(shape=(100, 100), pixel_size=1e-3, distance=0.5)
beam = reborn.source.Beam(photon_energy=10000 * eV)
q_vecs = pad.q_vecs(beam=beam)  # q vectors for 2D detector (Nx3 array)
# %%
# Load PDB file, get atomic coordinates and scattering factors
cryst = crystal.CrystalStructure('2LYZ')
r = cryst.molecule.coordinates  # These are atomic coordinates (Nx3 array)
f = cryst.molecule.get_scattering_factors(beam=beam)  # Complex f(0) scattering factors
# %%
# Initialize GPU simulation engine
simcore = reborn.simulate.clcore.ClCore(group_size=32, double_precision=False)
# %%
# Generate 3D mesh of amplitudes on the GPU
q_max = np.max(pad.q_mags(beam=beam))
q_min = -q_max
mesh_shape = (100, 100, 100)
amps_mesh_dev = simcore.to_device(shape=mesh_shape, dtype=simcore.complex_t)
_ = simcore.phase_factor_mesh(r, f, N=mesh_shape[0], q_min=q_min, q_max=q_max, a=amps_mesh_dev)
# %%
# Slice through the 3D GPU amplitude mesh to generate diffraction pattern
t = time.time()
amps_det = simcore.mesh_interpolation(a_map=amps_mesh_dev, q=q_vecs, N=mesh_shape, q_min=q_min, q_max=q_max)
intensity_slice_gpu = pad.reshape(np.abs(amps_det)**2)
print(time.time()-t, 'seconds')
# %%
# Slice through the 3D CPU intensity mesh to generate diffraction pattern
intensity_model = (np.abs(amps_mesh_dev.get())**2).astype(np.float64).reshape(mesh_shape)
t = time.time()
intensity_slice = trilinear_interpolation(intensity_model, q_vecs, x_min=q_min, x_max=q_max)
intensity_slice = pad.reshape(intensity_slice)
print(time.time()-t, 'seconds')
# %%
# Have a look:
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.log10(intensity_slice + 10), interpolation='none', cmap='gnuplot2')
plt.title('Sliced GPU Mesh')
plt.subplot(1, 2, 2)
plt.imshow(np.log10(intensity_slice + 10), interpolation='none', cmap='gnuplot2')
plt.title('Sliced CPU Mesh')
plt.show()
