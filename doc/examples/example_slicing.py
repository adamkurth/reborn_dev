# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

r"""
Density maps (incomplete)
=========================

Contributed by Richard A. Kirian

This is similar to the example :ref:`ewald_slice`.  However, here we generate a 3D map corresponding
to a crystal, and we work in the basis of the crystal lattice.

Imports, etc.
"""

import numpy as np
from numpy.fft import fftn, fftshift
from scipy import constants as const
import matplotlib.pyplot as plt
import reborn
from reborn.simulate import clcore
if plt.rcParams['backend'] == "agg":
    plt.show = lambda: None
eV = const.value('electron volt')
# %%
# Load PDB file with atomic coordinates:
cryst = reborn.target.crystal.CrystalStructure('2LYZ')
# %%
# Look up atomic scattering factors :math:`f(0)` (they are complex numbers):
f = cryst.molecule.get_scattering_factors(photon_energy=10000*eV)
# %%
# Set up a 3D mesh corresponding to a 2x2 block of crystal unit cells:
d = 0.5e-9  # Minimum resolution
s = 4       # Oversampling factor
dmap = reborn.target.crystal.CrystalDensityMap(cryst, d, s)
print('Grid size: (%d, %d, %d)' % tuple(dmap.shape))
h = dmap.h_vecs  # Miller indices (fractional)
# %%
# Create a crude density map from the atoms:
x = cryst.x
rho = dmap.place_atoms_in_map(cryst.x, np.real(np.abs(f)), mode='trilinear')
# %%
# Now that we have the density map for the molecule ("asymmetric unit"), we could replicate that density according to
# the crystallographic space group symmetry.  We won't do that here, but here is the needed operation:

# for i in range(1, len(dmap.get_sym_luts())):
#    rho += dmap.symmetry_transform(0, i, rho)
# %%
# From the density map, we create the intensity map via Fourier transform (modulus squared).
F = fftn(rho)
I = np.abs(F)**2
I = fftshift(I)
# %%
# Show orthogonal slices of diffraction intensities and density projections
rho = np.abs(rho)
fig = plt.figure()
dispargs = {'interpolation': 'nearest', 'cmap': 'gray'}
fig.add_subplot(2, 3, 1)
plt.imshow(np.log10(I[np.floor(dmap.shape[0] / 2).astype(int), :, :] + 10), **dispargs)
fig.add_subplot(2, 3, 4)
plt.imshow(np.sum(rho, axis=0), **dispargs)
fig.add_subplot(2, 3, 2)
plt.imshow(np.log10(I[:, np.floor(dmap.shape[1] / 2).astype(int), :] + 10),  **dispargs)
fig.add_subplot(2, 3, 5)
plt.imshow(np.sum(rho, axis=1), **dispargs)
fig.add_subplot(2, 3, 3)
plt.imshow(np.log10(I[:, :, np.floor(dmap.shape[2] / 2).astype(int)] + 10),  **dispargs)
fig.add_subplot(2, 3, 6)
plt.imshow(np.sum(rho, axis=2), **dispargs)
plt.show()
# %%
# Create a beam and detector:
pad = reborn.detector.PADGeometry(shape=(100, 100), pixel_size=100e-6, distance=0.05)
beam = reborn.source.Beam(wavelength=1e-10)
# %%
# Here are the :math:`\vec{q}` vectors corresponding to detector pixels.  They sample a portion of the Ewald sphere that
# slices through the origin of reciprocal space.
q = pad.q_vecs(beam=beam)
# %%
# We want to sample the 3D intensity map at the points corresponding to the detector :math:`\vec{q}` vectors.  We can
# use a trilinear interpolation to do this, but there is one complication: the 3D intensity map is sampled according
# to crystallographic Bragg sampling (actually, it is "oversampled", but still the samples are determined according
# to the crystal lattice).  We need to convert the :math:`\vec{q}` vectors into :math:`\vec{h}` vectors.  You can find
# more information about these transformations in the :ref:`working_with_crystals` page.
h = cryst.unitcell.q2h(q)
# %%
# Now we do the trilinear interpolation:
I_slice = reborn.misc.interpolate.trilinear_interpolation(densities=I.copy(), vectors=h, corners=dmap.h_limits[:, 0],
                                                        deltas=(dmap.h_max-dmap.h_min)/(dmap.shape-1))
I_slice = pad.reshape(I_slice)
# %%
# Now let's simulate the slice directly using atomic coordinates.  There are other :ref:`examples` that show what is
# going on in the lines below:
simcore = clcore.ClCore()
amps = simcore.phase_factor_pad(cryst.molecule.coordinates, np.real(np.abs(f)), pad=pad, beam=beam)
I_sim = pad.reshape(np.abs(amps)**2)
# %%
# Now compare the results for the two different methods:
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.log10(I_slice), **dispargs)
plt.title('Sliced intensity')
plt.subplot(1, 2, 2)
plt.imshow(np.log10(I_sim), **dispargs)
plt.title('Simulated intensity')
plt.show()
