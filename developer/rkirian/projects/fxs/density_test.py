r"""
Density maps (incomplete)
=========================

Contributed by Richard A. Kirian

"""

import numpy as np
from numpy.fft import fftn, ifftn, fftshift
from scipy import constants as const
import matplotlib.pyplot as plt
from reborn.target import crystal, atoms
from reborn.data import lysozyme_pdb_file
import reborn.simulate.clcore as core

hc = const.h*const.c

fftmethod = True

pdb_file = lysozyme_pdb_file
pdb_file = 'BDNA25_sp.pdb'
pdb_file = '3IYF'
print('Loading pdb file (%s)' % pdb_file)
cryst = crystal.CrystalStructure(pdb_file, tight_packing=True)


# Look up atomic scattering factors (they are complex numbers)
print('Getting scattering factors')
wavelength = 1.5e-10
f = cryst.molecule.get_scattering_factors(hc / wavelength)

print('Setting up 3D mesh')
d = 0.5e-9  # Minimum resolution
s = 2       # Oversampling factor
dmap = crystal.CrystalDensityMap(cryst, d, s)
print('Grid size: (%d, %d, %d)' % tuple(dmap.shape))
h = dmap.h_vecs  # Miller indices (fractional)

# if 0:  # Display the mesh as a scatterplot
#
#     print('Showing the mesh of h vectors')
#     v = h
#     shape = list(dmap.shape)
#     shape.append(3)
#     v3 = v.reshape(shape)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(v[:, 0], v[:, 1], v[:, 2], c='k', marker='.', s=1)
#     ax.scatter(v3[0, 0, :, 0], v3[0, 0, :, 1], v3[0, 0, :, 2], c='r', marker='.', s=200) #, edgecolor='')
#     ax.scatter(v3[0, :, 0, 0], v3[0, :, 0, 1], v3[0, :, 0, 2], c='g', marker='.', s=200) #, edgecolor='')
#     ax.scatter(v3[:, 0, 0, 0], v3[:, 0, 0, 1], v3[:, 0, 0, 2], c='b', marker='.', s=200) #, edgecolor='')
#     ax.set_aspect('auto')
#     plt.show()


# Function to show orthogonal slices of diffraction intensities and density projections
def show(I, rho_cell, dmap):
    fig = plt.figure()
    fig.add_subplot(2, 3, 1)
    print(I.shape)
    dispim = np.log10(I[int(I.shape[0] / 2), :, :] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
    fig.add_subplot(2, 3, 4)
    dispim = np.sum(rho_cell, axis=0)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
    fig.add_subplot(2, 3, 2)
    dispim = np.log10(I[:, int(I.shape[1] / 2), :] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
    fig.add_subplot(2, 3, 5)
    dispim = np.sum(rho_cell, axis=1)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
    fig.add_subplot(2, 3, 3)
    dispim = np.log10(I[:, :, int(I.shape[2] / 2)] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
    fig.add_subplot(2, 3, 6)
    dispim = np.sum(rho_cell, axis=2)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

# Simulate amplitudes via summation over atoms, then make 3D density map via FFT
print('Simulating diffraction amplitudes')
clcore = core.ClCore(group_size=32, double_precision=False)
F = clcore.phase_factor_qrf(2 * np.pi * h, cryst.x, f)
F = np.reshape(F, dmap.shape)
I = np.abs(F) ** 2
rho = ifftn(F)
rho = np.abs(rho)
rho_cell = np.zeros(dmap.shape)
for i in range(1): #range(0, len(dmap.get_sym_luts())):
    rho_cell += dmap.symmetry_transform(0, i, rho)
rho_cell = fftshift(rho_cell)
show(I, np.real(rho_cell), np.real(dmap))
plt.suptitle('Atomistic direct amplitude calculation')

# Make a density map from atoms, then FFT to get amplities
print('Creating density map directly from atoms')
x = cryst.x
rho = dmap.place_atoms_in_map(cryst.x % dmap.oversampling, f, mode='nearest')
voxel_vol = dmap.voxel_volume
f_dens = atoms.xraylib_scattering_density(compound='H20', density=1000, photon_energy=7000*1.6022e-19)

rho[rho == 0] = f_dens*voxel_vol
F = fftn(rho)
I = np.abs(F)**2
rho_cell = np.zeros_like(rho)
for i in range(1): # range(0, len(dmap.get_sym_luts())):
    rho_cell += dmap.symmetry_transform(0, i, rho)
I = fftshift(I)
rho_cell = fftshift(rho_cell)
show(I, np.real(rho_cell), np.real(dmap))
plt.suptitle('Fourier transform of density map')



plt.show()
