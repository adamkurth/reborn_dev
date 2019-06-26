from __future__ import division

import sys

import numpy as np
from numpy.fft import fftn, ifftn, fftshift
import matplotlib.pyplot as plt

from bornagain.viewers.qtviews import qtviews
from bornagain.target import crystal, density
from bornagain.data import lysozyme_pdb_file
import bornagain.simulate.clcore as core
from scipy import constants as const

hc = const.h*const.c

plot = True
if 'noplots' in sys.argv:
    plot = False

fftmethod = True

pdb_file = lysozyme_pdb_file
print('Loading pdb file (%s)' % pdb_file)
cryst = crystal.CrystalStructure(pdb_file)

if plot and 0:
    # Show atoms in scatter plot
    r = cryst.molecule.coordinates
    sv = qtviews.Scatter3D()
    sv.add_points(r)
    sv.show()


# Look up atomic scattering factors (they are complex numbers)
print('Getting scattering factors')
wavelength = 1.5e-10
f = cryst.molecule.get_scattering_factors(hc / wavelength)

print('Setting up 3D mesh')
d = 0.5e-9  # Minimum resolution
s = 2       # Oversampling factor
mt = density.CrystalDensityMap(cryst, d, s)
print('Grid size: (%d, %d, %d)' % tuple(mt.shape))
h = mt.h_vecs  # Miller indices (fractional)

if plot and 0:  # Display the mesh as a scatterplot

    print('Showing the mesh of h vectors')
    v = h
    v3 = mt.reshape3(v)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], c='k', marker='.', s=1)
    ax.scatter(v3[0, 0, :, 0], v3[0, 0, :, 1], v3[0, 0, :, 2], c='r', marker='.', s=200, edgecolor='')
    ax.scatter(v3[0, :, 0, 0], v3[0, :, 0, 1], v3[0, :, 0, 2], c='g', marker='.', s=200, edgecolor='')
    ax.scatter(v3[:, 0, 0, 0], v3[:, 0, 0, 1], v3[:, 0, 0, 2], c='b', marker='.', s=200, edgecolor='')
    ax.set_aspect('equal')
    plt.show()


if 1:  # Simulate amplitudes with clcore, then make 3D density map via FFT

    print('Simulating diffraction amplitudes')
    clcore = core.ClCore(group_size=32, double_precision=False)
    F = clcore.phase_factor_qrf(2 * np.pi * h, cryst.x, f)
    F = np.reshape(F, mt.shape)
    I = np.abs(F) ** 2
    rho = ifftn(F)
    rho = np.abs(rho)

    rho_cell = np.zeros(mt.shape)
    for i in range(0, len(mt.get_sym_luts())):
        rho_cell += mt.symmetry_transform(0, i, rho)


else:  # This is the direct way of making a density map from atoms and their structure factors

    print('Creating density map directly from atoms')
    x = cryst.x
    print(np.real(np.abs(f)))
    rho = mt.place_atoms_in_map(cryst.x % mt.oversampling, np.real(np.abs(f)))
    F = fftn(rho)
    I = np.abs(F)**2
    rho_cell = np.zeros_like(rho)
    for i in range(0, len(mt.get_sym_luts())):
        rho_cell += mt.symmetry_transform(0, i, rho)

rho_cell = fftshift(rho_cell)

if plot and 1:

    print('Showing 3D volumetric rendering of unit cell')
    vol = qtviews.Volumetric3D()
    for i in range(0, len(mt.get_sym_luts())):
        # print(rho)
        vol.add_density(fftshift(mt.symmetry_transform(0, i, rho)), qtviews.bright_colors(i))
    vol.show()

if plot and 1:

    print('Showing orthogonal slices of diffraction intensities and density projections')
    fig = plt.figure()

    fig.add_subplot(2, 3, 1)
    dispim = np.log10(I[np.floor(mt.shape[0]/2).astype(np.int), :, :] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 4)
    dispim = np.sum(rho_cell, axis=0)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 2)
    dispim = np.log10(I[:, np.floor(mt.shape[1]/2).astype(np.int), :] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 5)
    dispim = np.sum(rho_cell, axis=1)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 3)
    dispim = np.log10(I[:, :, np.floor(mt.shape[2]/2).astype(np.int)] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 6)
    dispim = np.sum(rho_cell, axis=2)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    plt.show()
