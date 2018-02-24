from __future__ import division

import sys

import numpy as np
from numpy.fft import fftn, ifftn, fftshift
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import binned_statistic_dd

sys.path.append("../..")
import bornagain as ba
from bornagain.viewers import qtviews
from bornagain.viewers import joesviews
from bornagain.target import crystal, map
import bornagain.simulate.clcore as core



pdbFile = '../data/pdb/1JB0.pdb'
print('Loading pdb file (%s)' % pdbFile)
cryst = crystal.structure(pdbFile)
print(cryst.cryst1.strip())

if 0:
    # Show atoms in scatter plot
    r = cryst.r
    sv = qtviews.Scatter3D()
    sv.add_points(r)
    sv.show()


# Look up atomic scattering factors (they are complex numbers)
print('Getting scattering factors')
wavelength = 1.5e-10
f = ba.simulate.atoms.get_scattering_factors(cryst.Z, ba.units.hc / wavelength)

print('Setting up 3D mesh')
d = 0.5e-9  # Minimum resolution
s = 2       # Oversampling factor
mt = map.CrystalMeshTool(cryst, d, s)
print('Grid size: (%d, %d, %d)' % (mt.N, mt.N, mt.N))
h = mt.get_h_vecs()  # Miller indices (fractional)

if 0:  # Display the mesh as a scatterplot

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
    F = mt.reshape(F)
    I = np.abs(F) ** 2
    rho = np.fft.ifftn(F)
    rho = np.abs(rho)

    rho_cell = mt.zeros()
    for i in range(0, len(mt.get_sym_luts())):
        rho_cell += mt.symmetry_transform(0, i, rho)


else:  # This is the direct way of making a density map from atoms and their structure factors

    print('Creating density map directly from atoms')
    x = cryst.x
    rho = mt.place_atoms_in_map(cryst.x % mt.s, np.abs(f))
    F = np.fft.fftn(rho)
    I = np.abs(F)**2
    rho_cell = mt.zeros()
    for i in range(0, len(mt.get_sym_luts())):
        rho_cell += mt.symmetry_transform(0, i, rho)

rho_cell = np.fft.fftshift(rho_cell)

if 1:

    print('Showing 3D volumetric rendering of unit cell')
    vol = qtviews.Volumetric3D()
    for i in range(0, len(mt.get_sym_luts())):
        vol.add_density(fftshift(mt.symmetry_transform(0, i, rho)), qtviews.bright_colors(i))
    vol.show()

if 0:  # Joe's isosurface function

    joesviews.meshplot3D(rho_cell, np.max(rho_cell)*0.05)

if 1:

    print('Showing orthogonal slices of diffraction intensities and density projections')
    fig = plt.figure()

    fig.add_subplot(2, 3, 1)
    dispim = np.log10(np.fft.fftshift(I)[np.ceil(mt.N/2).astype(np.int), :, :] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 4)
    dispim = np.sum(rho_cell, axis=0)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 2)
    dispim = np.log10(np.fft.fftshift(I)[:, np.ceil(mt.N/2).astype(np.int), :] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 5)
    dispim = np.sum(rho_cell, axis=1)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 3)
    dispim = np.log10(np.fft.fftshift(I)[:, :, np.ceil(mt.N/2).astype(np.int)] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 6)
    dispim = np.sum(rho_cell, axis=2)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    plt.show()