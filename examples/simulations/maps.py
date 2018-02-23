from __future__ import division

import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.stats import binned_statistic_dd

sys.path.append("../..")
import bornagain as ba
from bornagain.target import crystal, map
import bornagain.simulate.clcore as core

def meshplot3D(f, isosurface_value):
    # Use marching cubes to obtain a surface mesh
    if isosurface_value < np.min(f):
        isosurface_value = np.min(f)*1.1
    verts, faces = measure.marching_cubes(f, isosurface_value)

    # Display resulting triangular mesh using Matplotlib
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: "verts[faces]" to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    ax.add_collection3d(mesh)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim(0, f.shape[0])
    ax.set_ylim(0, f.shape[1])
    ax.set_zlim(0, f.shape[2])

    plt.show()


pdbFile = '../data/pdb/1JB0.pdb'
print('Loading pdb file (%s)' % pdbFile)
cryst = crystal.structure(pdbFile)
print(cryst.cryst1.strip())

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
print(mt.get_n_vecs())

if 0:
    # Display the mesh as a scatterplot
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


print('Simulating diffraction amplitudes')
clcore = core.ClCore(group_size=32, double_precision=False)
F = clcore.phase_factor_qrf(2*np.pi*h, cryst.x, f)
F = mt.reshape(F)
I = np.abs(F)**2
rho = np.fft.ifftn(F)
rho = np.abs(rho)

rho_cell = mt.zeros()

nops = len(mt.get_sym_luts())
for i in range(0, nops):
    rho_cell += mt.symmetry_transform(0, i, rho)

rho_cell = np.fft.fftshift(rho_cell)


if 0:
    # This is the direct way of making a density map from atoms and their structure factor
    x = cryst.x
    rho_cell = 0
    for i in range(0, nops):
        a, _, _ = binned_statistic_dd((np.dot(cryst.symRs[i], cryst.x.T).T + cryst.symTs[i]) % 1, np.abs(f), statistic='sum', bins=[mt.N]*3, range=[[0, mt.s], [0, mt.s], [0, mt.s]])
        rho_cell += a


# meshplot3D(rho_cell, np.max(rho_cell)*0.05)


if 1:

    fig = plt.figure()

    fig.add_subplot(2, 3, 1)
    dispim = np.log10(np.fft.fftshift(I)[np.ceil(mt.N/2), :, :] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 4)
    dispim = np.sum(rho_cell, axis=0)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 2)
    dispim = np.log10(np.fft.fftshift(I)[:, np.ceil(mt.N/2), :] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 5)
    dispim = np.sum(rho_cell, axis=1)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 3)
    dispim = np.log10(np.fft.fftshift(I)[:, :, np.ceil(mt.N/2)] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    fig.add_subplot(2, 3, 6)
    dispim = np.sum(rho_cell, axis=2)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')

    plt.show()