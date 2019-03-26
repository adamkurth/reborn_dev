from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys

import numpy as np

sys.path.append("..")
from bornagain.target import crystal, density


def test_transforms():

    cryst = crystal.CrystalStructure()
    cryst.set_spacegroup('P 63')
    cryst.set_cell(28e-9, 28e-9, 16e-9, 90*np.pi/180, 90*np.pi/180, 120*np.pi/180)

    for d in [0.2, 0.3, 0.4, 0.5]:

        mt = density.CrystalMeshTool(cryst, d, 1)
        dat0 = mt.reshape(np.arange(0, mt.N**3)).astype(np.float)
        dat1 = mt.symmetry_transform(0, 1, dat0)
        dat2 = mt.symmetry_transform(1, 0, dat1)

        assert(np.allclose(dat0, dat2))


def test_interpolations():

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    dens = np.ones([nx, ny, nz], dtype=float_t)
    lims = np.array([[0, nx-1], [0, ny-1], [0, nz-1]], dtype=float_t)
    xyz = np.ones((nx, 3), dtype=float_t)
    xyz[:, 0] = np.arange(0, nx)
    dens = density.trilinear_interpolation(dens, xyz, lims)
    assert(np.max(np.abs(dens[1:-1] - 1)) < 1e-6)

    def func(vecs):
        return np.sin(vecs[:, 0]/1000.0) + np.cos(3*vecs[:, 1]/1000.) + np.cos(2*vecs[:, 2]/1000.)

    nx, ny, nz = 6, 7, 8
    lims = np.array([[0, nx-1], [0, ny-1], [0, nz-1]], dtype=float_t)
    x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
    xyz = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t)
    dens = func(xyz).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(np.arange(1, nx-1), np.arange(1, ny-1), np.arange(1, nz-1), indexing='ij')
    xyz = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t)
    dens1 = func(xyz)
    dens2 = density.trilinear_interpolation(dens, xyz, lims)
    assert(np.max(np.abs(dens1 - dens2)) < 1e-3)
    xyz = np.array([[2, 3, 4]])
    vals = np.array([func(xyz)])
    dens3 = np.zeros_like(dens)
    counts3 = np.zeros_like(dens)
    density.trilinear_insertion(dens3, counts3, xyz, vals, lims)
    assert(dens3[xyz[0], xyz[1], xyz[2]]/counts3[xyz[0], xyz[1], xyz[2]] == vals[0])
