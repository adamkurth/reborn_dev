from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from bornagain.simulate.examples import lysozyme_pdb_file, psi_pdb_file
from bornagain.target import crystal, density

try:
    from bornagain.target import density_f
except ImportError:
    density_f = None


def test_crystal_density():

    cryst = crystal.CrystalStructure(psi_pdb_file)
    # Manually reconfigure P1 with rectangular lattice
    cryst.unitcell = crystal.UnitCell(1e-9, 2e-9, 4e-9, np.pi/2, np.pi/2, np.pi/2)
    cryst.spacegroup.sym_translations = [np.zeros((3,))]
    cryst.spacegroup.sym_rotations = [np.eye((3))]
    dens = density.CrystalDensityMap(cryst, 2e-9, 2)
    assert np.sum(np.abs(dens.n_vecs[8, :] - np.array([1, 0, 0]))) < 1e-8
    assert np.allclose(dens.x_vecs[9, :], np.array([1., 0., 0.5]))

    cryst = crystal.CrystalStructure(lysozyme_pdb_file)
    for d in np.array([5, 10])*1e-10:

        dens = density.CrystalDensityMap(cryst, d, 1)
        dat0 = dens.reshape(np.arange(0, dens.n_voxels)).astype(np.float)
        dat1 = dens.symmetry_transform(0, 1, dat0)
        dat2 = dens.symmetry_transform(1, 0, dat1)

        assert np.allclose(dat0, dat2)

    cryst = crystal.CrystalStructure(psi_pdb_file)
    for d in np.array([5, 10])*1e-10:

        dens = density.CrystalDensityMap(cryst, d, 2)
        dat0 = dens.reshape(np.arange(0, dens.n_voxels)).astype(np.float)
        dat1 = dens.symmetry_transform(0, 1, dat0)
        dat2 = dens.symmetry_transform(1, 0, dat1)

        assert np.allclose(dat0, dat2)


def test_transforms():


    cryst = crystal.CrystalStructure(psi_pdb_file)

    for d in [0.2, 0.3, 0.4, 0.5]:

        mt = density.CrystalDensityMap(cryst, d, 1)
        zero = mt.zeros()
        dat0 = mt.reshape(np.arange(0, zero.size)).astype(np.float)
        dat1 = mt.symmetry_transform(0, 1, dat0)
        dat2 = mt.symmetry_transform(1, 0, dat1)

        assert np.allclose(dat0, dat2)


def func(vecs):
    return vecs[:, 0].ravel().copy()


def func1(vecs):
    return np.sin(vecs[:, 0]/10.0) + np.cos(3*vecs[:, 1]/10.) + np.cos(2*vecs[:, 2]/10.)


def test_interpolations():

    if density_f is None:
        return

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    dens = np.ones([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = np.ones((nx, 3), dtype=float_t)
    vectors[:, 0] = np.arange(0, nx).astype(float_t)
    dens2 = density.trilinear_interpolation(dens, vectors, corners, deltas)
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs(dens2[:] - 1)) < 1e-6

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
    vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t)
    dens = func(vectors0).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(np.arange(1, nx-2), np.arange(1, ny-2), np.arange(1, nz-2), indexing='ij')
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t) + 0.1
    dens1 = func(vectors)
    dens2 = np.zeros_like(dens1)
    density.trilinear_interpolation(dens, vectors, corners, deltas, dens2)
    assert np.max(np.abs(dens1)) > 0
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs((dens1 - dens2)/dens1)) < 1e-8

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
    vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t)
    dens = func1(vectors0).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(np.arange(1, nx-2), np.arange(1, ny-2), np.arange(1, nz-2), indexing='ij')
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t) + 0.1
    dens1 = func1(vectors)
    dens2 = np.zeros_like(dens1)
    density.trilinear_interpolation(dens, vectors, corners, deltas, dens2)
    assert np.max(np.abs(dens1)) > 0
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs((dens1 - dens2)/dens1)) < 1e-2


def test_insertions():

    if density_f is None:
        return

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = np.array([[2, 3, 4]], dtype=float_t)
    vals = func1(vectors)
    density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
    assert np.max(np.abs(densities)) > 0
    assert (np.abs((vals - densities[2, 3, 4]) / vals)) < 1e-8

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = np.array([[2.5, 3.5, 4.5]], dtype=float_t)
    vals = func1(vectors)
    density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
    assert np.max(np.abs(densities)) > 0
    assert (np.abs((vals - densities[2, 3, 4]/counts[2, 3, 4]) / vals)) < 1e-8

    np.random.seed(0)
    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = (np.random.rand(10000, 3) * np.array([nx-1, ny-1, nz-1])).astype(float_t)
    vectors = np.floor(vectors)
    vals = func(vectors)
    density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
    val = func(np.array([[2, 3, 4]], dtype=float_t))
    assert np.max(np.abs(densities)) > 0
    assert (np.abs((val - densities[2, 3, 4]/counts[2, 3, 4]) / val)) < 1e-8

    np.random.seed(0)
    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    corners = np.array([0, 0, 0], dtype=float_t)
    deltas = np.array([1, 1, 1], dtype=float_t)
    vectors = (np.random.rand(10000, 3) * np.array([nx-1, ny-1, nz-1])).astype(float_t)
    vals = func1(vectors)
    density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
    val = func1(np.array([[2, 3, 4]], dtype=float_t))
    assert np.max(np.abs(densities)) > 0
    assert (np.abs((val - densities[2, 3, 4]/counts[2, 3, 4]) / val)) < 1e-2
