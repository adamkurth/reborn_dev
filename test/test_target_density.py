from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys

import numpy as np

sys.path.append("..")
from bornagain.simulate.examples import lysozyme_pdb_file, psi_pdb_file
from bornagain.target import crystal, density

try:
    from bornagain.target import density_f
except ImportError:
    density_f = None


def test_crystal_density():

    cryst = crystal.CrystalStructure()
    cryst.set_cell(1e-9, 2e-9, 4e-9, np.pi/2, np.pi/2, np.pi/2)
    cryst.set_spacegroup(hall_number=1)
    dens = density.CrystalDensityMap(cryst, 2e-9, 2)
    assert (np.sum(np.abs(dens.n_vecs[10, :] - np.array([0, 1, 0]))) < 1e-8)
    assert(np.allclose(dens.x_vecs[11, :], np.array([0., 0.33333333, 0.2])))

    cryst = crystal.CrystalStructure(lysozyme_pdb_file)
    for d in [0.2, 0.3, 0.4, 0.5]:

        dens = density.CrystalDensityMap(cryst, d, 1)
        dat0 = dens.reshape(np.arange(0, dens.n_voxels)).astype(np.float)
        dat1 = dens.symmetry_transform(0, 1, dat0)
        dat2 = dens.symmetry_transform(1, 0, dat1)

        assert(np.allclose(dat0, dat2))

    cryst = crystal.CrystalStructure(psi_pdb_file)
    for d in [0.2, 0.3, 0.4, 0.5]:

        dens = density.CrystalDensityMap(cryst, d, 2)
        dat0 = dens.reshape(np.arange(0, dens.n_voxels)).astype(np.float)
        dat1 = dens.symmetry_transform(0, 1, dat0)
        dat2 = dens.symmetry_transform(1, 0, dat1)

        assert(np.allclose(dat0, dat2))

    assert(np.max(dens.x_limits - 1.5) == 0)


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


def func(vecs):
    return np.sin(vecs[:, 0]/100.0) + np.cos(3*vecs[:, 1]/100.) + np.cos(2*vecs[:, 2]/100.)


def test_interpolations():

    if density_f is None:
        return

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    lims = np.array([[0, nx-1], [0, ny-1], [0, nz-1]], dtype=float_t)
    dens = np.ones([nx, ny, nz], dtype=float_t)
    xyz = np.ones((nx, 3), dtype=float_t)
    xyz[:, 0] = np.arange(0, nx)
    dens = density.trilinear_interpolation(dens, xyz, lims)
    assert(np.max(np.abs(dens)) > 0)
    assert(np.max(np.abs(dens[1:-1] - 1)) < 1e-6)

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    lims = np.array([[0, nx-1], [0, ny-1], [0, nz-1]], dtype=float_t)
    x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
    xyz = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t)
    dens = func(xyz).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(np.arange(1, nx-1), np.arange(1, ny-1), np.arange(1, nz-1), indexing='ij')
    xyz = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(float_t)
    dens1 = func(xyz)
    dens2 = np.zeros_like(dens1)
    density.trilinear_interpolation(dens, xyz, lims, dens2)
    assert(np.max(np.abs(dens1)) > 0)
    assert(np.max(np.abs(dens2)) > 0)
    assert(np.max(np.abs((dens1 - dens2)/dens1)) < 1e-2)  # Interpolations only good to the 1% level.  Why?


def test_insertions():

    if density_f is None:
        return

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    lims = np.array([[0, nx-1], [0, ny-1], [0, nz-1]], dtype=float_t)
    xyz = np.array([[2, 3, 4]], dtype=float_t)
    vals = func(xyz)
    density.trilinear_insertion(densities, counts, xyz, vals, lims)
    assert(np.max(np.abs(densities)) > 0)
    assert((np.abs((vals - densities[2, 3, 4]) / vals)) < 1e-8)

    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    lims = np.array([[0, nx-1], [0, ny-1], [0, nz-1]], dtype=float_t)
    xyz = np.array([[2.5, 3.5, 4.5]], dtype=float_t)
    vals = func(xyz)
    density.trilinear_insertion(densities, counts, xyz, vals, lims)
    assert(np.max(np.abs(densities)) > 0)
    assert((np.abs((vals - densities[2, 3, 4]/counts[2, 3, 4]) / vals)) < 1e-8)

    np.random.seed(0)
    float_t = np.float64
    nx, ny, nz = 6, 7, 8
    densities = np.zeros([nx, ny, nz], dtype=float_t)
    counts = np.zeros([nx, ny, nz], dtype=float_t)
    lims = np.array([[0, nx-1], [0, ny-1], [0, nz-1]], dtype=float_t)
    xyz = (np.random.rand(10000, 3) * np.array([nx-1, ny-1, nz-1]) ).astype(float_t)
    vals = func(xyz)
    density.trilinear_insertion(densities, counts, xyz, vals, lims)
    val = func(np.array([[2, 3, 4]], dtype=float_t))
    assert(np.max(np.abs(densities)) > 0)
    assert((np.abs((val - densities[2, 3, 4]/counts[2, 3, 4]) / val)) < 1e-3)


def test_wtf():

    # This is important to note when using f2py:
    #
    # "In general, if a NumPy array is proper-contiguous and has a proper type then it is directly passed to wrapped
    # Fortran/C function. Otherwise, an element-wise copy of an input array is made and the copy, being
    # proper-contiguous and with proper type, is used as an array argument."
    #
    # The tests below show how the above can cause a lot of confusion...

    if density_f is None:
        return

    def wrap(out1, out2, out3):
        density_f.wtf(np.asfortranarray(out1), out2, np.asfortranarray(out3))

    out1 = np.zeros(10)
    out2 = np.zeros((10, 10))
    out3 = np.zeros((10, 10, 10))
    density_f.wtf(np.asfortranarray(out1), np.asfortranarray(out2), np.asfortranarray(out3))
    # print(out1.flags.f_contiguous) # True
    assert(out1[1] == 10)
    # print(out2.flags.f_contiguous) # False
    # assert(out2[1, 0] == 10) # Fails
    # print(out3.flags.f_contiguous) # False
    # assert(out3[1, 0, 0] == 10) # Fails

    out1 = np.zeros(10)
    out2_0 = np.zeros((10, 10))
    out2 = np.asfortranarray(out2_0)
    assert(out2_0.data == out2.data)
    out3 = np.zeros((10, 10, 10))
    density_f.wtf(np.asfortranarray(out1), out2, np.asfortranarray(out3))
    # print(out1.flags.f_contiguous) # True
    assert(out1[1] == 10)
    # print(out2.flags.f_contiguous) # True
    assert(out2[1, 0] == 10)
    # assert(out2_0.data == out2.data) # Fails: conclusion: this function rewrites memory
    # print(out3.flags.f_contiguous) # False
    # assert(out3[1, 0, 0] == 10) # Fails

    out1 = np.zeros(10)
    out2 = np.asfortranarray(np.zeros((10, 10)))
    out3 = np.zeros((10, 10, 10)).T  # This array is now fortran contiguous
    wrap(out1, out2, out3)
    # print(out1.flags.f_contiguous) # True
    assert(out1[1] == 10)
    # print(out2.flags.f_contiguous) # True
    assert(out2[1, 0] == 10)
    # print(out3.flags.f_contiguous) # True
    assert(out3[1, 0, 0] == 10)
