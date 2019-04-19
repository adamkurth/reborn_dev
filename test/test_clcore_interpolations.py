from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys

import numpy as np

# sys.path.append("..")
import bornagain
from bornagain.simulate.examples import lysozyme_pdb_file, psi_pdb_file
from bornagain.target import crystal, density
# from bornagain.simulate.clcore import ClCore

try:
    import pyopencl
    from pyopencl import array as clarray
    from bornagain.simulate.clcore import ClCore
    cl_array = clarray.Array
    havecl = True
    # Check for double precision:
    test_core = ClCore(context=None, queue=None, group_size=1, double_precision=True)
    if test_core.double_precision:
        have_double = True
    else:
        have_double = False
    ctx = bornagain.simulate.clcore.create_some_gpu_context()
except ImportError:
    ClCore = None
    clarray = None
    pyopencl = None
    havecl = False

# try:
#     from bornagain.target import density_f
# except ImportError:
#     density_f = None


# def test_crystal_density():
#
#     cryst = crystal.CrystalStructure()
#     cryst.set_cell(1e-9, 2e-9, 4e-9, np.pi/2, np.pi/2, np.pi/2)
#     cryst.set_spacegroup(hall_number=1)
#     dens = density.CrystalDensityMap(cryst, 2e-9, 2)
#     assert np.sum(np.abs(dens.n_vecs[8, :] - np.array([1, 0, 0]))) < 1e-8
#     assert np.allclose(dens.x_vecs[9, :], np.array([1., 0., 0.5]))
#
#     cryst = crystal.CrystalStructure(lysozyme_pdb_file)
#     for d in np.array([5, 10])*1e-10:
#
#         dens = density.CrystalDensityMap(cryst, d, 1)
#         dat0 = dens.reshape(np.arange(0, dens.n_voxels)).astype(np.float)
#         dat1 = dens.symmetry_transform(0, 1, dat0)
#         dat2 = dens.symmetry_transform(1, 0, dat1)
#
#         assert np.allclose(dat0, dat2)
#
#     cryst = crystal.CrystalStructure(psi_pdb_file)
#     for d in np.array([5, 10])*1e-10:
#
#         dens = density.CrystalDensityMap(cryst, d, 2)
#         dat0 = dens.reshape(np.arange(0, dens.n_voxels)).astype(np.float)
#         dat1 = dens.symmetry_transform(0, 1, dat0)
#         dat2 = dens.symmetry_transform(1, 0, dat1)
#
#         assert np.allclose(dat0, dat2)
#
#
# def test_transforms():
#
#     cryst = crystal.CrystalStructure()
#     cryst.set_spacegroup('P 63')
#     cryst.set_cell(28e-9, 28e-9, 16e-9, 90*np.pi/180, 90*np.pi/180, 120*np.pi/180)
#
#     for d in [0.2, 0.3, 0.4, 0.5]:
#
#         mt = density.CrystalMeshTool(cryst, d, 1)
#         dat0 = mt.reshape(np.arange(0, mt.N**3)).astype(np.float)
#         dat1 = mt.symmetry_transform(0, 1, dat0)
#         dat2 = mt.symmetry_transform(1, 0, dat1)
#
#         assert np.allclose(dat0, dat2)

def func(vecs):
    return vecs[:, 0].ravel().copy()


def func1(vecs):
    return np.sin(vecs[:, 0]/10.0) + np.cos(3*vecs[:, 1]/10.) + np.cos(2*vecs[:, 2]/10.)


def test_interpolations():

    if ClCore is None:
        return

    clcore = ClCore(group_size=32)
    real_t = clcore.real_t

    nx, ny, nz = 6, 7, 8
    dens = np.ones([nx, ny, nz], dtype=real_t)
    corners = np.array([0, 0, 0], dtype=real_t)
    deltas = np.array([1, 1, 1], dtype=real_t)
    vectors = np.ones((nx, 3), dtype=real_t)
    vectors[:, 0] = np.arange(0, nx).astype(real_t)
    dens_gpu = clcore.to_device(dens)
    vectors_gpu = clcore.to_device(vectors)
    dens2 = clcore.mesh_interpolation(dens_gpu, vectors_gpu, q_min=corners, dq=deltas, N=dens.shape)
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs(dens2[:] - 1)) < 1e-6

    nx, ny, nz = 6, 7, 8
    corners = np.array([0, 0, 0], dtype=real_t)
    deltas = np.array([1, 1, 1], dtype=real_t)
    x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
    vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(real_t)
    dens = func(vectors0).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(np.arange(1, nx-2), np.arange(1, ny-2), np.arange(1, nz-2), indexing='ij')
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(real_t) + 0.1
    dens1 = func(vectors)
    dens2 = np.zeros_like(dens1)
    dens2_gpu = clcore.to_device(dens2)
    clcore.mesh_interpolation(dens, vectors, q_min=corners, dq=deltas, N=dens.shape, a=dens2_gpu)
    dens2 = dens2_gpu.get()
    assert np.max(np.abs(dens1)) > 0
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs((dens1 - dens2)/dens1)) < 1e-6

    nx, ny, nz = 6, 7, 8
    corners = np.array([0, 0, 0], dtype=real_t)
    deltas = np.array([1, 1, 1], dtype=real_t)
    x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
    vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(real_t)
    dens = func1(vectors0).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(np.arange(1, nx-2), np.arange(1, ny-2), np.arange(1, nz-2), indexing='ij')
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(real_t) + 0.1
    dens1 = func1(vectors)
    dens2 = np.zeros_like(dens1)
    dens2_gpu = clcore.to_device(dens2)
    clcore.mesh_interpolation(dens, vectors, q_min=corners, dq=deltas, N=dens.shape, a=dens2_gpu)
    dens2 = dens2_gpu.get()
    assert np.max(np.abs(dens1)) > 0
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs((dens1 - dens2)/dens1)) < 1e-2


def test_insertions():

    if ClCore is None:
        return

    clcore = ClCore(group_size=32)
    real_t = clcore.real_t
    int_t = clcore.int_t

    n = 101
    a = np.zeros(n)
    b = np.arange(n)
    a_gpu = clcore.to_device(a, dtype=int_t)
    b_gpu = clcore.to_device(b, dtype=int_t)
    clcore.test_atomic_add_int(a_gpu, b_gpu)
    assert a_gpu.get()[0] - np.sum(b)*n == 0

    n = 100
    a = np.zeros(n)
    b = np.arange(n)
    a_gpu = clcore.to_device(a, dtype=real_t)
    b_gpu = clcore.to_device(b, dtype=real_t)
    clcore.test_atomic_add_real(a_gpu, b_gpu)
    assert a_gpu.get()[0] - np.sum(b)*n == 0

    print(a_gpu.get()[0], np.sum(b)*n)

    return

    shape = (6, 7, 8)
    densities = np.zeros(shape, dtype=real_t)
    weights = np.zeros(shape, dtype=real_t)
    corner = np.array([0, 0, 0], dtype=real_t)
    deltas = np.array([1, 1, 1], dtype=real_t)
    vecs = np.array([[2, 3, 4], [3, 4, 4], [4, 4, 4]], dtype=real_t)
    vals = func1(vecs)
    densities_gpu = clcore.to_device(densities)
    weights_gpu = clcore.to_device(weights)
    vecs_gpu = clcore.to_device(vecs)
    vals_gpu = clcore.to_device(vals)
    clcore.mesh_insertion(densities_gpu, weights_gpu, vecs_gpu, vals_gpu, shape, deltas, corner, rot=None) #, trans, do_trans)
    densities = densities_gpu.get()
    print(densities[4, 4, 4], densities[3, 3, 3], np.max(densities), np.min(densities))
    assert np.max(np.abs(densities)) > 0
    assert (np.abs((vals - densities[2, 3, 4]) / vals)) < 1e-8
#
#     real_t = np.float64
#     nx, ny, nz = 6, 7, 8
#     densities = np.zeros([nx, ny, nz], dtype=real_t)
#     counts = np.zeros([nx, ny, nz], dtype=real_t)
#     corners = np.array([0, 0, 0], dtype=real_t)
#     deltas = np.array([1, 1, 1], dtype=real_t)
#     vectors = np.array([[2.5, 3.5, 4.5]], dtype=real_t)
#     vals = func1(vectors)
#     density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
#     assert np.max(np.abs(densities)) > 0
#     assert (np.abs((vals - densities[2, 3, 4]/counts[2, 3, 4]) / vals)) < 1e-8
#
#     np.random.seed(0)
#     real_t = np.float64
#     nx, ny, nz = 6, 7, 8
#     densities = np.zeros([nx, ny, nz], dtype=real_t)
#     counts = np.zeros([nx, ny, nz], dtype=real_t)
#     corners = np.array([0, 0, 0], dtype=real_t)
#     deltas = np.array([1, 1, 1], dtype=real_t)
#     vectors = (np.random.rand(10000, 3) * np.array([nx-1, ny-1, nz-1])).astype(real_t)
#     vectors = np.floor(vectors)
#     vals = func(vectors)
#     density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
#     val = func(np.array([[2, 3, 4]], dtype=real_t))
#     assert np.max(np.abs(densities)) > 0
#     assert (np.abs((val - densities[2, 3, 4]/counts[2, 3, 4]) / val)) < 1e-8
#
#     np.random.seed(0)
#     real_t = np.float64
#     nx, ny, nz = 6, 7, 8
#     densities = np.zeros([nx, ny, nz], dtype=real_t)
#     counts = np.zeros([nx, ny, nz], dtype=real_t)
#     corners = np.array([0, 0, 0], dtype=real_t)
#     deltas = np.array([1, 1, 1], dtype=real_t)
#     vectors = (np.random.rand(10000, 3) * np.array([nx-1, ny-1, nz-1])).astype(real_t)
#     vals = func1(vectors)
#     density.trilinear_insertion(densities, counts, vectors, vals, corners, deltas)
#     val = func1(np.array([[2, 3, 4]], dtype=real_t))
#     assert np.max(np.abs(densities)) > 0
#     assert (np.abs((val - densities[2, 3, 4]/counts[2, 3, 4]) / val)) < 1e-2
#
#
# def test_wtf():
#
#     # This is important to note when using f2py:
#     #
#     # "In general, if a NumPy array is proper-contiguous and has a proper type then it is directly passed to wrapped
#     # Fortran/C function. Otherwise, an element-wise copy of an input array is made and the copy, being
#     # proper-contiguous and with proper type, is used as an array argument."
#     #
#     # The tests below show how the above can cause a lot of confusion...
#
#     if density_f is None:
#         return
#
#     # The fortran function "wtf" does the following:
#     # out1(2) = 10
#     # out2(2,1) = 10
#     # out3(2,1,1) = 10
#
#     out1 = np.zeros(10)
#     out2 = np.zeros((10, 10))
#     out3 = np.zeros((10, 10, 10))
#     density_f.wtf(np.asfortranarray(out1), np.asfortranarray(out2), np.asfortranarray(out3))
#     assert out1.flags.f_contiguous
#     assert out1[1] == 10  # A 1D array can be passed into a fortran function and the function can modify the data
#     assert out2.flags.c_contiguous
#     assert out2[1, 0] != 10  # Look: the asfortranarray function will not let you modify the data; it makes a copy
#     assert out3.flags.c_contiguous
#     assert out3[1, 0, 0] != 10  # Once again, a copy is made.  Note that this issue pertains to multi-dimensional arrays
#
#     out1 = np.zeros(10)
#     out2_0 = np.zeros((10, 10))
#     out2 = np.asfortranarray(out2_0)
#     assert out2_0.data == out2.data  # This line shows that asfortranarray does not make a data copy immediately
#     out3 = np.zeros((10, 10, 10))
#     density_f.wtf(np.asfortranarray(out1), out2, np.asfortranarray(out3))
#     assert out1.flags.f_contiguous
#     assert out1[1] == 10
#     assert out2.flags.f_contiguous
#     assert out2[1, 0] == 10
#     assert out2_0.data != out2.data  # Compare to the above - the wtf function
#     assert not out3.flags.f_contiguous
#     assert out3[1, 0, 0] != 10
#
#     out1 = np.zeros(10)
#     out2 = np.asfortranarray(np.zeros((10, 10)))
#     out3 = np.zeros((10, 10, 10)).T  # This array is now fortran contiguous, as a result of the transpose.
#     density_f.wtf(out1, out2, out3)
#     assert out1.flags.f_contiguous
#     assert out1[1] == 10
#     assert out2.flags.f_contiguous
#     assert out2[1, 0] == 10
#     assert out3.flags.f_contiguous
#     assert out3[1, 0, 0] == 10
