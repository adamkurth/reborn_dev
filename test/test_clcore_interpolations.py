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

import numpy as np
# import pyopencl
from pyopencl import array as clarray
from reborn.simulate.clcore import ClCore #, help, print_device_info
cl_array = clarray.Array
try:
    test_core = ClCore(context=None, queue=None, group_size=1, double_precision=True)
except:
    test_core = ClCore(context=None, queue=None, group_size=1, double_precision=False)
from reborn.misc import interpolate

def test_nothing():
    assert test_core is not None
if test_core.double_precision:
    have_double = True
else:
    have_double = False

def func(vecs):
    return vecs[:, 0].ravel().copy()


def func1(vecs):
    return np.sin(vecs[:, 0]/10.0) + np.cos(3*vecs[:, 1]/10.) + np.cos(2*vecs[:, 2]/10.)


def test_interpolations_01():

    # if ClCore is None:
    #     return

    clcore = ClCore(group_size=32)
    shape = (6, 7, 8)
    dens = np.ones(shape, dtype=clcore.real_t)
    corners = np.array([0, 0, 0], dtype=clcore.real_t)
    deltas = np.array([1, 1, 1], dtype=clcore.real_t)
    vectors = np.ones((shape[0], 3), dtype=clcore.real_t)
    vectors[:, 0] = np.arange(0, shape[0]).astype(clcore.real_t)
    dens_gpu = clcore.to_device(dens)
    vectors_gpu = clcore.to_device(vectors)
    dens2 = clcore.mesh_interpolation(dens_gpu, vectors_gpu, q_min=corners, dq=deltas, N=shape)
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs(dens2[:] - 1)) < 1e-6


def test_interpolations_02():

    # if ClCore is None:
    #     return

    clcore = ClCore(group_size=32)
    shape = (6, 7, 8)
    corners = np.array([0, 0, 0], dtype=clcore.real_t)
    deltas = np.array([1, 1, 1], dtype=clcore.real_t)
    x, y, z = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]), np.arange(0, shape[2]), indexing='ij')
    vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(clcore.real_t)
    dens = func(vectors0).reshape(shape)
    x, y, z = np.meshgrid(np.arange(1, shape[0]-2), np.arange(1, shape[1]-2), np.arange(1, shape[2]-2), indexing='ij')
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(clcore.real_t) + 0.1
    dens1 = func(vectors)
    dens2 = np.zeros_like(dens1)
    dens2_gpu = clcore.to_device(dens2)
    clcore.mesh_interpolation(dens, vectors, q_min=corners, dq=deltas, N=shape, a=dens2_gpu)
    dens2 = dens2_gpu.get()
    assert np.max(np.abs(dens1)) > 0
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs((dens1 - dens2)/dens1)) < 1e-6


def test_interpolations_03():

    # if ClCore is None:
    #     return

    clcore = ClCore(group_size=32)
    nx, ny, nz = 6, 7, 8
    corners = np.array([0, 0, 0], dtype=clcore.real_t)
    deltas = np.array([1, 1, 1], dtype=clcore.real_t)
    x, y, z = np.meshgrid(np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing='ij')
    vectors0 = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(clcore.real_t)
    dens = func1(vectors0).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(np.arange(1, nx-2), np.arange(1, ny-2), np.arange(1, nz-2), indexing='ij')
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(clcore.real_t) + 0.1
    dens1 = func1(vectors)
    dens2 = np.zeros_like(dens1)
    dens2_gpu = clcore.to_device(dens2, dtype=clcore.real_t)
    clcore.mesh_interpolation(dens, vectors, q_min=corners, dq=deltas, N=dens.shape, a=dens2_gpu)
    dens2 = dens2_gpu.get()
    assert np.max(np.abs(dens1)) > 0
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs((dens1 - dens2)/dens1)) < 1e-2


def test_interpolations_04():
    core = ClCore(context=None, queue=None, group_size=1, double_precision=False)
    q_min = np.array([1, 2, 3])
    q_max = q_min + 1
    shape = np.array([2, 2, 2])
    dq = (q_max-q_min)/(shape-1)
    qx = np.arange(shape[0]) * dq[0] + q_min[0]
    qy = np.arange(shape[1]) * dq[1] + q_min[1]
    qz = np.arange(shape[2]) * dq[2] + q_min[2]
    qxx, qyy, qzz = np.meshgrid(qx, qy, qz, indexing='ij')
    q = np.vstack([qxx.ravel(), qyy.ravel(), qzz.ravel()]).T.copy()
    r = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    U = np.array([1, 2, 3])
    f = np.array([1, 2, 5])
    amps = core.phase_factor_mesh(r, f, q_min=q_min, q_max=q_max, N=shape, R=R, U=U)
    amps = amps.reshape(shape)
    interp1 = core.mesh_interpolation(a_map=amps, q=q, N=shape, q_min=q_min, q_max=q_max, dq=dq)
    # Check that GPU code agrees with CPU code
    interp2 = interpolate.trilinear_interpolation(amps.astype(np.complex128), q, corners=None, deltas=dq, x_min=q_min,
                                                  x_max=q_max)
    assert(np.max(np.abs(interp1-interp2)) == 0)

# def test_insertions_01():
#
#     if ClCore is None:
#         return
#
#     clcore = ClCore(group_size=32)
#     shape = (6, 7, 8)
#     densities = np.zeros(shape, dtype=clcore.real_t)
#     weights = np.zeros(shape, dtype=clcore.real_t)
#     corner = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     vecs = np.array([[2, 3, 4], [3, 4, 4], [4, 4, 4]], dtype=clcore.real_t)
#     vals = func1(vecs).astype(clcore.real_t)
#     densities_gpu = clcore.to_device(densities, dtype=clcore.real_t)
#     weights_gpu = clcore.to_device(weights, dtype=clcore.real_t)
#     vecs_gpu = clcore.to_device(vecs, dtype=clcore.real_t)
#     vals_gpu = clcore.to_device(vals, dtype=clcore.real_t)
#     clcore.mesh_insertion(densities_gpu, weights_gpu, vecs_gpu, vals_gpu, shape=shape, deltas=deltas, corner=corner)
#     densities = densities_gpu.get()
#     assert np.max(np.abs(densities)) > 0
#     assert np.max(np.abs((vals[0] - densities[2, 3, 4]) / vals)) < 1e-8


# def test_insertions_02():
#
#     if ClCore is None:
#         return
#
#     clcore = ClCore(group_size=32)
#     shape = (6, 7, 8)
#     densities = np.zeros(shape, dtype=clcore.real_t)
#     weights = np.zeros(shape, dtype=clcore.real_t)
#     corner = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     vecs = np.array([[2, 3, 4], [2.1, 3.1, 4.1], [1.9, 2.9, 3.9]], dtype=clcore.real_t)
#     vals = func1(vecs)
#     densities_gpu = clcore.to_device(densities, dtype=clcore.real_t)
#     weights_gpu = clcore.to_device(weights, dtype=clcore.real_t)
#     vecs_gpu = clcore.to_device(vecs, dtype=clcore.real_t)
#     vals_gpu = clcore.to_device(vals, dtype=clcore.real_t)
#     clcore.mesh_insertion(densities_gpu, weights_gpu, vecs_gpu, vals_gpu, shape=shape, corner=corner, deltas=deltas)
#     densities = densities_gpu.get()
#     weights = weights_gpu.get()
#     assert np.max(np.abs(densities)) > 0
#     assert np.max(np.abs((vals[0] - densities[2, 3, 4]/weights[2, 3, 4]) / vals[0])) < 1e-3


# def test_insertions_03():
#
#     if ClCore is None:
#         return
#
#     clcore = ClCore(group_size=32)
#     np.random.seed(0)
#     shape = (6, 7, 8)
#     densities = np.zeros(shape, dtype=clcore.real_t)
#     weights = np.zeros(shape, dtype=clcore.real_t)
#     corner = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     vecs = np.random.rand(1000, 3) * (np.array(shape)-1).astype(clcore.real_t)
#     vecs = np.floor(vecs)
#     vals = func(vecs)
#     densities_gpu = clcore.to_device(densities, dtype=clcore.real_t)
#     weights_gpu = clcore.to_device(weights, dtype=clcore.real_t)
#     vecs_gpu = clcore.to_device(vecs, dtype=clcore.real_t)
#     vals_gpu = clcore.to_device(vals, dtype=clcore.real_t)
#     clcore.mesh_insertion(densities_gpu, weights_gpu, vecs_gpu, vals_gpu, shape=shape, corner=corner, deltas=deltas)
#     val = func(np.array([[2, 3, 4]], dtype=clcore.real_t))
#     densities = densities_gpu.get()
#     weights = weights_gpu.get()
#     assert np.max(np.abs(densities)) > 0
#     assert (np.abs((val - densities[2, 3, 4]/weights[2, 3, 4]) / val)) < 1e-8


# def test_insertions_04():
#
#     if ClCore is None:
#         return
#
#     clcore = ClCore(group_size=32)
#     np.random.seed(0)
#     shape = (6, 7, 8)
#     densities = np.zeros(shape, dtype=clcore.real_t)
#     weights = np.zeros(shape, dtype=clcore.real_t)
#     corner = np.array([0, 0, 0], dtype=clcore.real_t)
#     deltas = np.array([1, 1, 1], dtype=clcore.real_t)
#     vecs = (np.random.rand(10000, 3) * (np.array(shape)-1)).astype(clcore.real_t)
#     vals = func1(vecs)
#     densities_gpu = clcore.to_device(densities, dtype=clcore.real_t)
#     weights_gpu = clcore.to_device(weights, dtype=clcore.real_t)
#     vecs_gpu = clcore.to_device(vecs, dtype=clcore.real_t)
#     vals_gpu = clcore.to_device(vals, dtype=clcore.real_t)
#     clcore.mesh_insertion(densities_gpu, weights_gpu, vecs_gpu, vals_gpu, shape=shape, corner=corner, deltas=deltas)
#     val = func1(np.array([[2, 3, 4]], dtype=clcore.real_t))
#     densities = densities_gpu.get()
#     weights = weights_gpu.get()
#     assert np.max(np.abs(densities)) > 0
#     assert (np.abs((val - densities[2, 3, 4]/weights[2, 3, 4]) / val)) < 1e-2
