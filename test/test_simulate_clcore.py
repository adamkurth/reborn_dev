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

import pyopencl
import numpy as np
from reborn import source, detector, utils
from reborn.misc import interpolate
from reborn.simulate import clcore


test_core = clcore.ClCore()
have_double = test_core.double_precision_is_available()
ctx = clcore.create_some_gpu_context()


def func(vecs):
    return vecs[:, 0].ravel().copy()


def func1(vecs):
    return (
        np.sin(vecs[:, 0] / 10.0)
        + np.cos(3 * vecs[:, 1] / 10.0)
        + np.cos(2 * vecs[:, 2] / 10.0)
    )


def test_clcore_float():
    _clcore(double_precision=False)
    _test_rotations(double_precision=False)
    # _test_ridiculous_sum(double_precision=False)


def test_clcore_double():
    _clcore(double_precision=True)
    _test_rotations(double_precision=True)
    # _test_ridiculous_sum(double_precision=True)


def test_clmath():
    ctx = clcore.create_some_gpu_context()

    q = pyopencl.CommandQueue(ctx)

    a = np.random.random((5, 5)).astype(np.complex64)
    a_gpu = pyopencl.array.to_device(q, a)

    b = np.random.random((5, 5)).astype(np.complex64)
    b_gpu = pyopencl.array.to_device(q, b)

    # Test addition
    c = a + b
    c_gpu = a_gpu + b_gpu
    c_gpu = c_gpu.get()
    assert np.max(np.abs(c - c_gpu)) < 1e-6

    # Test multiplication
    c = a * b
    c_gpu = a_gpu * b_gpu
    c_gpu = c_gpu.get()
    assert np.max(np.abs(c - c_gpu)) < 1e-6

    # Test exponentiation
    c = a**2
    c_gpu = a_gpu**2
    c_gpu = c_gpu.get()
    assert np.max(np.abs(c - c_gpu)) < 1e-6


def _clcore(double_precision=False):
    ###########################################################################
    # Setup the simulation core
    ###########################################################################

    core = clcore.ClCore(
        context=None, queue=None, group_size=1, double_precision=have_double
    )
    # core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=double_precision)

    # if double_precision is False:
    #     numbacore.real_t = np.float32
    #     numbacore.complex_t = np.complex64

    assert core.get_group_size() == core.group_size

    # print("Using group size: %d" %core.group_size)
    ###########################################################################
    # Check that there are no errors in phase_factor_qrf_inplace
    # TODO: check that the amplitudes are correct
    ###########################################################################

    beam = source.Beam(wavelength=1e-10)
    pad = detector.PADGeometry(shape=(4, 4), pixel_size=1, distance=1)
    n_atoms = 10
    rot = np.eye(3, dtype=core.real_t)
    q = pad.q_vecs(beam=beam)

    r_vecs = np.random.random([n_atoms, 3])
    f = np.random.random([n_atoms]) * 1j

    amps = core.phase_factor_qrf(q, r_vecs, f, R=rot)
    assert type(amps) is np.ndarray

    # ampsn = numbacore.phase_factor_qrf(q, r_vecs, f, R=rot)

    # assert(np.max(np.abs(amps - ampsn)) < 1e-3)  # Why such a big difference between CPU and GPU?

    # make device arrays first
    q = core.to_device(q)
    r_vecs = core.to_device(r_vecs)
    f = core.to_device(f, dtype=core.complex_t)
    a = core.to_device(shape=(q.shape[0],), dtype=core.complex_t)
    rot = None

    core.phase_factor_qrf(q, r_vecs, f, R=rot, a=a)
    amps1 = a.get()

    for _ in range(9):
        core.phase_factor_qrf(q, r_vecs, f, R=rot, a=a, add=True)

    amps10 = a.get()

    assert np.allclose(10 * amps1, amps10)

    del q, r_vecs, f, rot, n_atoms

    ###########################################################################
    # Check for errors in phase_factor_pad
    # TODO: check that amplitudes are correct
    ###########################################################################

    n_atoms = 10
    r_vecs = np.random.random([n_atoms, 3]).astype(dtype=core.real_t)
    f = np.random.random([n_atoms]).astype(dtype=core.complex_t)
    t_vec = np.array([0, 0, 0], dtype=core.real_t)
    f_vec = np.array([1, 0, 0], dtype=core.real_t)
    s_vec = np.array([0, 1, 0], dtype=core.real_t)
    b_vec = np.array([0, 0, 1], dtype=core.real_t)
    n_f = 3
    n_s = 4
    w = 1
    rot = np.eye(3, dtype=core.real_t)

    amps = core.phase_factor_pad(
        r_vecs, f, t_vec, f_vec, s_vec, b_vec, n_f, n_s, w, R=rot, a=None
    )
    assert type(amps) is np.ndarray

    r_vecs = core.to_device(r_vecs)
    f = core.to_device(f)
    a = core.to_device(shape=(n_f * n_s,), dtype=core.complex_t)

    amps = core.phase_factor_pad(
        r_vecs, f, t_vec, f_vec, s_vec, b_vec, n_f, n_s, w, R=rot, a=a
    )

    del n_atoms, r_vecs, f, t_vec, f_vec, s_vec, b_vec, n_f, n_s, w, rot, a, amps

    ###########################################################################
    # Check for errors in phase_factor_mesh
    # TODO: check that amplitudes are correct
    ###########################################################################

    n_atoms = 10
    n_mesh = np.array([2, 3, 4])
    q_min = np.array([-10] * 3)
    q_max = np.array([+10] * 3)
    r_vecs = np.random.random([n_atoms, 3])
    f = np.random.random([n_atoms]) * 1j

    amps = core.phase_factor_mesh(r_vecs, f, n_mesh, q_min, q_max)
    assert type(amps) is np.ndarray

    r_vecs = core.to_device(r_vecs)
    f = core.to_device(f)
    a = core.to_device(shape=(np.prod(n_mesh),), dtype=core.complex_t)

    amps = core.phase_factor_mesh(r_vecs, f, N=n_mesh, q_min=q_min, q_max=q_max, a=a)
    assert type(amps) is pyopencl.array.Array

    del n_atoms, n_mesh, q_min, q_max, r_vecs, f, amps, a

    ###########################################################################
    # Check for errors in mesh_interpolation
    # TODO: check that amplitudes are correct
    ###########################################################################

    n_atoms = 10
    n_mesh = np.array([2, 3, 4])
    q_min = np.array([-10] * 3)
    q_max = np.array([+10] * 3)
    r_vecs = np.random.random([n_atoms, 3])
    f = np.random.random([n_atoms]) * 1j

    pad = detector.PADGeometry(shape=(4, 4), pixel_size=1, distance=1)
    rot = np.eye(3, dtype=core.real_t)
    q = pad.q_vecs(beam=beam)

    amps = core.phase_factor_mesh(r_vecs, f, N=n_mesh, q_min=q_min, q_max=q_max)
    amps2 = core.mesh_interpolation(amps, q, N=n_mesh, q_min=q_min, q_max=q_max)
    assert type(amps) is np.ndarray
    assert type(amps2) is np.ndarray

    r_vecs = core.to_device(r_vecs)
    f = core.to_device(f)
    q = core.to_device(q)
    a = core.to_device(shape=(np.prod(n_mesh),), dtype=core.complex_t)
    a_out = core.to_device(shape=pad.n_pixels, dtype=core.complex_t)

    amps = core.phase_factor_mesh(r_vecs, f, N=n_mesh, q_min=q_min, q_max=q_max, a=a)
    amps2 = core.mesh_interpolation(
        a, q, N=n_mesh, q_min=q_min, q_max=q_max, R=rot, a=a_out
    )
    assert type(amps) is pyopencl.array.Array
    assert type(amps2) is pyopencl.array.Array

    del n_atoms, n_mesh, q_min, q_max, r_vecs, f, amps, amps2, a, q, a_out

    ###########################################################################
    # Check phase_factor_qrf_chunk_r
    ###########################################################################

    pad = detector.PADGeometry(shape=(4, 4), pixel_size=1, distance=1)
    n_atoms = 10
    rot = np.eye(3, dtype=core.real_t)
    q = pad.q_vecs(beam=beam)
    r_vecs = np.random.random([n_atoms, 3])
    f = np.random.random([n_atoms]) * 1j
    q_dev = core.to_device(q, dtype=core.real_t)
    # r_dev = core.to_device(r, dtype=core.real_t)
    # f_dev = core.to_device(f, dtype=core.complex_t)

    q1 = q_dev.get()
    q2 = q_dev.get()
    assert np.allclose(q1, q2)

    amps0 = core.phase_factor_qrf(q, r_vecs, f, rot)

    a = core.phase_factor_qrf(q, r_vecs, f, rot, n_chunks=3)
    assert type(a) is np.ndarray

    a_dev = core.to_device(shape=(q.shape[0],), dtype=core.complex_t)

    core.phase_factor_qrf(q, r_vecs, f, rot, a=a_dev, add=False, n_chunks=2)
    amps1 = a_dev.get()

    assert np.allclose(amps0, amps1)

    for _ in range(9):
        core.phase_factor_qrf(q, r_vecs, f, rot, a=a_dev, add=True, n_chunks=2)

    amps10 = a_dev.get()

    assert np.allclose(10 * amps1, amps10)

    core.phase_factor_qrf(q, r_vecs, f, rot, a=a_dev, add=False, n_chunks=3)
    amps1_3 = a_dev.get()

    assert np.allclose(amps1, amps1_3)

    for _ in range(9):
        core.phase_factor_qrf(q, r_vecs, f, rot, a=a_dev, add=True, n_chunks=3)

    amps10_3 = a_dev.get()

    assert np.allclose(10 * amps1_3, amps10_3)

    del (
        q,
        r_vecs,
        f,
        rot,
        n_atoms,
        q_dev,
        a_dev,
        amps0,
        amps1,
        amps10,
        amps1_3,
        amps10_3,
        pad,
    )

    ###########################################################################
    # Check that rotations and translations work
    ###########################################################################

    n_pixels = 20
    pixel_size = 300e-6
    distance = 0.5
    wavelength = 2e-10
    pad = detector.PADGeometry(
        shape=(n_pixels, n_pixels), pixel_size=pixel_size, distance=distance
    )
    beam = source.Beam(wavelength=wavelength)
    q = pad.q_vecs(beam=beam)

    rot = utils.rotation_about_axis(0.1, [1, 1, 0]).astype(core.real_t)
    trans = (np.array([1, 0, 1]) * 1e-9).astype(core.real_t)

    np.random.seed(0)
    n_atoms = 10
    r0 = np.random.random([n_atoms, 3]).astype(core.real_t) * 5e-9
    f = np.random.random([n_atoms]).astype(core.complex_t)

    # Do rotation and translation on CPU
    amps1 = core.phase_factor_qrf(q, np.dot(r0, rot.T) + trans, f, R=None, U=None)
    # Rotation and translation on GPU
    amps2 = core.phase_factor_qrf(q, r0, f, R=rot, U=trans)
    # Rotation on CPU, translation on GPU
    amps3 = core.phase_factor_qrf(q, np.dot(r0, rot.T), f, R=None, U=trans)
    # Rotation on GPU, translation on CPU
    amps4 = core.phase_factor_qrf(q, r0 + np.dot(trans, rot), f, R=rot, U=None)

    if double_precision:
        tol = 1e-6
        assert np.mean(np.abs(amps1 - amps2)) / np.mean(np.abs(amps1)) < tol
        assert np.mean(np.abs(amps1 - amps3)) / np.mean(np.abs(amps3)) < tol
        assert np.mean(np.abs(amps3 - amps2)) / np.mean(np.abs(amps2)) < tol
        assert np.mean(np.abs(amps3 - amps4)) / np.mean(np.abs(amps4)) < tol
    else:
        tol = 1e-6
        assert np.mean(np.abs(amps1 - amps2)) / np.mean(np.abs(amps1)) < tol
        assert np.mean(np.abs(amps1 - amps3)) / np.mean(np.abs(amps3)) < tol
        assert np.mean(np.abs(amps3 - amps2)) / np.mean(np.abs(amps2)) < tol
        assert np.mean(np.abs(amps3 - amps4)) / np.mean(np.abs(amps4)) < tol

    # Do rotation and translation on CPU
    amps1 = core.phase_factor_pad(
        np.dot(r0, rot.T) + trans, f, R=None, U=None, beam=beam, pad=pad
    )
    # Rotation and translation on GPU
    amps2 = core.phase_factor_pad(r0, f, R=rot, U=trans, beam=beam, pad=pad)
    # Rotation on CPU, translation on GPU
    amps3 = core.phase_factor_pad(
        np.dot(r0, rot.T), f, R=None, U=trans, beam=beam, pad=pad
    )
    # Rotation on GPU, translation on CPU
    amps4 = core.phase_factor_pad(
        r0 + np.dot(trans, rot), f, R=rot, U=None, beam=beam, pad=pad
    )

    if double_precision:
        tol = 1e-6
        assert np.mean(np.abs(amps1 - amps2)) / np.mean(np.abs(amps1)) < tol
        assert np.mean(np.abs(amps1 - amps3)) / np.mean(np.abs(amps3)) < tol
        assert np.mean(np.abs(amps3 - amps2)) / np.mean(np.abs(amps2)) < tol
        assert np.mean(np.abs(amps3 - amps4)) / np.mean(np.abs(amps4)) < tol
    else:
        tol = 1e-6
        assert np.mean(np.abs(amps1 - amps2)) / np.mean(np.abs(amps1)) < tol
        assert np.mean(np.abs(amps1 - amps3)) / np.mean(np.abs(amps3)) < tol
        assert np.mean(np.abs(amps3 - amps2)) / np.mean(np.abs(amps2)) < tol
        assert np.mean(np.abs(amps3 - amps4)) / np.mean(np.abs(amps4)) < tol


def _test_rotations(double_precision=False):
    core = clcore.ClCore(
        context=None, queue=None, group_size=1, double_precision=double_precision
    )

    theta = 25 * np.pi / 180.0
    sin = np.sin(theta)
    cos = np.cos(theta)

    rot = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], dtype=core.real_t)
    trans = np.array([1, 2, 3], dtype=core.real_t)
    vec1 = np.array([1, 2, 0], dtype=core.real_t)

    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec3 = np.dot(vec1, rot.T) + trans

    # Rotation on gpu and rotation with utils.rotate should do the same thing
    assert np.max(np.abs(vec2 - vec3)) <= 1e-6

    vec1 = np.array([1, 2, 0], dtype=core.real_t)
    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec4 = np.random.rand(10, 3).astype(core.real_t)
    vec4[0, :] = vec1
    vec3 = np.dot(vec4, rot.T) + trans
    vec3 = vec3[0, :]

    # Rotation on gpu and rotation with utils.rotate should do the same thing (even for many vectors; shape Nx3)
    assert np.max(np.abs(vec2 - vec3)) <= 1e-6

    rot = np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]])
    trans = np.zeros((3,), dtype=core.real_t)
    vec1 = np.array([1.0, 0, 0], dtype=core.real_t)
    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec3 = np.dot(vec1, rot.T) + trans
    vec_pred = np.array([0, -1.0, 0])

    # Check that results are as expected
    assert np.max(np.abs(vec2 - vec3)) < 1e-6
    assert np.max(np.abs(vec2 - vec_pred)) < 1e-6


def test_rotations(double_precision=False):
    core = clcore.ClCore(
        context=None, queue=None, group_size=1, double_precision=double_precision
    )

    theta = 25 * np.pi / 180.0
    sin = np.sin(theta)
    cos = np.cos(theta)

    rot = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], dtype=core.real_t)
    trans = np.array([1, 2, 3], dtype=core.real_t)
    vec1 = np.array([1, 2, 0], dtype=core.real_t)

    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec3 = np.dot(vec1, rot.T) + trans

    # Rotation on gpu and rotation with utils.rotate should do the same thing
    assert np.max(np.abs(vec2 - vec3)) <= 1e-6

    vec1 = np.array([1, 2, 0], dtype=core.real_t)
    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec4 = np.random.rand(10, 3).astype(core.real_t)
    vec4[0, :] = vec1
    vec3 = np.dot(vec4, rot.T) + trans
    vec3 = vec3[0, :]

    # Rotation on gpu and rotation with utils.rotate should do the same thing (even for many vectors; shape Nx3)
    assert np.max(np.abs(vec2 - vec3)) <= 1e-6

    rot = np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]])
    trans = np.zeros((3,), dtype=core.real_t)
    vec1 = np.array([1.0, 0, 0], dtype=core.real_t)
    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec3 = np.dot(vec1, rot.T) + trans
    vec_pred = np.array([0, -1.0, 0])

    # Check that results are as expected
    assert np.max(np.abs(vec2 - vec3)) < 1e-6
    assert np.max(np.abs(vec2 - vec_pred)) < 1e-6


def test_phase_factors(double_precision=False):
    r"""
    Check that phase_factor_qrf gives the same results as phase_factor_mesh, as per documentation.

    Args:
        double_precision (bool): Check that double precision works if available
    """

    core = clcore.ClCore(
        context=None, queue=None, group_size=1, double_precision=double_precision
    )

    q_min = np.array([1, 2, 3])
    q_max = q_min + 1
    shape = np.array([2, 2, 2])
    dq = (q_max - q_min) / (shape - 1)
    qx = np.arange(shape[0]) * dq[0] + q_min[0]
    qy = np.arange(shape[1]) * dq[1] + q_min[1]
    qz = np.arange(shape[2]) * dq[2] + q_min[2]
    qxx, qyy, qzz = np.meshgrid(qx, qy, qz, indexing="ij")
    q = np.vstack([qxx.ravel(), qyy.ravel(), qzz.ravel()]).T.copy()
    r = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    U = np.array([1, 2, 3])
    f = np.array([1, 2, 5])
    amps1 = core.phase_factor_mesh(r, f, q_min=q_min, q_max=q_max, N=shape, R=R, U=U)
    amps2 = core.phase_factor_qrf(q, r, f, R=R, U=U)
    assert amps1[0] == amps2[0]
    rp = np.dot(r, R.T) + U
    amps0 = 0
    for n in range(r.shape[0]):
        amps0 += f[n] * np.exp(-1j * np.dot(q[0, :], rp[n, :]))
    assert np.abs(amps0 - amps2[0]) / np.abs(amps0) < 1e-4


def test_interpolations_01():
    # if ClCore is None:
    #     return

    core = clcore.ClCore(group_size=32)
    shape = (6, 7, 8)
    dens = np.ones(shape, dtype=core.real_t)
    corners = np.array([0, 0, 0], dtype=core.real_t)
    deltas = np.array([1, 1, 1], dtype=core.real_t)
    vectors = np.ones((shape[0], 3), dtype=core.real_t)
    vectors[:, 0] = np.arange(0, shape[0]).astype(core.real_t)
    dens_gpu = core.to_device(dens)
    vectors_gpu = core.to_device(vectors)
    dens2 = core.mesh_interpolation(
        dens_gpu, vectors_gpu, q_min=corners, dq=deltas, N=shape
    )
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs(dens2[:] - 1)) < 1e-6


def test_interpolations_02():
    # if ClCore is None:
    #     return

    core = clcore.ClCore(group_size=32)
    shape = (6, 7, 8)
    corners = np.array([0, 0, 0], dtype=core.real_t)
    deltas = np.array([1, 1, 1], dtype=core.real_t)
    x, y, z = np.meshgrid(
        np.arange(0, shape[0]),
        np.arange(0, shape[1]),
        np.arange(0, shape[2]),
        indexing="ij",
    )
    vectors0 = (
        (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(core.real_t)
    )
    dens = func(vectors0).reshape(shape)
    x, y, z = np.meshgrid(
        np.arange(1, shape[0] - 2),
        np.arange(1, shape[1] - 2),
        np.arange(1, shape[2] - 2),
        indexing="ij",
    )
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(
        core.real_t
    ) + 0.1
    dens1 = func(vectors)
    dens2 = np.zeros_like(dens1)
    dens2_gpu = core.to_device(dens2)
    core.mesh_interpolation(
        dens, vectors, q_min=corners, dq=deltas, N=shape, a=dens2_gpu
    )
    dens2 = dens2_gpu.get()
    assert np.max(np.abs(dens1)) > 0
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs((dens1 - dens2) / dens1)) < 1e-6


def test_interpolations_03():
    # if ClCore is None:
    #     return

    core = clcore.ClCore(group_size=32)
    nx, ny, nz = 6, 7, 8
    corners = np.array([0, 0, 0], dtype=core.real_t)
    deltas = np.array([1, 1, 1], dtype=core.real_t)
    x, y, z = np.meshgrid(
        np.arange(0, nx), np.arange(0, ny), np.arange(0, nz), indexing="ij"
    )
    vectors0 = (
        (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(core.real_t)
    )
    dens = func1(vectors0).reshape([nx, ny, nz])
    x, y, z = np.meshgrid(
        np.arange(1, nx - 2), np.arange(1, ny - 2), np.arange(1, nz - 2), indexing="ij"
    )
    vectors = (np.vstack([x.ravel(), y.ravel(), z.ravel()])).T.copy().astype(
        core.real_t
    ) + 0.1
    dens1 = func1(vectors)
    dens2 = np.zeros_like(dens1)
    dens2_gpu = core.to_device(dens2, dtype=core.real_t)
    core.mesh_interpolation(
        dens, vectors, q_min=corners, dq=deltas, N=dens.shape, a=dens2_gpu
    )
    dens2 = dens2_gpu.get()
    assert np.max(np.abs(dens1)) > 0
    assert np.max(np.abs(dens2)) > 0
    assert np.max(np.abs((dens1 - dens2) / dens1)) < 1e-2


def test_interpolations_04():
    core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=False)
    q_min = np.array([1, 2, 3])
    q_max = q_min + 1
    shape = np.array([2, 2, 2])
    dq = (q_max - q_min) / (shape - 1)
    qx = np.arange(shape[0]) * dq[0] + q_min[0]
    qy = np.arange(shape[1]) * dq[1] + q_min[1]
    qz = np.arange(shape[2]) * dq[2] + q_min[2]
    qxx, qyy, qzz = np.meshgrid(qx, qy, qz, indexing="ij")
    q = np.vstack([qxx.ravel(), qyy.ravel(), qzz.ravel()]).T.copy()
    r = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    U = np.array([1, 2, 3])
    f = np.array([1, 2, 5])
    amps = core.phase_factor_mesh(r, f, q_min=q_min, q_max=q_max, N=shape, R=R, U=U)
    amps = amps.reshape(shape)
    interp1 = core.mesh_interpolation(
        a_map=amps, q=q, N=shape, q_min=q_min, q_max=q_max, dq=dq
    )
    # Check that GPU code agrees with CPU code
    interp2 = interpolate.trilinear_interpolation(
        amps.astype(np.complex128), q, corners=None, deltas=dq, x_min=q_min, x_max=q_max
    )
    assert np.max(np.abs(interp1 - interp2)) == 0


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

# def _test_ridiculous_sum(double_precision=False):
#
#     core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=double_precision)
#
#     # @jit(nopython=True)
#     # def numba_add(input, dtype):
#     #     out = np.array((0,), dtype=dtype)
#     #     for i in range(len(input)):
#     #         out[0] += input[i]
#     #     return out[0]
#
#     np.random.seed(0)
#     for n in np.array([2**n for n in range(0, 16)]):
#         a = np.random.rand(n).astype(core.real_t)
#         b = 0
#         for i in range(0, n):
#             b += a[i]
#         # d = numba_add(a, core.real_t)
#         c = core.test_simple_sum(a)
#         if double_precision:
#             assert(np.abs(b - c) < 1e-12)
#         else:
#             assert(np.abs(b - c)/np.abs(c) < 1e-5)
#             # assert(np.abs(b - d)/np.abs(d) < 1e-5)

# def test_atomics_01():
#     core = clcore.ClCore(group_size=32)
#     n = 3
#     a = np.zeros(n)
#     b = np.arange(n)
#     a_gpu = core.to_device(a, dtype=core.real_t)
#     b_gpu = core.to_device(b, dtype=core.real_t)
#     core.test_atomic_add_real(a_gpu, b_gpu)
#     assert a_gpu.get()[0] - np.sum(b) * n == 0
#
# def test_atomics_02():
#     core = clcore.ClCore(group_size=32)
#     n = 101
#     a = np.zeros(n)
#     b = np.arange(n)
#     a_gpu = core.to_device(a, dtype=core.int_t)
#     b_gpu = core.to_device(b, dtype=core.int_t)
#     core.test_atomic_add_int(a_gpu, b_gpu)
#     assert a_gpu.get()[0] - np.sum(b) * n == 0
#
# def test_atomics_03():
#     core = clcore.ClCore(group_size=32)
#     n = 100
#     a = np.zeros(n)
#     b = np.arange(n)
#     a_gpu = core.to_device(a, dtype=core.real_t)
#     b_gpu = core.to_device(b, dtype=core.real_t)
#     core.test_atomic_add_real(a_gpu, b_gpu)
#     assert a_gpu.get()[0] - np.sum(b) * n == 0
#
# def test_atomics_04():
#     core = clcore.ClCore(group_size=32)
#     n = 100
#     a = np.zeros(n)
#     b = np.arange(n)
#     a_gpu = core.to_device(a, dtype=core.real_t)
#     b_gpu = core.to_device(b, dtype=core.real_t)
#     m = 5
#     for _ in range(m):
#         core.test_atomic_add_real(a_gpu, b_gpu)
#     assert a_gpu.get()[0] - np.sum(b) * n * m == 0
