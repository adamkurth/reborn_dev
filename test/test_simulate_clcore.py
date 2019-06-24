"""
Test the clcore simulation engine in bornagain.simulate.  This requires pytest.  You can also run from main
like this: 
> python test_simulate_clcore.py
If you want to view results just add the keyword "view" 
> python test_simulate_clcore.py view
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
sys.path.append('..')
import pytest
import numpy as np
import bornagain as ba
from bornagain import utils

try:
    from bornagain.simulate import clcore
    import pyopencl
    from pyopencl import array as clarray
    cl_array = clarray.Array
    havecl = True
    # Check for double precision:
    test_core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=True)
    if test_core.double_precision:
        have_double = True
    else:
        have_double = False
    ctx = clcore.create_some_gpu_context()
except ImportError:
    clcore = None
    clarray = None
    pyopencl = None
    havecl = False

import bornagain.simulate.numbacore as numbacore
from bornagain.utils import rotation_about_axis

view = False

if len(sys.argv) > 1:
    view = True


@pytest.mark.cl
def test_clcore_float():

    if havecl:
        _clcore(double_precision=False)
        _test_rotations(double_precision=False)
        _test_ridiculous_sum(double_precision=False)


@pytest.mark.cl
def test_clcore_double():

    if havecl and have_double:
        _clcore(double_precision=True)
        _test_rotations(double_precision=True)
        _test_ridiculous_sum(double_precision=True)


def _clcore(double_precision=False):

    ###########################################################################
    # Setup the simulation core
    ###########################################################################

    core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=have_double)
    # core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=double_precision)

    if double_precision is False:
        numbacore.real_t = np.float32
        numbacore.complex_t = np.complex64

    assert(core.get_group_size() == core.group_size)
   
    # print("Using group size: %d" %core.group_size)
    ###########################################################################
    # Check that there are no errors in phase_factor_qrf_inplace
    # TODO: check that the amplitudes are correct
    ###########################################################################

    pad = ba.detector.PADGeometry()
    pad.simple_setup(n_pixels=4, pixel_size=1, distance=1)
    n_atoms = 10
    rot = np.eye(3, dtype=core.real_t)
    q = pad.q_vecs(beam_vec=[0, 0, 1], wavelength=1)
    
    r_vecs = np.random.random([n_atoms, 3])
    f = np.random.random([n_atoms])*1j

    amps = core.phase_factor_qrf(q, r_vecs, f, R=rot)
    assert(type(amps) is np.ndarray)

    ampsn = numbacore.phase_factor_qrf(q, r_vecs, f, R=rot)

    assert(np.max(np.abs(amps - ampsn)) < 1e-3)  # Why such a big difference between CPU and GPU?
    
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
    
    assert(np.allclose(10*amps1, amps10))

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

    amps = core.phase_factor_pad(r_vecs, f, t_vec, f_vec, s_vec, b_vec, n_f, n_s, w, R=rot, a=None)
    assert(type(amps) is np.ndarray)

    r_vecs = core.to_device(r_vecs)
    f = core.to_device(f)
    a = core.to_device(shape=(n_f * n_s,), dtype=core.complex_t)

    amps = core.phase_factor_pad(r_vecs, f, t_vec, f_vec, s_vec, b_vec, n_f, n_s, w, R=rot, a=a)

    del n_atoms, r_vecs, f, t_vec, f_vec, s_vec, b_vec, n_f, n_s, w, rot, a, amps

    ###########################################################################
    # Check for errors in phase_factor_mesh
    # TODO: check that amplitudes are correct
    ###########################################################################

    n_atoms = 10
    n_mesh = np.array([2, 3, 4])
    q_min = np.array([-1] * 3)
    q_max = np.array([+1] * 3)
    r_vecs = np.random.random([n_atoms, 3])
    f = np.random.random([n_atoms]) * 1j

    amps = core.phase_factor_mesh(r_vecs, f, n_mesh, q_min, q_max)
    assert(type(amps) is np.ndarray)

    r_vecs = core.to_device(r_vecs)
    f = core.to_device(f)
    a = core.to_device(shape=(np.prod(n_mesh),), dtype=core.complex_t)

    amps = core.phase_factor_mesh(r_vecs, f, N=n_mesh, q_min=q_min, q_max=q_max, a=a)
    assert(type(amps) is cl_array)

    del n_atoms, n_mesh, q_min, q_max, r_vecs, f, amps, a

    ###########################################################################
    # Check for errors in mesh_interpolation
    # TODO: check that amplitudes are correct
    ###########################################################################

    n_atoms = 10
    n_mesh = np.array([2, 3, 4])
    q_min = np.array([-1] * 3)
    q_max = np.array([+1] * 3)
    r_vecs = np.random.random([n_atoms, 3])
    f = np.random.random([n_atoms]) * 1j
    
    pad = ba.detector.PADGeometry()
    pad.simple_setup(n_pixels=4, pixel_size=1, distance=1)
    rot = np.eye(3, dtype=core.real_t)
    q = pad.q_vecs(beam_vec=[0, 0, 1], wavelength=1)

    amps = core.phase_factor_mesh(r_vecs, f, N=n_mesh, q_min=q_min, q_max=q_max)
    amps2 = core.mesh_interpolation(amps, q, N=n_mesh, q_min=q_min, q_max=q_max)
    assert(type(amps) is np.ndarray)
    assert(type(amps2) is np.ndarray)

    r_vecs = core.to_device(r_vecs)
    f = core.to_device(f)
    q = core.to_device(q)
    a = core.to_device(shape=(np.prod(n_mesh),), dtype=core.complex_t)
    a_out = core.to_device(shape=pad.n_pixels, dtype=core.complex_t)

    amps = core.phase_factor_mesh(r_vecs, f, N=n_mesh, q_min=q_min, q_max=q_max, a=a)
    amps2 = core.mesh_interpolation(a, q, N=n_mesh, q_min=q_min, q_max=q_max, R=rot, a=a_out)
    assert(type(amps) is cl_array)
    assert(type(amps2) is cl_array)

    del n_atoms, n_mesh, q_min, q_max, r_vecs, f, amps, amps2, a, q, a_out

    ###########################################################################
    # Check phase_factor_qrf_chunk_r
    ###########################################################################

    pad = ba.detector.PADGeometry()
    pad.simple_setup(n_pixels=4, pixel_size=1, distance=1)
    n_atoms = 10
    rot = np.eye(3, dtype=core.real_t)
    q = pad.q_vecs(beam_vec=[0, 0, 1], wavelength=1)
    r_vecs = np.random.random([n_atoms, 3])
    f = np.random.random([n_atoms]) * 1j
    q_dev = core.to_device(q, dtype=core.real_t)
    # r_dev = core.to_device(r, dtype=core.real_t)
    # f_dev = core.to_device(f, dtype=core.complex_t)

    q1 = q_dev.get()
    q2 = q_dev.get()
    assert(np.allclose(q1, q2))

    amps0 = core.phase_factor_qrf(q, r_vecs, f, rot)

    a = core.phase_factor_qrf_chunk_r(q, r_vecs, f, rot, n_chunks=3)
    assert(type(a) is np.ndarray)

    a_dev = core.to_device(shape=(q.shape[0],), dtype=core.complex_t)

    core.phase_factor_qrf_chunk_r(q, r_vecs, f, rot, a=a_dev, add=False, n_chunks=2)
    amps1 = a_dev.get()

    assert(np.allclose(amps0, amps1))

    for _ in range(9):
        core.phase_factor_qrf_chunk_r(q, r_vecs, f, rot, a=a_dev, add=True, n_chunks=2)

    amps10 = a_dev.get()

    assert(np.allclose(10 * amps1, amps10))

    core.phase_factor_qrf_chunk_r(q, r_vecs, f, rot, a=a_dev, add=False, n_chunks=3)
    amps1_3 = a_dev.get()

    assert(np.allclose(amps1, amps1_3))

    for _ in range(9):
        core.phase_factor_qrf_chunk_r(q, r_vecs, f, rot, a=a_dev, add=True, n_chunks=3)

    amps10_3 = a_dev.get()

    assert(np.allclose(10 * amps1_3, amps10_3))

    del q, r_vecs, f, rot, n_atoms, q_dev, a_dev, amps0, amps1, amps10, amps1_3, amps10_3, pad

    ###########################################################################
    # Check that rotations and translations work
    ###########################################################################

    n_pixels = 20
    pixel_size = 300e-6
    distance = 0.5
    wavelength = 2e-10
    pad = ba.detector.PADGeometry(n_pixels=n_pixels, pixel_size=pixel_size, distance=distance)
    beam = ba.source.Beam(wavelength=wavelength)
    q = pad.q_vecs(beam=beam)

    rot = rotation_about_axis(0.1, [1, 1, 0]).astype(core.real_t)
    trans = (np.array([1, 0, 1])*1e-9).astype(core.real_t)

    np.random.seed(0)
    n_atoms = 10
    r0 = np.random.random([n_atoms, 3]).astype(core.real_t)*5e-9
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
        assert (np.mean(np.abs(amps1 - amps2)) / np.mean(np.abs(amps1)) < tol)
        assert (np.mean(np.abs(amps1 - amps3)) / np.mean(np.abs(amps3)) < tol)
        assert (np.mean(np.abs(amps3 - amps2)) / np.mean(np.abs(amps2)) < tol)
        assert (np.mean(np.abs(amps3 - amps4)) / np.mean(np.abs(amps4)) < tol)
    else:
        tol = 1e-6
        assert (np.mean(np.abs(amps1 - amps2)) / np.mean(np.abs(amps1)) < tol)
        assert (np.mean(np.abs(amps1 - amps3)) / np.mean(np.abs(amps3)) < tol)
        assert (np.mean(np.abs(amps3 - amps2)) / np.mean(np.abs(amps2)) < tol)
        assert (np.mean(np.abs(amps3 - amps4)) / np.mean(np.abs(amps4)) < tol)

    # Do rotation and translation on CPU
    amps1 = core.phase_factor_pad(np.dot(r0, rot.T) + trans, f, R=None, U=None, beam=beam, pad=pad)
    # Rotation and translation on GPU
    amps2 = core.phase_factor_pad(r0, f, R=rot, U=trans, beam=beam, pad=pad)
    # Rotation on CPU, translation on GPU
    amps3 = core.phase_factor_pad(np.dot(r0, rot.T), f, R=None, U=trans, beam=beam, pad=pad)
    # Rotation on GPU, translation on CPU
    amps4 = core.phase_factor_pad(r0 + np.dot(trans, rot), f, R=rot, U=None, beam=beam, pad=pad)

    if double_precision:
        tol = 1e-6
        assert (np.mean(np.abs(amps1 - amps2)) / np.mean(np.abs(amps1)) < tol)
        assert (np.mean(np.abs(amps1 - amps3)) / np.mean(np.abs(amps3)) < tol)
        assert (np.mean(np.abs(amps3 - amps2)) / np.mean(np.abs(amps2)) < tol)
        assert (np.mean(np.abs(amps3 - amps4)) / np.mean(np.abs(amps4)) < tol)
    else:
        tol = 1e-6
        assert (np.mean(np.abs(amps1 - amps2)) / np.mean(np.abs(amps1)) < tol)
        assert (np.mean(np.abs(amps1 - amps3)) / np.mean(np.abs(amps3)) < tol)
        assert (np.mean(np.abs(amps3 - amps2)) / np.mean(np.abs(amps2)) < tol)
        assert (np.mean(np.abs(amps3 - amps4)) / np.mean(np.abs(amps4)) < tol)


@pytest.mark.cl
def _test_rotations(double_precision=False):

    if not havecl:
        return

    core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=double_precision)

    theta = 25*np.pi/180.
    sin = np.sin(theta)
    cos = np.cos(theta)

    rot = np.array([[cos, sin, 0],
                    [-sin, cos, 0],
                    [0, 0, 1]], dtype=core.real_t)
    trans = np.array([1, 2, 3], dtype=core.real_t)
    vec1 = np.array([1, 2, 0], dtype=core.real_t)

    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec3 = np.dot(vec1, rot.T) + trans

    # Rotation on gpu and rotation with utils.rotate should do the same thing
    assert np.max(np.abs(vec2-vec3)) <= 1e-6

    vec1 = np.array([1, 2, 0], dtype=core.real_t)
    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec4 = np.random.rand(10, 3).astype(core.real_t)
    vec4[0, :] = vec1
    vec3 = np.dot(vec4, rot.T) + trans
    vec3 = vec3[0, :]

    # Rotation on gpu and rotation with utils.rotate should do the same thing (even for many vectors; shape Nx3)
    assert(np.max(np.abs(vec2-vec3)) <= 1e-6)

    rot = np.array([[0, 1.0, 0],
                    [-1.0, 0, 0],
                    [0, 0, 1.0]])
    trans = np.zeros((3,), dtype=core.real_t)
    vec1 = np.array([1.0, 0, 0], dtype=core.real_t)
    vec2 = core.test_rotate_vec(rot, trans, vec1)
    vec3 = np.dot(vec1, rot.T) + trans
    vec_pred = np.array([0, -1.0, 0])

    # Check that results are as expected
    assert np.max(np.abs(vec2 - vec3)) < 1e-6
    assert np.max(np.abs(vec2 - vec_pred)) < 1e-6


def _test_ridiculous_sum(double_precision=False):

    if not havecl:
        return

    core = clcore.ClCore(context=None, queue=None, group_size=1, double_precision=double_precision)

    # @jit(nopython=True)
    # def numba_add(input, dtype):
    #     out = np.array((0,), dtype=dtype)
    #     for i in range(len(input)):
    #         out[0] += input[i]
    #     return out[0]

    np.random.seed(0)
    for n in np.array([2**n for n in range(0, 16)]):
        a = np.random.rand(n).astype(core.real_t)
        b = 0
        for i in range(0, n):
            b += a[i]
        # d = numba_add(a, core.real_t)
        c = core.test_simple_sum(a)
        if double_precision:
            assert(np.abs(b - c) < 1e-12)
        else:
            assert(np.abs(b - c)/np.abs(c) < 1e-5)
            # assert(np.abs(b - d)/np.abs(d) < 1e-5)
