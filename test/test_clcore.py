"""
Test the clcore simulation engine in bornagain.simulate.  This requires pytest.  You can also run from main
like this:
> python test_simulate_clcore.py
If you want to view results just add the keyword "view"
> python test_simulate_clcore.py view
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from bornagain import utils

try:
    from bornagain.simulate import clcore
    import pyopencl
    from pyopencl import array as clarray
    cl_array = clarray.Array
    havecl = True
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


def test_atomics_01():

    if not havecl:
        return

    core = clcore.ClCore(group_size=32)

    n = 3
    a = np.zeros(n)
    b = np.arange(n)
    a_gpu = core.to_device(a, dtype=core.real_t)
    b_gpu = core.to_device(b, dtype=core.real_t)
    core.test_atomic_add_real(a_gpu, b_gpu)
    assert a_gpu.get()[0] - np.sum(b) * n == 0


def test_atomics_02():

    if not havecl:
        return

    core = clcore.ClCore(group_size=32)

    n = 101
    a = np.zeros(n)
    b = np.arange(n)
    a_gpu = core.to_device(a, dtype=core.int_t)
    b_gpu = core.to_device(b, dtype=core.int_t)
    core.test_atomic_add_int(a_gpu, b_gpu)
    assert a_gpu.get()[0] - np.sum(b) * n == 0


def test_atomics_03():

    if not havecl:
        return

    core = clcore.ClCore(group_size=32)

    n = 100
    a = np.zeros(n)
    b = np.arange(n)
    a_gpu = core.to_device(a, dtype=core.real_t)
    b_gpu = core.to_device(b, dtype=core.real_t)
    core.test_atomic_add_real(a_gpu, b_gpu)
    assert a_gpu.get()[0] - np.sum(b) * n == 0


def test_atomics_04():

    if not havecl:
        return

    core = clcore.ClCore(group_size=32)

    n = 100
    a = np.zeros(n)
    b = np.arange(n)
    a_gpu = core.to_device(a, dtype=core.real_t)
    b_gpu = core.to_device(b, dtype=core.real_t)
    m = 5
    for _ in range(m):
        core.test_atomic_add_real(a_gpu, b_gpu)
    assert a_gpu.get()[0] - np.sum(b) * n * m == 0


def test_rotations(double_precision=False):

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