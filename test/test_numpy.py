from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
sys.path.append('..')
import numpy as np


def test_rotations_and_broadcasting():

    rot = np.array([[0, 1., 0],
                  [-1, 0, 0],
                  [0, 0, 1.]])
    vec = np.array([1, 2, 3])
    vec2 = np.random.rand(5, 3)
    vec2[0, :] = vec

    # We are checking that the following two method of rotating vectors are the same:
    vec_rotated = np.dot(vec2, rot.T)
    vec_rotated2 = np.dot(rot, vec2.T).T

    # Also note that the result is as expected:
    assert(np.allclose(vec_rotated[0, :], np.array([2, -1, 3])))
    assert(np.allclose(vec_rotated, vec_rotated2))
    assert(np.allclose(vec_rotated[0, :], np.array([2, -1, 3])))

    # Check that addition broadcasting works even if arrays don't have shape M x D and 1 x D
    vec1 = np.array([1, 2, 3])
    vec2 = np.zeros((5, 3))
    vec3 = vec2 + vec1
    assert(np.sum(vec3[:, 1]) == 10)

    vec1 = np.array([[1, 2, 3]])
    vec2 = np.zeros((5, 3))
    vec3 = vec2 + vec1
    assert(np.sum(vec3[:, 1]) == 10)


def test_fortranarray():

    a = np.zeros((10, 10))
    b = a
    c = np.asfortranarray(a)
    assert(a.data == b.data)
    assert(a.data == c.data)
    assert(b.data == c.data)
    # a[0, 0] = 1
    # assert(a.data == b.data)
    # assert(a.data == c.data)  # Fails. Why?
    # assert(b.data == c.data)
    # c[0, 0] = 1
    # assert(a.data == b.data)
    # assert(a.data == c.data)  # Fails. Why?
    # assert(b.data == c.data)