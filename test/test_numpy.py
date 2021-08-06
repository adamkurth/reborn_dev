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

from __future__ import (absolute_import, division, print_function, unicode_literals)

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

    c = np.arange(200).reshape(10, 20).copy()
    f = np.asfortranarray(c)
    c_original = c               # We need to keep track of the original arrays
    f_original = f
    c_flat = c.reshape(200)
    f_flat = f.reshape(200)

    # print(type(c.data))
    # print(type(c_flat.data))
    assert c.data != c_flat.data
    assert f.data != f_flat.data
    assert c.data == f.data      # ndarray.data is: "Python buffer object pointing to the start of the array's data."
    assert c_flat.data == f_flat.data
    assert c.shape[0] == 10      # They are still the same shape
    assert f.shape[0] == 10
    assert c.flags.c_contiguous  # Yet one is "c-contiguous" while the other is "f-contiguous"
    assert f.flags.f_contiguous  # This seems impossible to me.  How can c and f have the same shape and same data
                                 # without both of them being either c *or* f contiguous?
    assert c_flat[21] == 21      # Ravel makes the ND array into a 1D array, may or may not copy memory buffer.
    assert f_flat[21] == 21
    assert c_flat.data == f_flat.data
    assert c.data != c_flat.data
    assert f.data != f_flat.data
    assert c[0, 1] == 1
    c[0, 1] = 0                  # Now we modify the original array c
    assert c.flags.c_contiguous  # Still f/c contiguous as before
    assert f.flags.f_contiguous
    assert c.data != f.data      # But they no longer share the same memory.  The operation on c created a new array.
    assert c.data == c_original.data
    assert f.data == f_original.data


def test_phase_factor_sum_ordering():

    a = np.arange(10000)
    b = np.sum(a)*np.exp(1j*3)
    c = np.sum(a*np.exp(1j*3))
    assert np.sum(np.abs(c - b)) != 0
