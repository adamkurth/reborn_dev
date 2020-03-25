from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from reborn import utils


def test_rotation():

    R = np.array([[0, 1., 0],
                  [-1, 0, 0],
                  [0, 0, 1.]])
    vec = np.array([3, np.pi, np.pi*4])
    vec_rotated = np.dot(vec, R.T)
    vec_expected = np.array([np.pi, -3, np.pi*4])

    assert(np.max(np.abs(vec_rotated - vec_expected)) < 1e-10)

    vec10 = np.random.rand(10, 3)
    vec_rotated = np.dot(vec10, R.T)
    vec_expected = vec10.copy()
    vec_expected[:, 0] = vec10[:, 1]
    vec_expected[:, 1] = -vec10[:, 0]

    assert (np.max(np.abs(vec_rotated - vec_expected)) < 1e-10)
