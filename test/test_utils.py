from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys

import numpy as np

sys.path.append("..")
from bornagain import utils


def test_rotation():

    R = np.array([[0, 1., 0],
                  [-1, 0, 0],
                  [0, 0, 1.]])
    vec = np.array([3, np.pi, np.pi*4])
    vec_rotated = utils.rotate(R, vec)
    vec_expected = np.array([np.pi, -3, np.pi*4])

    assert(np.max(np.abs(vec_rotated - vec_expected)) < 1e-10)

    vec10 = np.random.rand(10, 3)
    vec_rotated = utils.rotate(R, vec10)
    vec_expected = vec10.copy()
    vec_expected[:, 0] = vec10[:, 1]
    vec_expected[:, 1] = -vec10[:, 0]

    assert (np.max(np.abs(vec_rotated - vec_expected)) < 1e-10)


def test_random_rotation_matrix(main=False):
    
    R = utils.random_rotation()
    d = np.linalg.det(R)
    
    if main:
        print("A random rotation matrix R:")
        print(R)
        print("Determinant of R:")
        print(d)
    assert(np.abs(d - 1) < 1e-15)
    
    
if __name__ == "__main__":
    
    main = True
    test_random_rotation(main)
