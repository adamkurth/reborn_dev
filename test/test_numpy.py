from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
sys.path.append('..')
from bornagain import utils
import numpy as np


def test_rotations_and_broadcasting():

    R = np.array([[0, 1., 0],
                  [-1, 0, 0],
                  [0, 0, 1.]])
    vec = np.array([1, 2, 3])
    vec2 = np.random.rand(5, 3)
    vec2[0, :] = vec

    # We are checking that the following two method of rotating vectors are the same:
    vec_rotated = np.dot(vec2, R.T)
    vec_rotated2 = np.dot(R, vec2.T).T

    # Also note that the result is as expected:
    assert(np.allclose(vec_rotated[0, :], np.array([2, -1, 3])))
    assert(np.allclose(vec_rotated, vec_rotated2))
    assert(np.allclose(vec_rotated[0, :], np.array([2, -1, 3])))

    # Check that addition broadcasting works even if arrays don't have shape M x D and 1 x D
    vec1 = np.array([1, 2, 3])
    vec2 = np.zeros((5, 3))
    vec3 = vec2 + vec1
    assert(np.sum(vec3[:, 1]) == 10)

    vec1 = utils.vec_check3(np.array([1, 2, 3]))
    vec2 = np.zeros((5, 3))
    vec3 = vec2 + vec1
    assert(np.sum(vec3[:, 1]) == 10)

if __name__ == "__main__":

    from time import time

    n = 100000
    m_iter = 100
    rot = np.random.rand(3, 3)
    vecs1 = np.random.rand(n, 3)
    vecs2 = np.random.rand(3, n)

    t = time()
    for m in range(0, m_iter):
        junk = np.dot(rot, vecs1.T).T
    t1 = time() - t
    print('Transpose vecs: %.3f' % t1)

    t = time()
    for m in range(0, m_iter):
        junk = np.dot(vecs1, rot.T)
    t1 = time() - t
    print('Transpose R   : %.3f' % t1)

    t = time()
    for m in range(0, m_iter):
        junk = np.dot(vecs1, rot.T.copy())
    t1 = time() - t
    print('Transpose R copy: %.3f' % t1)

    t = time()
    for m in range(0, m_iter):
        junk = utils.rotate(rot, vecs1)
    t3 = time() - t
    print('Utis rotate   : %.3f' % t3)

    t = time()
    for m in range(0, m_iter):
        junk = np.dot(rot, vecs2)
    t2 = time() - t
    print('No transpose  : %.3f' % t2)

    R = np.array([[0, 1., 0],
                  [-1, 0, 0],
                  [0, 0, 1.]])
    vec = np.array([1, 2, 3])
    vec_rotated = utils.rotate(R, vec)
    print(vec_rotated, 'should be [2, -1, 3]')
