import numpy as np
from reborn import utils
from reborn.utils import rotate3D


def test_max_pair_distance():

    vecs = np.arange(3*4, dtype=np.float64).reshape([4, 3])
    assert utils.max_pair_distance(vecs) == np.sqrt(np.sum((vecs[0, :] - vecs[3, :])**2))
    vecs = np.arange(3*4, dtype=np.int).reshape([4, 3])
    assert utils.max_pair_distance(vecs) == np.sqrt(np.sum((vecs[0, :] - vecs[3, :])**2))


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


def test_binned_statistic():

    x = np.arange(30)/3.0
    y = np.arange(30)
    z = utils.binned_statistic(x, y, np.median, 8, (1, 9))
    a = np.array([4,  7, 10, 13, 16, 19, 22, 25])
    assert(np.max(np.abs(z - a)) == 0)

    x = np.arange(6)
    y = np.arange(6)
    z = utils.binned_statistic(x, y, np.median, 6, (0, 5))
    a = np.array([0, 1, 2, 3, 4, 0])
    assert(np.max(np.abs(z - a)) == 0)

    x = np.array([0, 1, 4, 5])
    y = np.array([0, 1, 4, 5])
    z = utils.binned_statistic(x, y, np.median, 6, (0, 6))
    a = np.array([0, 1, 0, 0, 4, 5])
    assert(np.max(np.abs(z - a)) == 0)
