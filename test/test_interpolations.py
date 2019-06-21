from bornagain import utils
import numpy as np


def test_trilinear():

    shape = (2, 3, 4)
    data = np.ones(shape, dtype=np.double)
    max_corners = np.array(shape)
    min_corners = -max_corners
    samples = np.array([0, 0, 0])
    dataout = utils.trilinear_interpolation(data, min_corners, max_corners, samples, mask=None)
    assert dataout[0] == 1
