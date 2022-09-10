import numpy as np
from .. import fortran


def max_pair_distance(vecs):
    r"""
    Determine the maximum distance between to vectors in a list of vectors.

    Arguments:
        vecs (Nx3 |ndarray|) : Input vectors.

    Returns:
        float : The maximum pair distance.
    """
    vecs = np.double(vecs)
    if not vecs.flags.c_contiguous:
        vecs = vecs.copy()
    d_max = np.array([0], dtype=np.float64)
    fortran.utils_f.max_pair_distance(vecs.T, d_max)
    return d_max[0]
