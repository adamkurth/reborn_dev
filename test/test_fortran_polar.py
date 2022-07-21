import numpy as np
# try:
from reborn.fortran import polar_f
# except ImportError:
#     polar_f = None


def test_polar_simple():
    n_q_bins = 2
    n_phi_bins = 3
    polar_shape = (n_q_bins, n_phi_bins)
    qk = np.array([0, 1], dtype=int)
    pk = np.array([0, 2], dtype=int)
    dk = np.array([1, 2], dtype=np.float64)
    dsum, count = polar_f.polar_binning(n_q_bins, n_phi_bins, qk, pk, dk)
    cnt = count.reshape(polar_shape).astype(int).T
    sum_ = dsum.reshape(polar_shape).T
    assert cnt[0, 0] == 1
    assert cnt[2, 1] == 1


def test_fortran_python_polar_match():
    pass
