import numpy as np
from reborn.fortran import polar_f


def test_polar_simple():
    n_q_bins = 2
    n_phi_bins = 3
    polar_shape = (n_q_bins, n_phi_bins)
    data_points = n_q_bins * n_phi_bins * 2

    q_range = [0, 1]
    qs = np.linspace(0, 1, data_points)
    q_bin_size = (q_range[1] - q_range[0]) / float(n_q_bins - 1)
    q_min = q_range[0] - q_bin_size / 2
    phi_bin_size = 2 * np.pi / n_phi_bins
    phis = np.linspace(0, 2 * np.pi, data_points)
    phi_range = [phi_bin_size / 2, 2 * np.pi - phi_bin_size / 2]
    phi_min = phi_range[0] - phi_bin_size / 2

    index = np.arange(data_points)
    data = np.zeros(data_points)
    mask = np.ones(data_points)
    data[index % 2 == 0] = 1

    psum, count = polar_f.polar_binning(polar_shape[0],
                                        q_bin_size, q_min,
                                        polar_shape[1],
                                        phi_bin_size, phi_min,
                                        qs, phis,
                                        data, mask)
    cnt = count.reshape(polar_shape).astype(int)
    sum_ = psum.reshape(polar_shape).astype(float)
    mean_ = np.divide(sum_, cnt, out=np.zeros_like(sum_), where=cnt != 0)

    assert cnt[0, 1] == 2
    assert cnt[1, 1] == 2
