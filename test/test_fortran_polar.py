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

import numpy as np
from reborn.misc import polar

n_q_bins = 2
n_phi_bins = 3
polar_shape = (n_q_bins, n_phi_bins)
data_points = n_q_bins * n_phi_bins * 2

q_range = [0, 1]
qs = np.linspace(q_range[0], q_range[1], data_points)
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

# TEST CASE DATA:
#
# Data: [1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0.]
#
#     Q --->                                              Phi
# 0                      0.5                      1        |
# 1-----------------------|-----------------------|  0     |
# |   0         AVG=0.5   |             AVG=0     |        \/
# |       1               |                       |
# |           0           |                       |
# |-----------------------|-----------------------|
# |                1   0  |             AVG=0.5   |
# |                       |  1                    |  pi
# |  AVG=0.5              |      0                |
# |-----------------------|-----------------------|
# |                       |          1   0        |
# |                       |                   1   |
# |   AVG=0               |  AVG=0.5              |
# |-----------------------|-----------------------0  2 pi


args = [polar_shape[0], q_bin_size, q_min,
        polar_shape[1], phi_bin_size, phi_min,
        qs, phis, mask,  # weights
        data, mask]


def test_polar_indices_python_fortran_match():
    q_index_p, p_index_p = polar.get_polar_bin_indices_python(*args[:-3])
    q_index_f, p_index_f = polar.get_polar_bin_indices_fortran(*args[:-3])
    assert q_index_p.all() == q_index_f.all()
    assert p_index_p.all() == p_index_f.all()


def test_polar_simple():
    print('===================================================HELLO')
    print(qs)
    print(phis*3/(2*np.pi))
    print(data)
    pmean, pmask = polar.get_polar_bin_mean_fortran(*args)
    print(pmean)
    assert pmask[0, 1] == 0
    assert pmask[1, 1] == 0.5


def test_polar_bin_mean_fortran_python_match():
    pmean, pmask = polar.get_polar_bin_mean_python(*args)
    pmean_f, pmask_f = polar.get_polar_bin_mean_fortran(*args)
    assert pmean.all() == pmean_f.all()
    assert pmask.all() == pmask_f.all()


def test_polar_stats():
    print()
    nq = 3
    nphi = 4
    data = np.arange(nq*nphi, dtype=np.float64).reshape(nq, nphi)
    data[0, 1]
    weights = np.ones_like(data)
    weights[0, 1] = 0
    q = np.arange(nq, dtype=np.float64) + 1
    p = np.arange(nphi, dtype=np.float64) + 1
    q, p = np.meshgrid(q, p, indexing='ij')
    q = q.ravel()
    p = p.ravel()
    stats = polar.get_polar_stats(data, q, p, weights=weights, n_q_bins=nq, q_min=1, q_max=nq, n_p_bins=nphi,
                                  p_min=1, p_max=nphi, sum_=None, sum2=None, w_sum=None)
    # print(stats['mean'])
    assert stats['mean'][0, 0] == 0
    assert stats['mean'][0, 1] == 0
    assert stats['mean'][0, 2] == 2
