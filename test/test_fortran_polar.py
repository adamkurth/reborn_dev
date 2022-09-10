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
