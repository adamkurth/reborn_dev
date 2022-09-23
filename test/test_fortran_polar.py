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
from reborn.misc import polar

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


def test_polar_simple():
    mean_, count = polar_f.polar_binning.polar_mean(polar_shape[0],
                                                    q_bin_size,
                                                    q_min,
                                                    polar_shape[1],
                                                    phi_bin_size,
                                                    phi_min,
                                                    qs,
                                                    phis,
                                                    mask,  # weights
                                                    data,
                                                    mask)
    cnt = count.reshape(polar_shape).astype(float)

    assert cnt[0, 1] == 1
    assert cnt[1, 1] == 1


def test_polar_fortran_python_match():
    pmean, pmask = polar.polar_mean(polar_shape[0],
                                    q_bin_size,
                                    q_min,
                                    polar_shape[1],
                                    phi_bin_size,
                                    phi_min,
                                    qs,
                                    phis,
                                    mask,
                                    data,
                                    mask)
    pmean_f, pmask_f = polar_f.polar_binning.polar_mean(polar_shape[0],
                                                        q_bin_size,
                                                        q_min,
                                                        polar_shape[1],
                                                        phi_bin_size,
                                                        phi_min,
                                                        qs,
                                                        phis,
                                                        mask,  # weights
                                                        data,
                                                        mask)

    for x, y in zip(pmean, pmean_f):
        assert x == y
    for x, y in zip(pmask, pmask_f):
        assert x == y
