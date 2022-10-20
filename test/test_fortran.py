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
from types import ModuleType
from reborn import fortran


def test_01():
    assert(isinstance(fortran.utils_f, ModuleType))


def test_02():
    pattern = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    weights = np.array([1, 1, 2, 0.5, 1, 2], dtype=float)
    q = np.array([-1, 0, 0.5, 1.5, 2.5, 2.6], dtype=float)
    n_bins = 3
    sum_ = np.zeros(n_bins)
    sum2 = np.zeros(n_bins)
    w_sum = np.zeros(n_bins)
    q_min = 0.5  # Bins are |___0.5___|___1.5___|___2.5___|
    q_max = 2.5
    fortran.scatter_f.profile_stats(pattern, q, weights, n_bins, q_min, q_max, sum_, sum2, w_sum)
    assert(np.max(np.abs(sum_ - np.array([8, 2, 17]))) == 0)
    indices = np.empty(6, dtype=np.int32)
    fortran.scatter_f.profile_indices(q, n_bins, q_min, q_max, indices)
    sum_1 = sum_.copy()
    sum_ = np.zeros(n_bins)
    sum2 = np.zeros(n_bins)
    w_sum = np.zeros(n_bins)
    fortran.scatter_f.profile_stats_indexed(pattern, indices, weights, sum_, sum2, w_sum)
    assert(np.max(np.abs(sum_ - sum_1)) == 0)


def test_03():
    shape = (5, 5)
    dat = np.ones(shape, dtype=float)
    conv = np.empty_like(dat)
    n = 1
    fortran.peaks_f.boxconv(dat.T, conv.T, n, shape[0], shape[1])
    ans = np.array([[4., 6., 6., 6., 4.],
                    [6., 9., 9., 9., 6.],
                    [6., 9., 9., 9., 6.],
                    [6., 9., 9., 9., 6.],
                    [4., 6., 6., 6., 4.]])
    assert(np.sum(np.abs(ans-conv)) == 0)
