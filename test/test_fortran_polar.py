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
phis = np.linspace(0, 2 * np.pi, data_points) * 0.99999
phi_range = [phi_bin_size / 2, 2 * np.pi - phi_bin_size / 2]
phi_min = phi_range[0] - phi_bin_size / 2

index = np.arange(data_points)
data = np.zeros(data_points)
mask = np.ones(data_points)
data[index % 2 == 0] = 1

# TEST CASE DATA: Beware that the first and last datapoints
# are borderline cases because they lie just at the boundaries.
# In particular, the value at 2*pi is affected by the modulo
# operation and might be unpredictable as it depends on the
# precision of pi.  Therefore, it is shifted slightly downward
# to make clear which bin this should fall in.
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
# |   AVG=0               |  AVG=0.5             0|
# |-----------------------|-----------------------|  2 pi

q_index_p, p_index_p = polar.bin_indices(
    n_q_bins=polar_shape[0],
    q_bin_size=q_bin_size,
    q_min=q_min,
    n_p_bins=polar_shape[1],
    p_bin_size=phi_bin_size,
    p_min=phi_min,
    qs=qs,
    ps=phis,
    py=True,
)
q_index_f, p_index_f = polar.bin_indices(
    n_q_bins=polar_shape[0],
    q_bin_size=q_bin_size,
    q_min=q_min,
    n_p_bins=polar_shape[1],
    p_bin_size=phi_bin_size,
    p_min=phi_min,
    qs=qs,
    ps=phis,
    py=False,
)


def test_bin_indices_python_fortran_match():
    for qp, pp, qf, pf in zip(q_index_p, p_index_p, q_index_f, p_index_f):
        assert qp == qf
        assert pp == pf


def test_bin_sum_python_fortran_match():
    fsum = polar.bin_sum(
        n_q_bins=polar_shape[0],
        n_p_bins=polar_shape[1],
        q_index=q_index_f,
        p_index=p_index_f,
        array=data,
        py=False,
    )
    psum = polar.bin_sum(
        n_q_bins=polar_shape[0],
        n_p_bins=polar_shape[1],
        q_index=q_index_p,
        p_index=p_index_p,
        array=data,
        py=True,
    )
    for ps, fs in zip(psum.ravel(), fsum.ravel()):
        assert ps == fs


def test_bin_mean_python_fortran_match():
    args = [
        polar_shape[0],
        q_bin_size,
        q_min,
        polar_shape[1],
        phi_bin_size,
        phi_min,
        qs,
        phis,
        mask,  # weights
        data,
        mask,
    ]
    pmean, pmask = polar.bin_mean(*args, py=True)
    pmean_f, pmask_f = polar.bin_mean(*args, py=False)
    assert pmean.all() == pmean_f.all()
    assert pmask.all() == pmask_f.all()


def test_bin_mean_python():
    pmean, pmask = polar.bin_mean(
        n_q_bins=polar_shape[0],
        q_bin_size=q_bin_size,
        q_min=q_min,
        n_p_bins=polar_shape[1],
        p_bin_size=phi_bin_size,
        p_min=phi_min,
        qs=qs,
        ps=phis,
        weights=mask,
        data=data,
        mask=mask,
        py=True,
    )
    assert pmean[0, 0] == 0.5
    assert pmean[0, 1] == 0.5
    assert pmean[0, 2] == 0.0
    assert pmean[1, 0] == 0.0
    assert pmean[1, 1] == 0.5
    assert pmean[1, 2] == 0.5
    assert pmask[0, 0] == 1.0
    assert pmask[0, 1] == 1.0
    assert pmask[0, 2] == 0.0
    assert pmask[1, 0] == 0.0
    assert pmask[1, 1] == 1.0
    assert pmask[1, 2] == 1.0


def test_bin_mean_fortran():
    fmean, fmask = polar.bin_mean(
        n_q_bins=polar_shape[0],
        q_bin_size=q_bin_size,
        q_min=q_min,
        n_p_bins=polar_shape[1],
        p_bin_size=phi_bin_size,
        p_min=phi_min,
        qs=qs,
        ps=phis,
        weights=mask,
        data=data,
        mask=mask,
        py=False,
    )
    assert fmean[0, 0] == 0.5
    assert fmean[0, 1] == 0.5
    assert fmean[0, 2] == 0.0
    assert fmean[1, 0] == 0.0
    assert fmean[1, 1] == 0.5
    assert fmean[1, 2] == 0.5
    assert fmask[0, 0] == 1.0
    assert fmask[1, 2] == 1.0
    assert fmask[0, 0] == 1.0
    assert fmask[0, 1] == 1.0
    assert fmask[0, 2] == 0.0
    assert fmask[1, 0] == 0.0
    assert fmask[1, 1] == 1.0
    assert fmask[1, 2] == 1.0


def test_polar_stats():
    print()
    nq = 3
    nphi = 4
    data = np.arange(nq * nphi, dtype=np.float64).reshape(nq, nphi)
    data[0, 1]
    weights = np.ones_like(data)
    weights[0, 1] = 0
    q = np.arange(nq, dtype=np.float64) + 1
    p = np.arange(nphi, dtype=np.float64) + 1
    q, p = np.meshgrid(q, p, indexing="ij")
    q = q.ravel()
    p = p.ravel()
    stats = polar.stats(
        data,
        q,
        p,
        weights=weights,
        n_q_bins=nq,
        q_min=1,
        q_max=nq,
        n_p_bins=nphi,
        p_min=1,
        p_max=nphi,
        sum_=None,
        sum2=None,
        w_sum=None,
    )
    assert stats["mean"][0, 0] == 0
    assert stats["mean"][0, 1] == 0
    assert stats["mean"][0, 2] == 2
