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
from .. import fortran


def get_polar_bin_indices_python(n_q_bins, q_bin_size, q_min,
                                 n_phi_bins, phi_bin_size, phi_min,
                                 qs, phis):
    tp = 2 * np.pi
    q_index = np.floor((qs - q_min) / q_bin_size)
    # ps = phis % 2 * np.pi
    # https://gcc.gnu.org/onlinedocs/gfortran/MODULO.html#MODULO
    # MODULO(A, P) = A - FLOOR(A / P) * P
    ps = phis - np.floor(phis / tp) * tp  # modulo as defined in fortran
    p_index = np.floor((ps - phi_min) / phi_bin_size)
    q_index[q_index < 0] = 0
    q_index[q_index > n_q_bins] = 0
    p_index[p_index < 0] = 0
    p_index[p_index > n_phi_bins] = 0
    return q_index.astype(int), p_index.astype(int)


def get_polar_bin_indices_fortran(n_q_bins, q_bin_size, q_min,
                                  n_phi_bins, phi_bin_size, phi_min,
                                  qs, phis):
    args = [n_q_bins, q_bin_size, q_min, n_phi_bins, phi_bin_size, phi_min, qs, phis]
    q_index, p_index = fortran.polar_f.polar_binning.polar_bin_indices(*args)
    return q_index.astype(int), p_index.astype(int)


def get_polar_bin_indices(n_q_bins, q_bin_size, q_min,
                          n_phi_bins, phi_bin_size, phi_min,
                          qs, phis, python=False):
    r""" Create the mean polar-binned average intensities.
    Arguments:
        n_q_bins (int): number of q bins
        q_bin_size (float): q bin size
        q_min (float): minimum q
        n_phi_bins (int): number of phi bins
        phi_bin_size (float): phi bin size
        phi_min (float): minimum phi
        qs (|ndarray|): q coordinates
        phis (|ndarray|): phi coordinates
        python (bool): compute in python instead of fortran

    Returns:
        q_index (|ndarray|): q polar mapping index (value of 0 means out of bounds)
        p_index (|ndarray|): phi polar mapping index (value of 0 means out of bounds)
    """
    args = [n_q_bins, q_bin_size, q_min, n_phi_bins, phi_bin_size, phi_min, qs, phis]
    if python:
        q_index, p_index = get_polar_bin_indices_python(*args)
    else:
        q_index, p_index = get_polar_bin_indices_fortran(*args)
    return q_index, p_index


def get_polar_bin_mean_python(n_q_bins, q_bin_size, q_min,
                              n_phi_bins, phi_bin_size, phi_min,
                              qs, phis, weights,
                              data, mask):
    q_index, p_index = get_polar_bin_indices_python(n_q_bins, q_bin_size, q_min,
                                                    n_phi_bins, phi_bin_size, phi_min,
                                                    qs, phis)
    polar_shape = (n_q_bins, n_phi_bins)
    dsum = np.zeros(polar_shape)
    wsum = np.zeros(polar_shape)
    count = np.zeros(polar_shape, dtype=int)
    keepers = (q_index != 0) & (p_index != 0) & (mask != 0)
    for d, w, qi, pi in zip(data[keepers], weights[keepers], q_index[keepers], p_index[keepers]):
        dsum[qi, pi] += d
        wsum[qi, pi] += w
        count[qi, pi] += 1
    polar_mean_data = np.zeros(polar_shape)
    polar_mean_mask = np.zeros(polar_shape)
    np.divide(dsum, count, out=polar_mean_data, where=count != 0)
    np.divide(wsum, count, out=polar_mean_mask, where=count != 0)
    return polar_mean_data, polar_mean_mask


def get_polar_bin_mean_fortran(n_q_bins, q_bin_size, q_min,
                               n_phi_bins, phi_bin_size, phi_min,
                               qs, phis, weights,
                               data, mask):
    polar_shape = (n_q_bins, n_phi_bins)
    args = [n_q_bins, q_bin_size, q_min, n_phi_bins, phi_bin_size, phi_min,
            qs.ravel(), phis.ravel(), weights.ravel(), data.ravel(), mask.astype(int).ravel()]
    polar_mean_data, polar_mean_mask = fortran.polar_f.polar_binning.polar_bin_avg(*args)
    return polar_mean_data.reshape(polar_shape).astype(float), polar_mean_mask.reshape(polar_shape).astype(float)


def get_polar_bin_mean(n_q_bins, q_bin_size, q_min,
                       n_phi_bins, phi_bin_size, phi_min,
                       qs, phis, weights,
                       data, mask, python=False):
    r""" Create the mean polar-binned average intensities.
    Arguments:
        n_q_bins (int): number of q bins
        q_bin_size (float): q bin size
        q_min (float): minimum q
        n_phi_bins (int): number of phi bins
        phi_bin_size (float): phi bin size
        phi_min (float): minimum phi
        qs (|ndarray|): q coordinates
        phis (|ndarray|): phi coordinates
        weights (|ndarray|): weight factors
        data (|ndarray|): The PAD data to be binned.
        mask (|ndarray|): A mask to indicate ignored pixels.

    Returns:
        polar_mean (|ndarray|): the (n_q_bins x n_phi_bins) array of polar binned data
        polar_mask (|ndarray|): the (n_q_bins x n_phi_bins) array of polar binned mask * weight
    """
    args = [n_q_bins, q_bin_size, q_min, n_phi_bins, phi_bin_size, phi_min, qs, phis, weights, data, mask]
    if python:
        polar_mean_data, polar_mean_mask = get_polar_bin_mean_python(*args)
    else:
        polar_mean_data, polar_mean_mask = get_polar_bin_mean_fortran(*args)
    return polar_mean_data, polar_mean_mask


def get_polar_stats(data, q, p, weights=None, n_q_bins=100, q_min=0, q_max=3e10, n_p_bins=360, p_min=0,
                    p_max=3.141592653589793, sum_=None, sum2=None, w_sum=None):
    # Check all datatypes
    n_q_bins = int(n_q_bins)
    n_p_bins = int(n_p_bins)
    q_min = float(q_min)
    q_max = float(q_max)
    p_min = float(p_min)
    p_max = float(p_max)
    if data.dtype != np.float64:
        raise ValueError('Data type must be np.float64')
    if weights is None:
        weights = np.ones_like(data)
    if weights.dtype != np.float64:
        raise ValueError('Data type must be np.float64')
    if q.dtype != np.float64:
        raise ValueError('Data type must be np.float64')
    if p.dtype != np.float64:
        raise ValueError('Data type must be np.float64')
    shape = (n_q_bins, n_p_bins)
    if sum_ is None:
        sum_ = np.zeros(shape, dtype=np.float64)
    if sum2 is None:
        sum2 = np.zeros(shape, dtype=np.float64)
    if w_sum is None:
        w_sum = np.zeros(shape, dtype=np.float64)
    if sum_.dtype != np.float64:
        raise ValueError('Data type must be np.float64')
    if sum2.dtype != np.float64:
        raise ValueError('Data type must be np.float64')
    if w_sum.dtype != np.float64:
        raise ValueError('Data type must be np.float64')
    # if indices is None:  # Cache for faster computing next time.
    #     indices = np.zeros(len(q), dtype=np.int32)
    #     fortran.scatter_f.profile_indices(q, n_bins, q_min, q_max, indices)
    #     self._fortran_indices = indices
    data = data.ravel()
    q = q.ravel()
    p = p.ravel()
    weights = weights.ravel()
    sum_ = sum_.reshape(n_q_bins, n_p_bins)
    sum2 = sum2.reshape(n_q_bins, n_p_bins)
    w_sum = w_sum.reshape(n_q_bins, n_p_bins)
    fortran.polar_f.polar_binning.polar_stats(data.T, q.T, p.T, weights.T, n_q_bins, q_min, q_max, n_p_bins, p_min,
                                              p_max,
                                              sum_.T, sum2.T, w_sum.T, 1)
    meen = np.empty(shape, dtype=np.float64)
    std = np.empty(shape, dtype=np.float64)
    fortran.polar_f.polar_binning.polar_stats_avg(sum_.T, sum2.T, w_sum.T, meen.T, std.T)
    out = dict(mean=meen, sdev=std, sum=sum_, sum2=sum2, weight_sum=w_sum)
    return out
