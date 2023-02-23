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
import numba as nb
from .. import fortran


@nb.njit('i8[:](i8, f8, f8, f8[:])')
def q_bin_indices(n_q_bins, q_bin_size, q_min, qs):
    r""" Polar q bin indices.
    Arguments:
        n_q_bins (int): number of q bins
        q_bin_size (float): q bin size
        q_min (float): minimum q
        qs (|ndarray|): q coordinates

    Returns:
        q_index (|ndarray|): q polar mapping index (value of 0 means out of bounds)
    """
    q_index = np.floor((qs - q_min) / q_bin_size)
    q_index[q_index < 0] = 0
    q_index[q_index > n_q_bins] = 0
    return q_index.astype(nb.i8)


@nb.njit('i8[:](i8, f8, f8, f8[:])')
def p_bin_indices(n_p_bins, p_bin_size, p_min, ps):
    r""" Polar p bin indices.
    Arguments:
        n_p_bins (int): number of phi bins
        p_bin_size (float): phi bin size
        p_min (float): minimum phi
        ps (|ndarray|): phi coordinates

    Returns:
        p_index (|ndarray|): phi polar mapping index (value of 0 means out of bounds)
    """
    tp = 2 * np.pi
    # https://gcc.gnu.org/onlinedocs/gfortran/MODULO.html#MODULO
    # MODULO(A, P) = A - FLOOR(A / P) * P
    # ps = phis % 2 * np.pi
    p = ps - np.floor(ps / tp) * tp  # modulo as defined in fortran
    p_index = np.floor((p - p_min) / p_bin_size)
    p_index[p_index < 0] = 0
    p_index[p_index > n_p_bins] = 0
    return p_index.astype(nb.i8)


@nb.njit('f8[:, :](i8, i8, i8[:], i8[:], f8[:])')
def bin_sum(n_q_bins, n_p_bins, q_index, p_index, array):
    r""" Polar p bin indices.
    Arguments:
        n_q_bins (int): number of q bins
        n_p_bins (int): number of phi bins
        q_index (|ndarray|): q bin indices
        p_index (|ndarray|): phi bin indices
        array (|ndarray|): array to bin

    Returns:
        a_sum (|ndarray|): binned array
    """
    a_sum = np.zeros((n_q_bins, n_p_bins))
    for a, qi, pi in zip(array, q_index, p_index):
        a_sum[qi, pi] += a
    return a_sum


def polar_bin_indices(n_q_bins, q_bin_size, q_min,
                      n_p_bins, p_bin_size, p_min,
                      qs, ps, py=False):
    r""" Create the mean polar-binned average intensities.
    Arguments:
        n_q_bins (int): number of q bins
        q_bin_size (float): q bin size
        q_min (float): minimum q
        n_p_bins (int): number of phi bins
        p_bin_size (float): phi bin size
        p_min (float): minimum phi
        qs (|ndarray|): q coordinates
        ps (|ndarray|): phi coordinates
        py (bool): compute in python instead of fortran

    Returns:
        q_index (|ndarray|): q polar mapping index (value of 0 means out of bounds)
        p_index (|ndarray|): phi polar mapping index (value of 0 means out of bounds)
    """
    q_args = [n_q_bins, q_bin_size, q_min, qs]
    p_args = [n_p_bins, p_bin_size, p_min, ps]
    if py:  # python with numba
        q_index = q_bin_indices(*q_args)
        p_index = p_bin_indices(*p_args)
    else:  # fortran
        q_index = fortran.polar_f.polar_binning.q_bin_indices(*q_args)
        p_index = fortran.polar_f.polar_binning.p_bin_indices(*p_args)
        # compensate fortran indexing from 1 instead of 0
        q_index -= 1
        p_index -= 1
    return q_index.astype(int), p_index.astype(int)


def get_polar_bin_mean(n_q_bins, q_bin_size, q_min,
                       n_p_bins, p_bin_size, p_min,
                       qs, ps, weights,
                       data, mask, py=False):
    r""" Create the mean polar-binned average intensities.
    Arguments:
        n_q_bins (int): number of q bins
        q_bin_size (float): q bin size
        q_min (float): minimum q
        n_p_bins (int): number of phi bins
        p_bin_size (float): phi bin size
        p_min (float): minimum phi
        qs (|ndarray|): q coordinates
        ps (|ndarray|): phi coordinates
        weights (|ndarray|): weight factors
        data (|ndarray|): The PAD data to be binned.
        mask (|ndarray|): A mask to indicate ignored pixels.
        py (bool): compute in python instead of fortran

    Returns:
        polar_mean (|ndarray|): the (n_q_bins x n_phi_bins) array of polar binned data
        polar_mask (|ndarray|): the (n_q_bins x n_phi_bins) array of polar binned mask * weight
    """
    polar_shape = (n_q_bins, n_p_bins)
    if py:  # python with numba
        polar_mean_data = np.zeros(polar_shape)
        polar_mean_mask = np.zeros(polar_shape)
        q_index = q_bin_indices(n_q_bins, q_bin_size, q_min, qs)
        p_index = p_bin_indices(n_p_bins, p_bin_size, p_min, ps)
        keepers = (q_index != 0) & (p_index != 0) & (mask != 0)
        dsum = bin_sum(n_q_bins, n_p_bins, q_index[keepers], p_index[keepers], data[keepers])
        wsum = bin_sum(n_q_bins, n_p_bins, q_index[keepers], p_index[keepers], weights[keepers])
        count = bin_sum(n_q_bins, n_p_bins, q_index[keepers], p_index[keepers], mask[keepers]).astype(int)
        np.divide(dsum, count, out=polar_mean_data, where=count != 0)
        np.divide(wsum, count, out=polar_mean_mask, where=count != 0)
    else:  # fortran
        args = [n_q_bins, q_bin_size, q_min, n_p_bins, p_bin_size, p_min,
                qs.ravel(), ps.ravel(), weights.ravel(), data.ravel(), mask.astype(int).ravel()]
        polar_data, polar_mask = fortran.polar_f.polar_binning.polar_bin_avg(*args)
        polar_mean_data = polar_data.reshape(polar_shape).astype(float)
        polar_mean_mask = polar_mask.reshape(polar_shape).astype(float)
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
