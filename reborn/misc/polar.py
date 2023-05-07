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
    q_index[q_index < 0] = -1
    q_index[q_index > n_q_bins] = -1
    return q_index.astype(int)


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
    p_index[p_index < 0] = -1
    p_index[p_index > n_p_bins] = -1
    return p_index.astype(int)


def bin_indices(n_q_bins, q_bin_size, q_min,
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


def bin_sum(n_q_bins, n_p_bins, q_index, p_index, array, py=False):
    r""" Polar p bin indices.
    Arguments:
        n_q_bins (int): number of q bins
        n_p_bins (int): number of phi bins
        q_index (|ndarray|): q bin indices
        p_index (|ndarray|): phi bin indices
        array (|ndarray|): array to bin
        py (bool): computes with fortran if set to false

    Returns:
        a_sum (|ndarray|): binned array
    """
    if py:
        a_sum = np.zeros((n_q_bins + 1, n_p_bins + 1))
        # for a, qi, pi in zip(array, q_index, p_index):
        #     a_sum[qi, pi] += a
        order = np.lexsort((q_index, p_index))
        row = q_index[order]
        col = p_index[order]
        data = array[order]
        unique_mask = np.append(True, ((row[1:] != row[:-1]) |
                                       (col[1:] != col[:-1])))
        unique_indices, = np.nonzero(unique_mask)
        qi = row[unique_mask]
        pi = col[unique_mask]
        a = np.add.reduceat(data, unique_indices)
        a_sum[qi, pi] = a
        a_sum = a_sum[0:-1, 0:-1]
    else:  # fortran
        q_i = q_index + 1
        p_i = p_index + 1
        a = fortran.polar_f.polar_binning.bin_sum(n_q_bins, n_p_bins,
                                                  q_i, p_i, array)
        a_sum = a.reshape((n_q_bins, n_p_bins)).astype(float)
    return a_sum


def bin_mean(n_q_bins, q_bin_size, q_min,
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
    if py:  # python
        polar_mean_data = np.zeros(polar_shape)
        polar_mean_mask = np.zeros(polar_shape)
        q_index = q_bin_indices(n_q_bins, q_bin_size, q_min, qs)
        p_index = p_bin_indices(n_p_bins, p_bin_size, p_min, ps)
        keepers = (q_index >= 0) & (p_index >= 0) & (mask != 0)
        dsum = bin_sum(n_q_bins, n_p_bins, q_index[keepers],
                       p_index[keepers], data[keepers], py=py)
        wsum = bin_sum(n_q_bins, n_p_bins, q_index[keepers],
                       p_index[keepers], weights[keepers], py=py)
        csum = bin_sum(n_q_bins, n_p_bins, q_index[keepers],
                       p_index[keepers], mask[keepers], py=py).astype(int)
        np.divide(dsum, csum, out=polar_mean_data, where=csum != 0)
        np.divide(wsum, csum, out=polar_mean_mask, where=csum != 0)
    else:  # fortran
        args = [n_q_bins, q_bin_size, q_min, n_p_bins, p_bin_size, p_min,
                qs.ravel(), ps.ravel(), weights.ravel(), data.ravel()]
        polar_data, polar_mask = fortran.polar_f.polar_binning.bin_mean(*args)
        polar_mean_data = polar_data.reshape(polar_shape).astype(float)
        polar_mean_mask = polar_mask.reshape(polar_shape).astype(float)
    return polar_mean_data, polar_mean_mask


def stats(data, q, p, weights=None,
          n_q_bins=100, q_min=0, q_max=3e10,
          n_p_bins=360, p_min=0, p_max=3.141592653589793,
          sum_=None, sum2=None, w_sum=None):
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
    fortran.polar_f.polar_binning.stats(data.T, q.T, p.T, weights.T,
                                        n_q_bins, q_min, q_max,
                                        n_p_bins, p_min, p_max,
                                        sum_.T, sum2.T, w_sum.T, 1)
    meen = np.empty(shape, dtype=np.float64)
    std = np.empty(shape, dtype=np.float64)
    fortran.polar_f.polar_binning.stats_mean(sum_.T, sum2.T, w_sum.T, meen.T, std.T)
    out = dict(mean=meen, sdev=std, sum=sum_, sum2=sum2, weight_sum=w_sum)
    return out
