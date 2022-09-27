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


def polar_mean(n_q_bins, q_bin_size, q_min,
               n_phi_bins, phi_bin_size, phi_min,
               qs, phis, weights,
               data, mask):
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
        weight (|ndarray|): weight factors
        data (|ndarray|): The PAD data to be binned.
        mask (|ndarray|): A mask to indicate ignored pixels.

    Returns:
        polar_mean (|ndarray|): the (n_q_bins x n_phi_bins) array of polar binned data
        polar_mask (|ndarray|): the (n_q_bins x n_phi_bins) array of polar binned mask * weight
    """
    m = mask.astype(bool)
    dat = data[m]
    wgt = weights[m]
    _p = phis % (2 * np.pi)
    p_index = np.floor((_p - phi_min) / phi_bin_size).astype(int)
    q_index = np.floor((qs - q_min) / q_bin_size).astype(int)
    # conditions
    _cq = (q_index >= n_q_bins) | (q_index < 0)
    _cp = (p_index >= n_phi_bins) | (p_index < 0)
    keepers = ~ (_cq | _cp)
    # data to keep
    dk = dat[keepers]
    dw = wgt[keepers]
    # calculate average binned pixel
    shp = n_q_bins * n_phi_bins
    counts = np.zeros(shp, dtype=int)  # number of raw pixels binned to polar pixel
    dsum = np.zeros(shp, dtype=float)  # binned data sum
    pbdw = np.zeros(shp, dtype=float)  # pixel binned data weights
    idx = (n_phi_bins * q_index[keepers] + p_index[keepers]).astype(int)
    slices = np.array_split(idx, shp)
    d = np.array_split(dk, shp)
    w = np.array_split(dw, shp)
    for s, dd, ww in zip(slices, d, w):
        np.add.at(counts, s, 1)
        np.add.at(dsum, s, dd)
        np.add.at(pbdw, s, ww)
    pmask = np.zeros(shp, dtype=float)
    pmean = np.zeros(shp, dtype=float)
    np.divide(pbdw, counts, out=pmask, where=counts != 0)
    np.divide(dsum, pmask, out=pmean, where=pmask != 0)
    return pmean, pmask
