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


def correlate(s1, s2=None, cached=False):
    r"""
    Computes correlation function.
    If two signals are provided computes cross correlation.
    If one singal is provided computes auto correlation.
    If cached, assumes Fourier transforms are already computed.

    Computed via Fourier transforms:
        cf = iFT(FT(s1) FT(s2)*)

    Arguments:
        s1 (|ndarray|): signal 1
        s2 (|ndarray|): signal 2
        cache (bool): provide ffts instead of computing
    Returns:
        correlation (|ndarray|): correlation of s1 and s2
    """
    if cached:
        a = s1
        b = a.copy() if s2 is None else s2
    else:
        a = np.fft.fft(s1, axis=1)
        b = a.copy() if s2 is None else np.fft.fft(s2, axis=1)
    correlation = np.fft.ifft(a * np.conj(b))
    return np.real(correlation)


def subtract_masked_data_mean(data, mask):
    r"""
    Subtract average q for each q ring in data, ignores masked pixels.
    This normalizes and centers the data around 0.

    Arguments:
        data (|ndarray|): data
        mask (|ndarray|): mask (i.e. data to ignore)
    Returns:
        data (|ndarray|): (data - <data>_q) / max(data)
    """
    d_sum = np.sum(data, axis=1)
    count = np.sum(mask, axis=1)
    d_zro = np.zeros_like(d_sum)
    data -= np.divide(d_sum, count, out=d_zro, where=count != 0)
    data /= np.max(data)  # normalization
    data *= mask  # re-zero masked pixels
    return data


def data_correlation(n, data, mask, cached=False):
    r"""
    Computes cross correlation of data with data shifted by n.

    Note: For n = 0 this returns the auto correlation of the data.

    Arguments:
        n (int): number of q rings to shift
        data (|ndarray|): data
        mask (|ndarray|): mask (i.e. data to ignore)
        cache (bool): provide ffts instead of computing
    Returns:
        ccf (|ndarray|): cross correlation function of data
    """
    zros = np.zeros_like(data)
    if not cached:
        data = subtract_masked_data_mean(data, mask)
    if n == 0:
        d_cf = correlate(s1=data, cached=cached)
        m_cf = correlate(s1=mask, cached=cached)
    else:
        data_roll = np.roll(data, n, axis=0)
        mask_roll = np.roll(mask, n, axis=0)
        d_cf = correlate(s1=data, s2=data_roll, cached=cached)
        m_cf = correlate(s1=mask, s2=mask_roll, cached=cached)
    return np.divide(d_cf, m_cf, out=zros, where=m_cf != 0)


def compute_data_correlations(data, mask):
    r"""
    Computes cross correlation of data with data shifted by n.

    Note: For n = 0 this returns the auto correlation of the data.

    Arguments:
        data (|ndarray|): data
        mask (|ndarray|): mask (i.e. data to ignore)
    Returns:
        correlations (dict): correlations of data
    """
    data = subtract_masked_data_mean(data, mask)
    data = np.fft.fft(data, axis=1)
    mask = np.fft.fft(mask, axis=1)
    q_range = data.shape[0]
    correlations = {n: data_correlation(n=n, data=data,
                                        mask=mask, cached=True)
                    for n in range(q_range)}
    return correlations
