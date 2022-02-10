import numpy as np


def correlate(s1, s2=None):
    r"""
    Computes correlation function.
    If two signals are provided computes cross correlation.
    If one singal is provided computes auto correlation.

    Computed via Fourier transforms:
        cf = iFT(FT(s1) FT(s2)*)

    Arguments:
        s1 (|ndarray|): signal 1
        s2 (|ndarray|): signal 2
    Returns:
        cf (|ndarray|): correlation of s1 and s2
    """
    a = np.fft.fft(s1)
    b = a.copy() if s2 is None else np.fft.fft(s2)
    return np.real(np.fft.ifft(a * np.conj(b)))


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


def data_correlation(n, data, mask):
    r"""
    Computes cross correlation of data with data shifted by n.

    Note: For n = 0 this returns the auto correlation of the data.

    Arguments:
        n (int): number of q rings to shift
        data (|ndarray|): data
        mask (|ndarray|): mask (i.e. data to ignore)
    Returns:
        ccf (|ndarray|): cross correlation function of data
    """
    data = subtract_masked_data_mean(data, mask)
    zros = np.zeros_like(data)
    if n == 0:
        d_cf = correlate(data)
        m_cf = correlate(mask)
    else:
        d_cf = correlate(data, np.roll(data, n, axis=0))
        m_cf = correlate(mask, np.roll(mask, n, axis=0))
    return np.divide(d_cf, m_cf, out=zros, where=m_cf != 0)
