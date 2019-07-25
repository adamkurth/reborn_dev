from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


import numpy as np
from scipy.ndimage import measurements
from bornagain.utils import warn

try:
    from bornagain.fortran import peaks_f
except ImportError:
    warn('You need to compile the fortran code.  See the documentation: https://rkirian.gitlab.io/bornagain')
    peaks_f = None


class PeakFinder(object):

    mask = None
    snr_threshold = None
    radii = None

    snr = None
    signal = None
    labels = None
    n_labels = 0
    centroids = None

    def __init__(self, snr_threshold=10, radii=(3, 8, 10), mask=None):

        self.snr_threshold = snr_threshold
        self.radii = radii
        self.mask = mask

    def find_peaks(self, data, mask=None):

        if mask is None:
            if self.mask is None:
                self.mask = np.ones_like(data)
            mask = self.mask

        self.snr, self.signal = boxsnr(data, mask, mask, self.radii[0], self.radii[1], self.radii[2])
        self.labels, self.n_labels = measurements.label(self.snr > self.snr_threshold)
        print('self.n_labels', self.n_labels)
        if self.n_labels > 0:
            sig = self.signal.copy()
            sig[sig < 0] = 0
            cent = measurements.center_of_mass(sig, self.labels, np.arange(1, self.n_labels+1))
            cent = np.array(cent)
            if len(cent.shape) == 1:
                cent = np.expand_dims(cent, axis=0)
            cent = cent[:, ::-1].copy()
            print('cent', cent)
            self.centroids = cent
        else:
            self.centroids = None

        return self.centroids


def boxsnr(dat, mask, mask2, nin, ncent, nout):

    r"""
    Arguments:
        dat: The image to analyze
        mask: The mask for the square central integration region
        mask2: The mask for the square annulus integration region
        nin: Size of the central integration region; integrate from (-nin, nin), inclusively.
        ncent: Define the annulus integration region; we ignore the box from (-ncent, ncent), inclusively
        nout: Define the annulus integration region; we include the box from (-nout, nout), inclusively

    Returns: snr (numpy array), signal (numpy array)
    """

    float_t = np.float64
    snr = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    signal = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    d = np.asfortranarray(dat.astype(float_t))
    m = np.asfortranarray(mask.astype(float_t))
    m2 = np.asfortranarray(mask2.astype(float_t))
    peaks_f.boxsnr(d, m, m2, snr, signal, nin, ncent, nout)
    return snr, signal


def boxconv(dat, width):

    r"""
    Arguments:
        dat: The image to analyze
        mask: The mask for the square central integration region
        mask2: The mask for the square annulus integration region
        nin: Size of the central integration region; integrate from (-nin, nin), inclusively.
        ncent: Define the annulus integration region; we ignore the box from (-ncent, ncent), inclusively
        nout: Define the annulus integration region; we include the box from (-nout, nout), inclusively

    Returns: snr (numpy array), signal (numpy array)
    """

    float_t = np.float64
    datconv = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    d = np.asfortranarray(dat.astype(float_t))
    peaks_f.boxconv(d, datconv, width)
    return datconv

