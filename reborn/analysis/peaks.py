from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy.ndimage import measurements
from ..fortran import peaks_f


class PeakFinder(object):
    r"""
    A crude peak finder.  It works by firstly running a signal-to-noise filter over an entire image, and then searching
    for connected regions with SNR above some threshold.  It is not fully developed.  For example, it does not yet
    check for the minimum distance between peaks.
    """

    mask = None  #: Ignore pixels where mask == 0.
    snr_threshold = 10  # : Peaks must have a signal-to-noise ratio above this value.
    radii = [1, 20, 30]  # : These are the radii associated with the :func:`boxsnr <reborn.analysis.peaks.boxsnr>` function.

    snr = None  # : The SNR array from the most recent call to the find_peaks method.
    signal = None  # : The signal array from the most recent call to the find_peaks method.
    labels = None  # : Labeled regions of peak candidates
    n_labels = 0  # : Number of peak candidates
    centroids = None  # : Centroids of each peak candidate

    def __init__(self, snr_threshold=10, radii=(3, 8, 10), mask=None):
        r"""
        Args:
            snr_threshold: Peaks must have a signal-to-noise ratio above this value
            radii: These are the radii associated with the :func:`boxsnr <reborn.analysis.peaks.boxsnr>` function.
            mask: Ignore pixels where mask == 0.
        """
        self.snr_threshold = snr_threshold
        self.radii = radii
        self.mask = mask

    def find_peaks(self, data, mask=None):
        r"""
        Do peak finding on a data array.

        Args:
            data: The data to perform peak finding on.
            mask: A mask, if desired.  By default, the mask used on creation of a PeakFinder instance will be used.
                  This defaults to ones everywhere.

        Returns:

        """
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


# def boxconv(dat, width):
#
#     r"""
#     Arguments:
#         dat: The image to analyze
#         mask: The mask for the square central integration region
#         mask2: The mask for the square annulus integration region
#         nin: Size of the central integration region; integrate from (-nin, nin), inclusively.
#         ncent: Define the annulus integration region; we ignore the box from (-ncent, ncent), inclusively
#         nout: Define the annulus integration region; we include the box from (-nout, nout), inclusively
#
#     Returns: snr (numpy array), signal (numpy array)
#     """
#
#     float_t = np.float64
#     datconv = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
#     d = np.asfortranarray(dat.astype(float_t))
#     peaks_f.boxconv(d, datconv, width)
#     return datconv
#
