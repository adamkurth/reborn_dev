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
    snr_threshold = None  # : Peaks must have a signal-to-noise ratio above this value.
    radii = None  # : These are the radii associated with the :func:`boxsnr <reborn.analysis.peaks.boxsnr>` function.
    snr = None  # : The SNR array from the most recent call to the find_peaks method.
    signal = None  # : The signal array from the most recent call to the find_peaks method.
    labels = None  # : Labeled regions of peak candidates
    n_labels = 0  # : Number of peak candidates
    centroids = None  # : Centroids of each peak candidate
    beam = None  # :

    def __init__(self, snr_threshold=6, radii=(3, 5, 10), mask=None, max_iterations=3, beam=None):
        r"""
        Args:
            snr_threshold: Peaks must have a signal-to-noise ratio above this value
            radii: These are the radii associated with the :func:`boxsnr <reborn.analysis.peaks.boxsnr>` function.
            mask: Ignore pixels where mask == 0.
        """
        self.snr_threshold = snr_threshold
        self.radii = radii
        self.mask = mask
        self.max_iterations = max_iterations

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
        mask_a = mask.copy()
        for i in range(self.max_iterations):
            snr, signal = boxsnr(data, mask, mask_a, self.radii[0], self.radii[1], self.radii[2])
            ab = snr > self.snr_threshold
            if np.sum(ab) > 0:
                mask_a[ab] = 0
            else:
                break
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


def boxsnr(dat, mask_in, mask_out, n_in, n_cent, n_out):

    r"""
    Transform an 2D image into a map of local signal-to-noise ratio by the following equivalent steps:

    (1) For every pixel in the input data, do a local signal integration within a square region of size
        :math:`n_\text{in}*2+1`.  Pixels masked by `mask_in` will be ignored.  Masked pixels are indicated by the value
        zero, while unmasked pixels are indicated by the value one.

    (2) Estimate background via a local integration within a square annulus of outer size 
        :math:`2 n_\text{out} + 1` and inner size :math:`2 n_\text{cent} - 1`.  Pixels within `mask_out` will be
        ignored.

    (3) From every pixel in the local signal integration square, subtract the average background value from step (2).

    (4) Compute the standard deviation :math:`\sigma` in the square annulus.  Pixels within `mask_out` will be ignored.

    (5) Divide the locally-integrated signal-minus-background by the standard error.  The standard error is 
        equal to :math:`\sigma*\sqrt(M)` where :math:`M` is the number of unmasked pixels in the locally-integratied
        signal region, and :math:`\sigma` comes from step (4).

    Note: The use of two distinct masks allows for multi-pass SNR computations in which the results of the first pass 
    may be used to exclude high-SNR regions from contributing to error estimates in the annulus.  See
    :func:`snr_mask <reborn.analysis.peaks.snr_mask>` if you want to generate a mask this way.

    Note: This routine will attempt to use openmp to parallelize the computations.  It is affected by the environment
    variable `OMP_NUM_THREADS`.

    Arguments:
        dat: The image to analyze
        mask_in: The mask for the square central integration region
        mask_out: The mask for the square annulus integration region
        n_in: Size of the central integration region; integrate from (-nin, nin), inclusively.
        n_cent: Define the annulus integration region; we ignore the box from (-ncent, ncent), inclusively
        n_out: Define the annulus integration region; we include the box from (-nout, nout), inclusively

    Returns: snr (numpy array), signal (numpy array)
    """

    float_t = np.float64
    snr = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    signal = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    d = np.asfortranarray(dat.astype(float_t))
    m = np.asfortranarray(mask_in.astype(float_t))
    m2 = np.asfortranarray(mask_out.astype(float_t))
    peaks_f.boxsnr(d, m, m2, snr, signal, n_in, n_cent, n_out)
    return snr, signal


def snr_mask(dat, mask, nin, ncent, nout, threshold, mask_negative=True, max_iterations=3):
    r"""
    Mask out pixels above some chosen SNR threshold.  The image is converted to a map of SNR using boxsnr.  Additional
    iterations follow, in which pixels above threshold in the previous run are also masked in the annulus.
    This iterative procedure helps avoid contributions of peak signals to the Noise calculation.

    Arguments:
        dat (numpy array) : Input data to calculate SNR from.
        mask (numpy array) : Mask indicating bad pixels (zero means bad, one means ok)
        nin (int) : See boxsnr function.
        ncent (int) : See boxsnr function.
        nout (int) : See boxsnr function.
        threshold (float) : Reject pixels above this SNR.
        mask_negative (bool) : Also reject pixels below the negative of the SNR threshold (default: True).
        max_iterations (int) : The maxumum number of iterations (note: the loop exits if the mask stops changing).

    Returns:
        numpy array : The mask with pixels above the SNR threshold
    """

    mask_a = mask.copy()
    for i in range(max_iterations):
        a, _ = boxsnr(dat, mask, mask_a, nin, ncent, nout)
        ab = a > threshold
        if mask_negative:
            ab *= a < -threshold
        if np.sum(ab) > 0:
            mask_a[ab] = 0
        else:
            break
    return mask_a



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
