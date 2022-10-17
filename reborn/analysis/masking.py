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
from ..fortran import peaks_f
from ..detector import RadialProfiler


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
        equal to :math:`\sigma/\sqrt{M}` where :math:`M` is the number of unmasked pixels in the locally-integratied
        signal region, and :math:`\sigma` comes from step (4).

    The use of two distinct masks allows for multi-pass SNR computations in which the results of the first pass
    may be used to exclude high-SNR regions from contributing to error estimates in the annulus.  See
    :func:`snr_mask <reborn.analysis.peaks.snr_mask>` if you want to generate a mask this way.

    Note:
        This routine will attempt to use openmp to parallelize the computations.  It is affected by the environment
        variable `OMP_NUM_THREADS`.  You can check how many cores are used by openmp by running the following:

        .. code-block::

            import reborn.fortran; reborn.fortran.omp_test_f.omp_test()

    Arguments:
        dat (|ndarray|): The image to analyze.
        mask_in (|ndarray|): The mask for the square central integration region.
        mask_out (|ndarray|): The mask for the square annulus integration region.
        n_in (int): Size of the central integration region; integrate from :math:`(-n_{in}, n_{in})`, inclusively.
        n_cent (int): Define the annulus integration region; we ignore the box from (-n_cent, n_cent), inclusively.
        n_out (int): Define the annulus integration region; we include the box from (-n_out, n_out), inclusively.

    Returns:
        (tuple):

        **snr** (|ndarray|): The SNR array.

        **signal** (|ndarray|): The signal array.
    """
    float_t = np.float64
    snr = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    signal = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    d = np.asfortranarray(dat.astype(float_t))
    m = np.asfortranarray(mask_in.astype(float_t))
    m2 = np.asfortranarray(mask_out.astype(float_t))
    peaks_f.boxsnr(d, m, m2, snr, signal, n_in, n_cent, n_out)
    return snr, signal


def snr_mask(dat, mask, nin, ncent, nout, threshold=6, mask_negative=True, max_iterations=3,
             pad_geometry=None, beam=None, subtract_median=False):
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
        pad_geometry (|PADGeometryList|) : PAD geometry (optional)
        subtract_median (bool) : Subtract median profiler before masking.

    Returns:
        numpy array : The mask with pixels above the SNR threshold
    """
    if subtract_median:
        if pad_geometry is None:
            raise ValueError('pad_geometry is None.  Cannot subtract median.')
        if beam is None:
            raise ValueError('beam is None.  Cannot subtract median.')
        profiler = RadialProfiler(pad_geometry=pad_geometry, beam=beam, mask=mask)
        dat = profiler.subtract_median_profile(dat)
    if isinstance(dat, list):
        return [snr_mask(d, m, nin, ncent, nout, threshold=threshold, mask_negative=mask_negative,
                         max_iterations=max_iterations, subtract_median=False) for (d, m) in zip(dat, mask)]
    mask_a = mask.copy()
    prev = 0
    for i in range(max_iterations):
        a, _ = boxsnr(dat, mask, mask_a, nin, ncent, nout)
        if mask_negative:
            a = np.abs(a)
        ab = a > threshold
        above = np.sum(ab)
        mask_a[ab] = 0
        if above == prev:
            break
        prev = above
    return mask_a
