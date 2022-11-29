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

import time
import numpy as np
from scipy.signal import convolve, find_peaks
from .. import misc
from ..fortran import peaks_f
from ..detector import RadialProfiler, PADGeometryList
from ..source import Beam


def snr_filter_test(data, mask, mask2, nin, ncent, nout):
    r""" This is a slow version of snr_filter, which is about 10-fold faster on multi-core CPUs.  """
    kin = np.ones((2*nin+1, 2*nin+1))
    kout = np.ones((2*nout+1, 2*nout+1))
    d = (nout-ncent)
    kout[d:-d, d:-d] = 0
    cin = convolve(mask, kin, mode='same')
    cout = convolve(mask2, kout, mode='same')
    bak = convolve(data*mask2, kout, mode='same') / cout
    sig = convolve(data*mask, kin, mode='same') / cin - bak
    bak2 = convolve(data**2*mask2, kout, mode='same') / cout
    stderr = np.sqrt((bak2 - bak**2) / cin)
    snr = sig / stderr
    return snr, sig


def snr_filter(dat, mask_in, mask_out, n_in, n_cent, n_out):

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
    # snr = np.asfortranarray(np.empty(dat.shape, dtype=float_t))
    # signal = np.asfortranarray(np.empty(dat.shape, dtype=float_t))
    snr = np.empty(dat.shape, dtype=float_t)
    signal = np.empty(dat.shape, dtype=float_t)
    dat = np.asfortranarray(dat.astype(float_t))
    mask_in = np.asfortranarray(mask_in.astype(float_t))
    mask_out = np.asfortranarray(mask_out.astype(float_t))
    peaks_f.boxsnr(dat.T, mask_in.T, mask_out.T, snr.T, signal.T, n_in, n_cent, n_out)
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
    if isinstance(dat, list):  # Recursive function calls in the case of a list
        zipped = [snr_mask(d, m, nin, ncent, nout, threshold=threshold, mask_negative=mask_negative,
                         max_iterations=max_iterations, subtract_median=False) for (d, m) in zip(dat, mask)]
        m = []
        d = []
        for i in range(len(zipped)):
            m.append(zipped[i][0])
            d.append(zipped[i][1])
        return m, d
    amask = mask.copy()
    prev = 0
    d = dat.copy()
    for i in range(max_iterations):
        # print('iteration', i)
        # t = time.time()
        # snr, sig = boxsnr(d, mask, amask, nin, ncent, nout)
        # print(time.time()-t)
        # t = time.time()
        snr, sig = snr_filter(d, mask, amask, nin, ncent, nout)
        # print(time.time()-t)
        if mask_negative:
            snr = np.abs(snr)
        ab = snr > threshold
        # above = np.sum(ab)
        amask[ab] = 0
        # if above == prev:
        #     break
        # prev = above
    return amask, snr


class StreakMasker:

    def __init__(self, geom: PADGeometryList, beam: Beam, n_q=100, q_range=(0, 2e10), prominence=0.8, max_streaks=2,
                 debug=1):
        r"""
        A tool for masking jet streaks or other streak-like features in diffraction patterns.  It is assumed that
        the streak crosses through the beam center.

        Arguments:
            geom (|PADGeometryList|): PAD geometry.
            beam (|Beam|): Beam info.
            n_q (int): Number of q bins.  Default: 100.
            q_range (float tuple): Range of q values to consider. Default: (0, 2e10).
            prominence (float): Look at the corresponding parameter in scipy.signal.find_peaks.  Default: 0.8.
            max_streaks (int): Maximum number of streaks.  Default: 2.
        """
        self.debug = debug
        self.dbgmsg('Initializing')
        self.prominence = prominence
        self.max_streaks = max_streaks
        self.n_p = 360
        d_p = 2*np.pi/self.n_p
        self.p_r = (d_p/2, 2*np.pi-d_p/2)
        self.phi = np.linspace(self.p_r[0], self.p_r[1], self.n_p)
        self.q_r = q_range
        self.q = geom.q_mags(beam=beam)
        self.p = geom.azimuthal_angles(beam=beam)
        self.beam = beam
        self.n_q = n_q
        self.geom = geom

    def dbgmsg(self, *args, **kwargs):
        if self.debug:
            print('DEBUG:StreakMasker:', *args, **kwargs)

    def get_mask(self, pattern, mask=None):
        r""" Find streaks and return a mask.

        Arguments:
            pattern (|ndarray|): Diffraction intensities.
            mask (|ndarray|): Diffraction intensity mask.  Zero means ignore.

        Returns: |ndarray|"""
        if mask is None:
            mask = self.geom.ones()
        pattern = self.geom.concat_data(pattern)
        mask = self.geom.concat_data(mask)
        stats = misc.polar.get_polar_stats(pattern.astype(np.float64), self.q, self.p,
            weights=mask.astype(np.float64), n_q_bins=self.n_q, q_min=self.q_r[0], q_max=self.q_r[1], n_p_bins=360,
            p_min=0, p_max=6.283185307179586)
        smask = self.geom.ones()
        polar = stats['mean']
        pmask = stats['weight_sum']
        pmask[pmask != 0] = 1
        polar = polar[:, 0:180] + polar[:, 180:360]
        pmask = pmask[:, 0:180] + pmask[:, 180:360]
        polar = np.divide(polar, pmask, out=np.zeros_like(polar), where=pmask != 0)
        pmask[pmask != 0] = 1
        polar *= pmask
        m = np.sum(pmask, 1)
        p = np.sum(polar, 1)
        pt = polar.T
        pt -= np.divide(p, m, out=np.zeros_like(p), where=m > 0)
        polar *= pmask
        proj = np.sum(polar, axis=0)
        m = np.sum(pmask, axis=0)
        proj = np.divide(proj, m, out=np.zeros_like(proj), where=m > 0)
        peaks = find_peaks(np.concatenate([proj, proj]), prominence=self.prominence)
        self.dbgmsg(f"Result from scipy.signal.find_peaks:", peaks)
        if len(peaks[0]) == 0:
            return smask
        peak_angles, indices = np.unique(peaks[0] % 180, return_index=True)
        peak_prominences = peaks[1]['prominences'][indices]
        s = np.argsort(peak_prominences)[::-1]
        peak_prominences = peak_prominences[s]
        peak_angles = peak_angles[s]
        c = 0
        for (angle, prominence) in zip(peak_angles, peak_prominences):
            c += 1
            self.dbgmsg(f'Masking streak at {angle} degrees (prominence = {prominence})')
            if c > self.max_streaks:
                break
            angle = angle*np.pi/180 + np.pi/2
            streak_vec = self.beam.e1_vec * np.cos(angle) + self.beam.e2_vec * np.sin(angle)
            smask = smask * self.geom.streak_mask(vec=streak_vec, angle=0.01)
        return smask