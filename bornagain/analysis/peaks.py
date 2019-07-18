from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


import numpy as np
from scipy.ndimage import measurements
# from numba import jit
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

        if peaks_f is not None:
            self.snr_transform = boxsnr_fortran
        else:
            self.snr_transform = boxsnr_numba

        self.snr_threshold = snr_threshold
        self.radii = radii
        self.mask = mask

    def find_peaks(self, data, mask=None):

        if mask is None:
            if self.mask is None:
                self.mask = np.ones_like(data)
            mask = self.mask

        self.snr, self.signal = self.snr_transform(data, mask, self.radii[0], self.radii[1], self.radii[2])
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


def boxsnr(dat, mask, nin, ncent, nout):
    # if peaks_f is not None:
    snr, signal = boxsnr_fortran(dat, mask, nin, ncent, nout)
    # else:
    #     snr, signal = boxsnr_numba(dat, mask, nin, ncent, nout)
    return snr, signal


def boxsnr_fortran(dat, mask, nin, ncent, nout):

    float_t = np.float64
    snr = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    signal = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    peaks_f.peaker.boxsnr(np.asfortranarray(dat.astype(float_t)), np.asfortranarray(mask.astype(float_t)), snr, signal, nin, ncent, nout)
    return snr, signal


def boxsnr2(dat, mask, nin, ncent, nout):
    # if peaks_f is not None:
    snr, signal = boxsnr2_fortran(dat, mask, nin, ncent, nout)
    # else:
    #     snr, signal = boxsnr_numba(dat, mask, nin, ncent, nout)
    return snr, signal


def boxsnr2_fortran(dat, mask, nin, ncent, nout):

    float_t = np.float64
    snr = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    signal = np.asfortranarray(np.ones(dat.shape, dtype=float_t))
    peaks_f.boxsnr2(np.asfortranarray(dat.astype(float_t)), np.asfortranarray(mask.astype(float_t)), snr, signal, nin, ncent, nout)
    return snr, signal


# @jit(nopython=True)
# def boxsnr_numba(dat, mask, nin, ncent, nout):
#
#     dtype = np.float64
#     npx = dat.shape[1]
#     npy = dat.shape[0]
#     cumx = np.empty((npy, npx + 1), dtype=dtype)
#     cum2x = np.empty((npy, npx + 1), dtype=dtype)
#     cummx = np.empty((npy, npx + 1), dtype=dtype)
#     sqix = np.empty((npy, npx), dtype=dtype)
#     sqmix = np.empty((npy, npx), dtype=dtype)
#     sqcx = np.empty((npy, npx), dtype=dtype)
#     sq2cx = np.empty((npy, npx), dtype=dtype)
#     sqmcx = np.empty((npy, npx), dtype=dtype)
#     sqox = np.empty((npy, npx), dtype=dtype)
#     sq2ox = np.empty((npy, npx), dtype=dtype)
#     sqmox = np.empty((npy, npx), dtype=dtype)
#     cumiy = np.empty((npy + 1, npx), dtype=dtype)
#     cummiy = np.empty((npy + 1, npx), dtype=dtype)
#     cumcy = np.empty((npy + 1, npx), dtype=dtype)
#     cum2cy = np.empty((npy + 1, npx), dtype=dtype)
#     cummcy = np.empty((npy + 1, npx), dtype=dtype)
#     cumoy = np.empty((npy + 1, npx), dtype=dtype)
#     cum2oy = np.empty((npy + 1, npx), dtype=dtype)
#     cummoy = np.empty((npy + 1, npx), dtype=dtype)
#
#     cumx[:, 0] = 0.
#     cum2x[:, 0] = 0.
#     cummx[:, 0] = 0.
#
#     for i in range(0, npx):
#         cumx[:, i+1] = cumx[:, i] + dat[:, i] * mask[:, i]
#         cum2x[:, i+1] = cum2x[:, i] + dat[:, i]**2 * mask[:, i]
#         cummx[:, i+1] = cummx[:, i] + mask[:, i]
#
#     a = 1
#
#     for i in range(0, npx):
#         mn = min(npx, i+nin+a)
#         mx = max(0, i-nin-1+a)
#         sqix[:, i] = cumx[:, mn] - cumx[:, mx]
#         sqmix[:, i] = cummx[:, mn] - cummx[:, mx]
#         mn = min(npx, i+ncent+a)
#         mx = max(0, i-ncent-1+a)
#         sqcx[:, i] = cumx[:, mn] - cumx[:, mx]
#         sq2cx[:, i] = cum2x[:, mn] - cum2x[:, mx]
#         sqmcx[:, i] = cummx[:, mn] - cummx[:, mx]
#         mn = min(npx, i+nout+a)
#         mx = max(0, i-nout-1+a)
#         sqox[:, i] = cumx[:, mn] - cumx[:, mx]
#         sq2ox[:, i] = cum2x[:, mn] - cum2x[:, mx]
#         sqmox[:, i] = cummx[:, mn] - cummx[:, mx]
#
#     cumiy[0, :] = 0
#     cummiy[0, :] = 0
#     cumcy[0, :] = 0
#     cum2cy[0, :] = 0
#     cummcy[0, :] = 0
#     cumoy[0, :] = 0
#     cum2oy[0, :] = 0
#     cummoy[0, :] = 0
#
#     for i in range(0, npy):
#         cumiy[i+1, :] = cumiy[i, :] + sqix[i, :]
#         cummiy[i+1, :] = cummiy[i, :] + sqmix[i, :]
#         cumcy[i+1, :] = cumcy[i, :] + sqcx[i, :]
#         cum2cy[i+1, :] = cum2cy[i, :] + sq2cx[i, :]
#         cummcy[i+1, :] = cummcy[i, :] + sqmcx[i, :]
#         cumoy[i+1, :] = cumoy[i, :] + sqox[i, :]
#         cum2oy[i+1, :] = cum2oy[i, :] + sq2ox[i, :]
#         cummoy[i+1, :] = cummoy[i, :] + sqmox[i, :]
#
#     for i in range(0, npy):
#         mn = min(npy, i+nin+a)
#         mx = max(0, i-nin-1+a)
#         sqix[i, :] = cumiy[mn, :] - cumiy[mx, :]
#         sqmix[i, :] = cummiy[mn, :] - cummiy[mx, :]
#         mn = min(npy, i+ncent+a)
#         mx = max(0, i-ncent-1+a)
#         sqcx[i, :] = cumcy[mn, :] - cumcy[mx, :]
#         sq2cx[i, :] = cum2cy[mn, :] - cum2cy[mx, :]
#         sqmcx[i, :] = cummcy[mn, :] - cummcy[mx, :]
#         mn = min(npy, i+nout+a)
#         mx = max(0, i-nout-1+a)
#         sqox[i, :] = cumoy[mn, :] - cumoy[mx, :]
#         sq2ox[i, :] = cum2oy[mn, :] - cum2oy[mx, :]
#         sqmox[i, :] = cummoy[mn, :] - cummoy[mx, :]
#
#     small = 1.0e-15
#     sqox = sqox - sqcx
#     sq2ox = sq2ox - sq2cx
#     sqmox = sqmox - sqmcx + small
#     sqmix = sqmix + small
#
#     signal = sqix - sqox * sqmix / sqmox
#     snr = signal / (np.sqrt(sqmix) * (np.sqrt(sq2ox / sqmox - (sqox / sqmox) ** 2) + small))
#
#     return snr, signal
