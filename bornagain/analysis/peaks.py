from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


import numpy as np
# from time import time
from numpy.fft import fft2, ifft2, ifftshift
from scipy.signal import convolve2d
from bornagain.analysis.peaks_f import peak_snr_filter as peak_snr_filter_f

from numba import jit

# from skimage.morphology import disk
# from skimage.filters.rank import median as median_filter


# def annulus(inner, outer):
#
#     return disk(outer) - np.pad(disk(inner), outer-inner, mode='constant')


# class PeakFinderA(object):
#
#     def __init__(self, shape=None):
#
#         self.annulus = annulus(8, 12)
#         self.shape = shape
#
#     def median_filter(self, dat):
#
#         scl = np.max(np.abs(dat))
#
#         return median_filter(dat/scl, self.annulus)*scl


# class PeakFinderB(object):
#
#     def __init__(self, shape=None, radii=(4, 8, 12)):
#
#         self.tophat = disk(radii[0])
#         self.tophat = np.pad(self.tophat, radii[2] - radii[0], mode='constant')
#         self.n_tophat = np.sum(self.tophat)
#         self.annulus = annulus(radii[1], radii[2])
#         self.n_annulus = np.sum(self.annulus)
#         self.annulus = self.annulus / self.n_annulus
#         self.tophat_annulus = self.tophat - self.annulus * self.n_tophat
#         self.shape = shape
#
#     def get_snr(self, dat):
#
#         # t = time()
#         bak = convolve2d(dat, self.annulus, mode='full', boundary='symm')
#         bak2 = convolve2d(dat**2, self.annulus, mode='full', boundary='symm')
#         sigma = np.sqrt(bak2 - bak**2)
#
#         signal = convolve2d(dat, self.tophat_annulus, mode='full', boundary='symm')
#
#         snr = signal/sigma
#         snr[~np.isfinite(snr)] = 0
#
#         # print(bak2 - bak**2)
#         # print(time() - t)
#
#         return snr

from numba import jit
from multiprocessing import Pool

def snr_filter_pool(data):

    pool = Pool(6)
    return pool.map(snr_filter, data)



def snr_filter_numba(data, radii=(1, 18, 20), mask=None, local_max_only=1):

    data = data.astype(np.float32)
    a = int(radii[0])
    b = int(radii[1])
    c = int(radii[2])
    if mask is None:
        mask = np.ones_like(data)
    mask = mask.astype(np.int)
    local_max_only = int(local_max_only)

    return _snr_filter_numba(data, a, b, c, mask, local_max_only)

@jit(nopython=True)
def _snr_filter_numba(data, a, b, c, mask, local_max_only):

    nf = data.shape[1]
    ns = data.shape[0]

    snr = np.zeros_like(data)
    signal = np.zeros_like(data)

    for i in range(1, ns-1):
        for j in range(1, nf-1):

            # Skip masked pixels
            if mask[i, j] == 0:
                continue

            if local_max_only == 1:
                # Skip pixels that aren't local maxima
                this_val = data[i, j]
                if data[i-1, j] > this_val:
                    continue
                if data[i+1, j] > this_val:
                    continue
                if data[i, j-1] > this_val:
                    continue
                if data[i, j+1] > this_val:
                    continue
                if data[i-1, j-1] > this_val:
                    continue
                if data[i-1, j+1] > this_val:
                    continue
                if data[i+1, j-1] > this_val:
                    continue
                if data[i+1, j+1] > this_val:
                    continue

            # Now we will compute the locally integrated signal, and the locally integrated signal squared

            local_signal = 0
            local_signal2 = 0
            n_local = 0

            annulus_signal = 0
            annulus_signal2 = 0
            n_annulus = 0

            for q in range(-c, c+1):

                ii = i + q

                if ii < 0:
                    continue
                if ii >= ns:
                    continue

                q2 = q**2

                for r in range(-c, c+1):

                    jj = j+r

                    if jj < 0:
                        continue
                    if jj >= nf:
                        continue

                    if mask[ii, jj] == 0:
                        continue

                    rad = np.sqrt(q2 + r**2)

                    if rad <= a:

                        n_local += 1
                        local_signal += data[ii, jj]
                        local_signal2 += data[ii, jj]**2

                    if rad >= b and rad <= c:

                        n_annulus += 1
                        annulus_signal += data[ii, jj]
                        annulus_signal2 += data[ii, jj] ** 2

            if n_local == 0 or n_annulus == 0:
                continue

            # We subtract the local background from the signal
            signal[i, j] = local_signal/n_local - annulus_signal/n_annulus

            noise = np.sqrt(annulus_signal2/n_annulus - (annulus_signal/n_annulus)**2)
            snr[i, j] = signal[i, j]/noise

    return snr


def snr_filter_fortran(data, radii=(1, 18, 20), mask=None, local_max_only=1):

    data = data.copy('f')
    if mask is None:
        mask = np.ones_like(data, order='f')
    a = radii[0]
    b = radii[1]
    c = radii[2]
    snr = np.zeros_like(data, order='f')
    signal = np.zeros_like(data, order='f')
    peak_snr_filter_f(data, a, b, c, mask, local_max_only, snr, signal)

    return snr


class PeakFinderV1(object):

    def __init__(self, shape=None, radii=(1, 18, 20)):

        nx = shape[1]
        ny = shape[0]

        x = np.arange(-np.floor(nx / 2), np.ceil(nx / 2))
        y = np.arange(-np.floor(ny / 2), np.ceil(ny / 2))

        xx, yy = np.meshgrid(x, y)

        r = np.sqrt(xx ** 2 + yy ** 2)

        inner = np.zeros(shape)
        outer = np.zeros(shape)

        inner[r <= radii[0]] = 1
        outer[(r <= radii[2]) * (r > radii[1])] = 1

        inner = ifftshift(inner)
        outer = ifftshift(outer)

        n_inner = np.sum(inner)
        n_outer = np.sum(outer)

        inner_ft = fft2(inner)
        outer_ft = fft2(outer)

        self.inner = inner.astype(np.float32)
        self.outer = outer
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.inner_ft = inner_ft
        self.outer_ft = outer_ft
        self.inner_outer = self.inner - self.outer * self.n_inner / self.n_outer
        self.inner_outer_ft = self.inner_ft / self.n_inner - self.outer_ft / self.n_outer

    def get_signal_above_background(self, dat):

        return np.real(ifft2(fft2(dat)*(self.inner_ft / self.n_inner - self.outer_ft / self.n_outer)))

    def snr_filter(self, dat):

        # t = time()
        bak = np.real(ifft2(fft2(dat)*(self.outer_ft))) / self.n_outer
        bak2 = np.real(ifft2(fft2(dat**2)*(self.outer_ft))) / self.n_outer
        sigma = np.sqrt(bak2 - bak**2)

        signal = np.real(ifft2(fft2(dat)*(self.inner_outer_ft)))

        snr = signal/sigma

        snr[np.isinf(snr)] = 0
        snr[np.isnan(snr)] = 0

        return snr

    # def get_snr2(self, dat):
    #
    #     bak = convolve2d(dat, self.outer, mode='valid', boundary='symm')/self.n_outer
    #     bak2 = convolve2d(dat**2, self.outer, mode='valid', boundary='symm')/self.n_outer
    #     sigma = np.sqrt(bak2 - bak**2)
    #     signal = convolve2d(dat, self.inner_outer, mode='valid', boundary='symm')
    #     snr = signal/sigma
    #
    #     return snr

@jit(nopython=True)
def boxsnr(dat, mask, nin, ncent, nout):

    dtype = np.float64
    npx = dat.shape[1]
    npy = dat.shape[0]
    cumx = np.empty((npy, npx + 1), dtype=dtype)
    cum2x = np.empty((npy, npx + 1), dtype=dtype)
    cummx = np.empty((npy, npx + 1), dtype=dtype)
    sqix = np.empty((npy, npx), dtype=dtype)
    sq2ix = np.empty((npy, npx), dtype=dtype)
    sqmix = np.empty((npy, npx), dtype=dtype)
    sqcx = np.empty((npy, npx), dtype=dtype)
    sq2cx = np.empty((npy, npx), dtype=dtype)
    sqmcx = np.empty((npy, npx), dtype=dtype)
    sqox = np.empty((npy, npx), dtype=dtype)
    sq2ox = np.empty((npy, npx), dtype=dtype)
    sqmox = np.empty((npy, npx), dtype=dtype)
    cumiy = np.empty((npy + 1, npx), dtype=dtype)
    cum2iy = np.empty((npy + 1, npx), dtype=dtype)
    cummiy = np.empty((npy + 1, npx), dtype=dtype)
    cumcy = np.empty((npy + 1, npx), dtype=dtype)
    cum2cy = np.empty((npy + 1, npx), dtype=dtype)
    cummcy = np.empty((npy + 1, npx), dtype=dtype)
    cumoy = np.empty((npy + 1, npx), dtype=dtype)
    cum2oy = np.empty((npy + 1, npx), dtype=dtype)
    cummoy = np.empty((npy + 1, npx), dtype=dtype)

    cumx[:, 0] = 0
    cum2x[:, 0] = 0
    cummx[:, 0] = 0

    for ix in range(0, npx):
        cumx[:, ix+1] = cumx[:, ix-1] + dat[:, ix] * mask[:, ix]
        cum2x[:, ix+1] = cum2x[:, ix - 1] + dat[:, ix] * mask[:, ix]
        cummx[:, ix+1] = cummx[:, ix - 1] + mask[:, ix]

    for ix in range(0, npx):
        mn = min(npx, ix + nin)
        mx = max(0, ix - nin - 1)
        sqix[:, ix] = cumx[:, mn] - cumx[:, mx]
        sq2ix[:, ix] = cum2x[:, mn] - cum2x[:, mx]
        sqmix[:, ix] = cummx[:, mn] - cummx[:, mx]
        mn = min(npx, ix + ncent)
        mx = max(0, ix - ncent - 1)
        sqcx[:, ix] = cumx[:, mn] - cumx[:, mx]
        sq2cx[:, ix] = cum2x[:, mn] - cum2x[:, mx]
        sqmcx[:, ix] = cummx[:, mn] - cummx[:, mx]
        mn = min(npx, ix + nout)
        mx = max(0, ix - nout - 1)
        sqox[:, ix] = cumx[:, mn] - cumx[:, mx]
        sq2ox[:, ix] = cum2x[:, mn] - cum2x[:, mx]
        sqmox[:, ix] = cummx[:, mn] - cummx[:, mx]

    cumiy[0, :] = 0
    cum2iy[0, :] = 0
    cummiy[0, :] = 0
    cumcy[0, :] = 0
    cum2cy[0, :] = 0
    cummcy[0, :] = 0
    cumoy[0, :] = 0
    cum2oy[0, :] = 0
    cummoy[0, :] = 0

    for iy in range(1, npy + 1):
        cumiy[iy, :] = cumiy[iy-1, :] + sqix[iy, :]
        cum2iy[iy, :] = cum2iy[iy-1, :] + sq2ix[iy, :]
        cummiy[iy, :] = cummiy[iy-1, :] + sqmix[iy, :]
        cumcy[iy, :] = cumcy[iy-1, :] + sqcx[iy, :]
        cum2cy[iy, :] = cum2cy[iy-1, :] + sq2cx[iy, :]
        cummcy[iy, :] = cummcy[iy-1, :] + sqmcx[iy, :]
        cumoy[iy, :] = cumoy[iy-1, :] + sqox[iy, :]
        cum2oy[iy, :] = cum2oy[iy-1, :] + sq2ox[iy, :]
        cummoy[iy, :] = cummoy[iy-1, :] + sqmox[iy, :]

    for iy in range(0, npy):
        mn = min(npy, iy + nin)
        mx = max(0, iy - nin - 1)
        sqix[iy, :] = cumiy[mn, :] - cumiy[mx, :]
        sq2ix[iy, :] = cum2iy[mn, :] - cum2iy[mx, :]
        sqmix[iy, :] = cummiy[mn, :] - cummiy[mx, :]
        mn = min(npy, iy + ncent)
        mx = max(0, iy - ncent - 1)
        sqcx[iy, :] = cumcy[mn, :] - cumcy[mx, :]
        sq2cx[iy, :] = cum2cy[mn, :] - cum2cy[mx, :]
        sqmcx[iy, :] = cummcy[mn, :] - cummcy[mx, :]
        mn = min(npy, iy + nout)
        mx = max(0, iy - nout - 1)
        sqox[iy, :] = cumoy[mn, :] - cumoy[mx, :]
        sq2ox[iy, :] = cum2oy[mn, :] - cum2oy[mx, :]
        sqmox[iy, :] = cummoy[mn, :] - cummoy[mx, :]

    small = 1.0e-15
    sqox = sqox - sqcx
    sq2ox = sq2ox - sq2cx
    sqmox = sqmox - sqmcx + small
    sqmix = sqmix + small
    return (sqix - sqox * sqmix / sqmox) / (np.sqrt(sqmix) * (np.sqrt(sq2ox / sqmox - (sqox / sqmox) ** 2) + small))