from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


import numpy as np
# from numpy.fft import fft2, ifft2, ifftshift
from bornagain.analysis import peaks_f
from numba import jit
# from multiprocessing import Pool


def boxsnr_fortran(dat, mask, nin, ncent, nout):

    out = np.ones_like(dat)
    out = np.asfortranarray(out)
    peaks_f.peaker.boxsnr(np.asfortranarray(dat), np.asfortranarray(mask), out, nin, ncent, nout)
    return out


@jit(nopython=True)
def boxsnr_numba(dat, mask, nin, ncent, nout):

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
    # cum2iy = np.empty((npy + 1, npx), dtype=dtype)
    cummiy = np.empty((npy + 1, npx), dtype=dtype)
    cumcy = np.empty((npy + 1, npx), dtype=dtype)
    cum2cy = np.empty((npy + 1, npx), dtype=dtype)
    cummcy = np.empty((npy + 1, npx), dtype=dtype)
    cumoy = np.empty((npy + 1, npx), dtype=dtype)
    cum2oy = np.empty((npy + 1, npx), dtype=dtype)
    cummoy = np.empty((npy + 1, npx), dtype=dtype)

    cumx[:, 0] = 0.
    cum2x[:, 0] = 0.
    cummx[:, 0] = 0.

    for i in range(0, npx):
        cumx[:, i+1] = cumx[:, i] + dat[:, i] * mask[:, i]
        cum2x[:, i+1] = cum2x[:, i] + dat[:, i]**2 * mask[:, i]
        cummx[:, i+1] = cummx[:, i] + mask[:, i]

    # return cumx

    a = 1

    for i in range(0, npx):
        mn = min(npx, i+nin+a)
        mx = max(0, i-nin-1+a)
        sqix[:, i] = cumx[:, mn] - cumx[:, mx]
        # sq2ix[:, i] = cum2x[:, mn] - cum2x[:, mx]
        sqmix[:, i] = cummx[:, mn] - cummx[:, mx]
        mn = min(npx, i+ncent+a)
        mx = max(0, i-ncent-1+a)
        sqcx[:, i] = cumx[:, mn] - cumx[:, mx]
        sq2cx[:, i] = cum2x[:, mn] - cum2x[:, mx]
        sqmcx[:, i] = cummx[:, mn] - cummx[:, mx]
        mn = min(npx, i+nout+a)
        mx = max(0, i-nout-1+a)
        sqox[:, i] = cumx[:, mn] - cumx[:, mx]
        sq2ox[:, i] = cum2x[:, mn] - cum2x[:, mx]
        sqmox[:, i] = cummx[:, mn] - cummx[:, mx]

    # return sqix

    cumiy[0, :] = 0
    # cum2iy[0, :] = 0
    cummiy[0, :] = 0
    cumcy[0, :] = 0
    cum2cy[0, :] = 0
    cummcy[0, :] = 0
    cumoy[0, :] = 0
    cum2oy[0, :] = 0
    cummoy[0, :] = 0

    for i in range(0, npy):
        cumiy[i+1, :] = cumiy[i, :] + sqix[i, :]
        # cum2iy[i+1, :] = cum2iy[i-1, :] + sq2ix[i, :]
        cummiy[i+1, :] = cummiy[i, :] + sqmix[i, :]
        cumcy[i+1, :] = cumcy[i, :] + sqcx[i, :]
        cum2cy[i+1, :] = cum2cy[i, :] + sq2cx[i, :]
        cummcy[i+1, :] = cummcy[i, :] + sqmcx[i, :]
        cumoy[i+1, :] = cumoy[i, :] + sqox[i, :]
        cum2oy[i+1, :] = cum2oy[i, :] + sq2ox[i, :]
        cummoy[i+1, :] = cummoy[i, :] + sqmox[i, :]

    # return cumiy

    for i in range(0, npy):
        mn = min(npy, i+nin+a)
        mx = max(0, i-nin-1+a)
        sqix[i, :] = cumiy[mn, :] - cumiy[mx, :]
        # sq2ix[i, :] = cum2iy[mn, :] - cum2iy[mx, :]
        sqmix[i, :] = cummiy[mn, :] - cummiy[mx, :]
        mn = min(npy, i+ncent+a)
        mx = max(0, i-ncent-1+a)
        sqcx[i, :] = cumcy[mn, :] - cumcy[mx, :]
        sq2cx[i, :] = cum2cy[mn, :] - cum2cy[mx, :]
        sqmcx[i, :] = cummcy[mn, :] - cummcy[mx, :]
        mn = min(npy, i+nout+a)
        mx = max(0, i-nout-1+a)
        sqox[i, :] = cumoy[mn, :] - cumoy[mx, :]
        sq2ox[i, :] = cum2oy[mn, :] - cum2oy[mx, :]
        sqmox[i, :] = cummoy[mn, :] - cummoy[mx, :]

    # return sqix

    small = 1.0e-15
    sqox = sqox - sqcx
    sq2ox = sq2ox - sq2cx
    sqmox = sqmox - sqmcx + small
    sqmix = sqmix + small
    return (sqix - sqox * sqmix / sqmox) / (np.sqrt(sqmix) * (np.sqrt(sq2ox / sqmox - (sqox / sqmox) ** 2) + small))

# def snr_filter_pool(data):
#
#     pool = Pool(6)
#     return pool.map(snr_filter, data)


# def snr_filter_numba(data, radii=(1, 18, 20), mask=None, local_max_only=1):
#
#     data = data.astype(np.float32)
#     a = int(radii[0])
#     b = int(radii[1])
#     c = int(radii[2])
#     if mask is None:
#         mask = np.ones_like(data)
#     mask = mask.astype(np.int)
#     local_max_only = int(local_max_only)
#
#     return _snr_filter_numba(data, a, b, c, mask, local_max_only)


# This works but commented out because simplicity is preferred (i.e. just one peak finder)
# @jit(nopython=True)
# def _snr_filter_numba(data, a, b, c, mask, local_max_only):
#
#     nf = data.shape[1]
#     ns = data.shape[0]
#
#     snr = np.zeros_like(data)
#     signal = np.zeros_like(data)
#
#     for i in range(1, ns-1):
#         for j in range(1, nf-1):
#
#             # Skip masked pixels
#             if mask[i, j] == 0:
#                 continue
#
#             if local_max_only == 1:
#                 # Skip pixels that aren't local maxima
#                 this_val = data[i, j]
#                 if data[i-1, j] > this_val:
#                     continue
#                 if data[i+1, j] > this_val:
#                     continue
#                 if data[i, j-1] > this_val:
#                     continue
#                 if data[i, j+1] > this_val:
#                     continue
#                 if data[i-1, j-1] > this_val:
#                     continue
#                 if data[i-1, j+1] > this_val:
#                     continue
#                 if data[i+1, j-1] > this_val:
#                     continue
#                 if data[i+1, j+1] > this_val:
#                     continue
#
#             # Now we will compute the locally integrated signal, and the locally integrated signal squared
#
#             local_signal = 0
#             local_signal2 = 0
#             n_local = 0
#
#             annulus_signal = 0
#             annulus_signal2 = 0
#             n_annulus = 0
#
#             for q in range(-c, c+1):
#
#                 ii = i + q
#
#                 if ii < 0:
#                     continue
#                 if ii >= ns:
#                     continue
#
#                 q2 = q**2
#
#                 for r in range(-c, c+1):
#
#                     jj = j+r
#
#                     if jj < 0:
#                         continue
#                     if jj >= nf:
#                         continue
#
#                     if mask[ii, jj] == 0:
#                         continue
#
#                     rad = np.sqrt(q2 + r**2)
#
#                     if rad <= a:
#
#                         n_local += 1
#                         local_signal += data[ii, jj]
#                         local_signal2 += data[ii, jj]**2
#
#                     if rad >= b and rad <= c:
#
#                         n_annulus += 1
#                         annulus_signal += data[ii, jj]
#                         annulus_signal2 += data[ii, jj] ** 2
#
#             if n_local == 0 or n_annulus == 0:
#                 continue
#
#             # We subtract the local background from the signal
#             signal[i, j] = local_signal/n_local - annulus_signal/n_annulus
#
#             noise = np.sqrt(annulus_signal2/n_annulus - (annulus_signal/n_annulus)**2)
#             snr[i, j] = signal[i, j]/noise
#
#     return snr


# def snr_filter_fortran(data, radii=(1, 18, 20), mask=None, local_max_only=1):
#
#     data = data.copy('f')
#     if mask is None:
#         mask = np.ones_like(data, order='f')
#     a = radii[0]
#     b = radii[1]
#     c = radii[2]
#     snr = np.zeros_like(data, order='f')
#     signal = np.zeros_like(data, order='f')
#     peak_snr_filter_f(data, a, b, c, mask, local_max_only, snr, signal)
#
#     return snr


# class PeakFinderV1(object):
#
#     def __init__(self, shape=None, radii=(1, 18, 20)):
#
#         nx = shape[1]
#         ny = shape[0]
#
#         x = np.arange(-np.floor(nx / 2), np.ceil(nx / 2))
#         y = np.arange(-np.floor(ny / 2), np.ceil(ny / 2))
#
#         xx, yy = np.meshgrid(x, y)
#
#         r = np.sqrt(xx ** 2 + yy ** 2)
#
#         inner = np.zeros(shape)
#         outer = np.zeros(shape)
#
#         inner[r <= radii[0]] = 1
#         outer[(r <= radii[2]) * (r > radii[1])] = 1
#
#         inner = ifftshift(inner)
#         outer = ifftshift(outer)
#
#         n_inner = np.sum(inner)
#         n_outer = np.sum(outer)
#
#         inner_ft = fft2(inner)
#         outer_ft = fft2(outer)
#
#         self.inner = inner.astype(np.float32)
#         self.outer = outer
#         self.n_inner = n_inner
#         self.n_outer = n_outer
#         self.inner_ft = inner_ft
#         self.outer_ft = outer_ft
#         self.inner_outer = self.inner - self.outer * self.n_inner / self.n_outer
#         self.inner_outer_ft = self.inner_ft / self.n_inner - self.outer_ft / self.n_outer
#
#     def get_signal_above_background(self, dat):
#
#         return np.real(ifft2(fft2(dat)*(self.inner_ft / self.n_inner - self.outer_ft / self.n_outer)))
#
#     def snr_filter(self, dat):
#
#         # t = time()
#         bak = np.real(ifft2(fft2(dat)*(self.outer_ft))) / self.n_outer
#         bak2 = np.real(ifft2(fft2(dat**2)*(self.outer_ft))) / self.n_outer
#         sigma = np.sqrt(bak2 - bak**2)
#
#         signal = np.real(ifft2(fft2(dat)*(self.inner_outer_ft)))
#
#         snr = signal/sigma
#
#         snr[np.isinf(snr)] = 0
#         snr[np.isnan(snr)] = 0
#
#         return snr
#
#     def get_snr2(self, dat):
#
#         bak = convolve2d(dat, self.outer, mode='valid', boundary='symm')/self.n_outer
#         bak2 = convolve2d(dat**2, self.outer, mode='valid', boundary='symm')/self.n_outer
#         sigma = np.sqrt(bak2 - bak**2)
#         signal = convolve2d(dat, self.inner_outer, mode='valid', boundary='symm')
#         snr = signal/sigma
#
#         return snr
