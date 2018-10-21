from __future__ import (absolute_import, division, print_function, unicode_literals)


import numpy as np
from time import time
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d


from skimage.morphology import disk
from skimage.filters.rank import median as median_filter


def annulus(inner, outer):

    return disk(outer) - np.pad(disk(inner), outer-inner, mode='constant')


class PeakFinderA(object):

    def __init__(self, shape=None):

        self.annulus = annulus(8, 12)
        self.shape = shape

    def median_filter(self, dat):

        scl = np.max(np.abs(dat))

        return median_filter(dat/scl, self.annulus)*scl


class PeakFinderB(object):

    def __init__(self, shape=None, radii=(4, 8, 12)):

        self.tophat = disk(radii[0])
        self.tophat = np.pad(self.tophat, radii[2] - radii[0], mode='constant')
        self.n_tophat = np.sum(self.tophat)
        self.annulus = annulus(radii[1], radii[2])
        self.n_annulus = np.sum(self.annulus)
        self.annulus = self.annulus / self.n_annulus
        self.tophat_annulus = self.tophat - self.annulus * self.n_tophat
        self.shape = shape

    def get_snr(self, dat):

        # t = time()
        bak = convolve2d(dat, self.annulus, mode='full', boundary='symm')
        bak2 = convolve2d(dat**2, self.annulus, mode='full', boundary='symm')
        sigma = np.sqrt(bak2 - bak**2)

        signal = convolve2d(dat, self.tophat_annulus, mode='full', boundary='symm')

        snr = signal/sigma
        snr[~np.isfinite(snr)] = 0

        # print(bak2 - bak**2)
        # print(time() - t)

        return snr









class PeakFinderV1(object):

    def __init__(self, shape=None, radii=None):

        if radii is None:
            radii = (1, 4, 7)

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

        self.inner = inner
        self.outer = outer
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.inner_ft = inner_ft
        self.outer_ft = outer_ft
        self.inner_outer = self.inner - self.outer * self.n_inner / self.n_outer
        self.inner_outer_ft = self.inner_ft / self.n_inner - self.outer_ft / self.n_outer

    def get_signal_above_background(self, dat):

        return np.real(ifft2(fft2(dat)*(self.inner_ft / self.n_inner - self.outer_ft / self.n_outer)))

    def get_snr(self, dat):

        # t = time()
        bak = np.real(ifft2(fft2(dat)*(self.outer_ft))) / self.n_outer
        bak2 = np.real(ifft2(fft2(dat**2)*(self.outer_ft))) / self.n_outer
        sigma = np.sqrt(bak2 - bak**2)

        signal = np.real(ifft2(fft2(dat)*(self.inner_outer_ft)))

        snr = signal/sigma

        snr[np.isinf(snr)] = 0
        snr[np.isnan(snr)] = 0

        # print(time() - t)

        return snr

    def get_snr2(self, dat):

        # t = time()
        bak = convolve2d(dat, self.outer, mode='valid', boundary='symm')/self.n_outer
        bak2 = convolve2d(dat**2, self.outer, mode='valid', boundary='symm')/self.n_outer
        sigma = np.sqrt(bak2 - bak**2)

        signal = convolve2d(dat, self.inner_outer, mode='valid', boundary='symm')

        snr = signal/sigma
        # print(np.max(signal), np.max(bak), np.max(bak2), np.max(snr), np.min(snr))

        # print(bak2 - bak**2)
        # print(time() - t)

        return snr

