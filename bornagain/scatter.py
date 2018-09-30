r"""
Classes related to x-ray "scattering", which loosely means diffraction from many objects in random orientations.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import numpy as np


class RadialProfile(object):
    r"""
    Helper class to create radial profiles.
    """

    def __init__(self):

        self.n_bins = None
        self.bins = None
        self.bin_size = None
        self.bin_indices = None
        self.q_mags = None
        self.mask = None
        self.counts = None
        self.q_range = None
        self.counts_non_zero = None

    def make_plan(self, q_mags, mask=None, n_bins=100, q_range=None):
        r"""
        Setup the binning indices for the creation of radial profiles.

        Arguments:
            q_mags (numpy array) :
                Scattering vector magnitudes.
            mask (numpy array) :
                Pixel mask.  Should be ones and zeros, where one means "good" and zero means "bad".
            n_bins (int) :
                Number of bins.
            q_range (list-like) :
                The minimum and maximum of the scattering vector magnitudes.  The bin size will be equal to
                (max_q - min_q) / n_bins
        """

        q_mags = q_mags.ravel()

        if q_range is None:
            min_q = np.min(q_mags)
            max_q = np.max(q_mags)
            q_range = np.array([min_q, max_q])
        else:
            q_range = q_range.copy()
            min_q = q_range[0]
            max_q = q_range[1]

        bin_size = (max_q - min_q) / float(n_bins)
        bins = (np.arange(0, n_bins) + 0.5) * bin_size + min_q
        bin_indices = np.int64(np.floor((q_mags - min_q) / bin_size))
        bin_indices[bin_indices < 0] = 0
        bin_indices[bin_indices >= n_bins] = n_bins - 1
        if mask is None:
            mask = np.ones([len(bin_indices)])
        else:
            mask = mask.copy().ravel()
        # print(bin_indices.shape, mask.shape, n_bins)
        counts = np.bincount(bin_indices, mask, n_bins)
        counts_non_zero = counts > 0

        self.n_bins = n_bins
        self.bins = bins
        self.bin_size = bin_size
        self.bin_indices = bin_indices
        self.q_mags = q_mags
        self.mask = mask
        self.counts = counts
        self.q_range = q_range
        self.counts_non_zero = counts_non_zero

    def get_profile(self, data, average=True):
        r"""
        Create a radial profile for a particular dataframe.

        Arguments:
            data (numpy array) :
                Intensity data.
            average (bool) :
                If true, divide the sum in each bin by the counts, else return the sum.  Default: True.

        Returns:
            profile (numpy array) :
                The requested radial profile.
        """

        profile = np.bincount(self.bin_indices, data.ravel() * self.mask, self.n_bins)
        if average:
            profile.flat[self.counts_non_zero] /= self.counts.flat[self.counts_non_zero]

        return profile
