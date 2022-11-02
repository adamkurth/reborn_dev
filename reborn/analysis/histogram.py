import numpy as np


class PixelHistogram:
    _bin_min = None
    _bin_max = None
    _bin_delta = None
    _n_pixel_bins = None
    _n_detector_pixels = None
    _detector_pixels_index = None
    _pixel_histogram = None

    def __init__(self, bin_min, bin_max, n_pixel_bins, n_detector_pixels):
        self._bin_min = bin_min
        self._bin_max = bin_max
        self._n_pixel_bins = n_pixel_bins
        self._bin_delta = (self._bin_max - self._bin_min) / (self._n_pixel_bins - 1)
        self._n_detector_pixels = n_detector_pixels
        self._detector_pixels_index = np.arange(self._n_detector_pixels, dtype=int)
        self._pixel_histogram = np.zeros((self._n_pixel_bins, self._n_detector_pixels), dtype=int)

    @property
    def bin_min(self):
        return self._bin_min

    @property
    def bin_max(self):
        return self._bin_max

    @property
    def bin_delta(self):
        return self._bin_delta

    @property
    def n_pixel_bins(self):
        return self._n_pixel_bins

    @property
    def n_detector_pixels(self):
        return self._n_detector_pixels

    @property
    def pixel_histogram(self):
        return self._pixel_histogram

    def add_frame(self, dataframe):
        data = dataframe.get_raw_data_flat()
        mask = dataframe.get_mask_flat()
        data[mask == 0] = 0
        # bin data
        bin_index = np.floor((data - self._bin_min) / self._bin_delta).astype(int)
        idx = np.ravel_multi_index((bin_index, self._detector_pixels_index),
                                   (self._n_pixel_bins, self._n_detector_pixels),
                                   mode='clip')
        self._pixel_histogram.flat[idx] += 1

