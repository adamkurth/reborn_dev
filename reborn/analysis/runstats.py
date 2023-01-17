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

r""" Utilities for gathering statistics from data runs. """
import numpy as np
import pyqtgraph as pg
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None
from .parallel import get_setup_data
from .parallel import ParallelAnalyzer
from .. import detector
from .. import fileio
from ..dataframe import DataFrame
from ..external.pyqtgraph import imview
from ..fileio.getters import ListFrameGetter
from ..fortran import scatter_f
from ..viewers.qtviews.padviews import PADView


class PixelHistogram:

    count = 0  #: Number of frames contributing to the histogram.
    bin_min = None  #: The minimum value corresponding to histogram bin *centers*.
    bin_max = None  #: The maximum value corresponding to histogram bin *centers*.
    n_bins = None  #: The number of histogram bins.

    def __init__(self, bin_min=None, bin_max=None, n_bins=None, n_pixels=None, zero_photon_peak=0, one_photon_peak=20,
                 peak_width=None):
        r""" Creates an intensity histogram for each pixel in a PAD.  For a PAD with N pixels in total, this class
        will produce an array of shape (M, N) assuming that you requested M bins in the histogram.

        Arguments:
            bin_min (float): The minimum value corresponding to histogram bin *centers*.
            bin_max (float): The maximum value corresponding to histogram bin *centers*.
            n_bins (int): The number of histogram bins.
            n_pixels (int): How many pixels there are in the detector.
            zero_photon_peak (float): Where you think the peak of the zero-photon signal should be.
            one_photon_peak (float): Where you think the peak of the one-photon signal should be.
            peak_width (float): Width of peaks for fitting to 2nd-order polynomial
            """
        self.bin_min = float(bin_min)
        self.bin_max = float(bin_max)
        self.n_bins = int(n_bins)
        self.n_pixels = int(n_pixels)
        self.zero_photon_peak = zero_photon_peak
        self.one_photon_peak = one_photon_peak
        self.peak_width = peak_width
        if self.peak_width is None:
            self.peak_width = (one_photon_peak - zero_photon_peak)/4
        self.bin_delta = (self.bin_max - self.bin_min) / (self.n_bins - 1)
        self._idx = np.arange(self.n_pixels, dtype=int)
        self.histogram = np.zeros((self.n_pixels, self.n_bins), dtype=np.uint16)

    def to_dict(self):
        r""" Convert class to dictionary for storage. """
        return dict(bin_min=self.bin_min,
                    bin_max=self.bin_max,
                    n_bins=self.n_bins,
                    n_pixels=self.n_pixels,
                    one_photon_peak=self.one_photon_peak,
                    zero_photon_peak=self.zero_photon_peak,
                    peak_width=self.peak_width,
                    histogram=self.histogram)

    def from_dict(self, d):
        r""" Set data according to data stored in dictionary created by to_dict.  """
        self.bin_min = d['bin_min']
        self.bin_max = d['bin_max']
        self.n_bins = d['n_bins']
        self.n_pixels = d['n_pixels']
        self.one_photon_peak = d['one_photon_peak']
        self.zero_photon_peak = d['zero_photon_peak']
        self.histogram = d['histogram']

    def concatentate(self, hist):
        r""" Combines two histograms (e.g. if parallel processing and chunks need to be combined). """
        self.histogram += hist.histogram

    def get_bin_centers(self):
        r""" Returns an 1D array of histogram bin centers. """
        return np.linspace(self.bin_min, self.bin_max, self.n_bins)

    def get_histogram_normalized(self):
        r""" Returns a normalized histogram - an |ndarray| of shape (M, N) where M is the number of pixels and
        N is the number of requested bins per pixel. """
        count = np.sum(self.histogram, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(self.histogram.T/count).T

    def get_histogram(self):
        r""" Returns a copy of the histogram - an |ndarray| of shape (M, N) where M is the number of pixels and
        N is the number of requested bins per pixel."""
        return self.histogram.copy()

    def add_frame(self, data, mask=None):
        r""" Add PAD measurement to the histogram."""
        bin_index = np.floor((data - self.bin_min) / self.bin_delta).astype(int)
        idx = np.ravel_multi_index((self._idx, bin_index), (self.n_pixels, self.n_bins), mode='clip')
        if mask is not None:
            idx = idx[mask > 0]
        self.histogram.flat[idx] += 1

    def get_peaks_fast(self):
        r""" Finds the local maxima """
        hist = self.get_histogram_normalized()
        x = self.get_bin_centers()
        p0 = self.zero_photon_peak
        p1 = self.one_photon_peak
        g = (p1 - p0) / 2
        m0 = (x > p0 - g) * (x < p0 + g)
        m1 = (x > p1 - g) * (x < p1 + g)
        i0 = np.argmax(hist * m0, axis=1)
        i1 = np.argmax(hist * m1, axis=1)
        return x[i0], x[i1]

    def gain_and_offset(self):
        x0, x1 = self.get_peaks_fast()
        return x1-x0, x0

    def refined_gain_and_offset(self):
        r""" Does a 2nd-order polynomial fit to the peak.  First guess is the local maximum. """
        hist = self.get_histogram_normalized()
        x = self.get_bin_centers()
        n_pixels = hist.shape[0]
        n_bins = hist.shape[1]
        p0, p1 = self.get_peaks_fast()
        poly = np.polynomial.Polynomial
        peak_width = int(np.ceil(self.peak_width/(x[1]-x[0])/2))
        if peak_width < 2:
            raise Exception("Peak width is too small for polynomial fitting")
        peak0 = np.zeros(n_pixels)
        peak1 = np.zeros(n_pixels)
        for i in range(n_pixels):
            print(i)
            h = hist[i, :]
            p = p0[i]
            a = max(0, p-peak_width)
            b = min(n_bins-1, p+peak_width)
            yd = h[a:b+1]
            xd = x[a:b+1]
            fit, extra = poly.fit(xd, yd, 2, full=True)
            peak0[i] = fit.deriv(1).roots()[0]
            p = p1[i]
            a = max(0, p-peak_width)
            b = min(n_bins-1, p+peak_width)
            yd = h[a:b+1]
            xd = x[a:b+1]
            fit, extra = poly.fit(xd, yd, 2, full=True)
            peak1[i] = fit.deriv(1).roots()[0]
        return peak1 - peak0, peak0


class PADStats(ParallelAnalyzer):
    def __init__(self, framegetter=None, start=0, stop=None, step=1, n_processes=1, config=None, **kwargs):
        r""" A class for gathering PAD statistics from a framegetter. """
        super().__init__(framegetter=framegetter, start=start, stop=stop, step=step, n_processes=n_processes,
                         config=config, **kwargs)
        self.analyzer_name = 'PADStats'
        self.initialized = False
        self.histogrammer = None  # For building up pixel histograms, optionally
        self.histogram_params = self.config.get('histogram_params')
        self.dataset_id = None  # Often something like "r0045" or similar
        self.pad_geometry = None  # Only the first geometry found
        self.beam = None  # Will have the median wavelength
        self.min_pad = None  # Array of minimum intensities
        self.max_pad = None  # Array of maximum intensities
        self.sum_pad = None  # Sum of all intensities
        self.sum_pad2 = None  # Sum of all squared intensities
        self.mask = None  # First mask found
        self.wavelengths = None  # Array of all wavelengths
        self.gain = None
        self.offset = None

    def setup_histogram(self):
        if self.histogram_params is not None:
            self.logger.info('Setting up histogram')
            if self.histogram_params.get('n_pixels', None) is None:
                self.histogram_params['n_pixels'] = self.sum_pad.shape[0]
            self.histogrammer = PixelHistogram(**self.histogram_params)

    def add_frame(self, dat):
        if dat is None:
            self.logger.warning(f"DataFrame {self.processing_index} is None.  Skipping frame")
            return
        rdat = dat.get_raw_data_flat()
        if rdat is None:
            self.logger.warning(f'Raw data is None.  Skipping frame.')
            return
        if not self.initialized:
            self.initialize_data(rdat)
        if dat.validate():
            beam = dat.get_beam()
            self.wavelengths[self.processing_index] = beam.wavelength
            if self.beam is None:
                self.beam = beam
        else:
            self.logger.warning("DataFrame is invalid.  If it is a dark run this could be due to missing Beam info.")
        self.sum_pad += rdat
        self.sum_pad2 += rdat ** 2
        self.min_pad = np.minimum(self.min_pad, rdat)
        self.max_pad = np.maximum(self.max_pad, rdat)
        if self.dataset_id is None:
            self.dataset_id = dat.get_dataset_id()
        if self.pad_geometry is None:
            self.pad_geometry = dat.get_pad_geometry()
        if self.mask is None:
            self.mask = dat.get_mask_flat()
        if self.histogrammer is not None:
            self.histogrammer.add_frame(rdat)

    def finalize(self):
        self.logger.info('Finalizing analysis')
        if self.histogrammer is not None:
            self.logger.info('Attempting to get gain and offset from histogram')
            self.gain, self.offset = self.histogrammer.gain_and_offset()

    def clear_data(self):
        self.wavelengths = None
        self.sum_pad = None
        self.sum_pad2 = None
        self.max_pad = None
        self.min_pad = None
        self.n_processed = 0
        self.histogrammer = None
        self.initialized = False

    def initialize_data(self, rdat):
        self.logger.debug('Initializing arrays')
        s = rdat.size
        self.wavelengths = np.zeros(self.n_chunk)
        self.sum_pad = np.zeros(s)
        self.sum_pad2 = np.zeros(s)
        self.max_pad = rdat
        self.min_pad = rdat
        self.n_processed = 0
        self.setup_histogram()
        self.initialized = True

    def to_dict(self):
        stats = dict()
        stats['dataset_id'] = self.dataset_id
        stats['pad_geometry'] = self.pad_geometry
        stats['mask'] = self.mask
        stats['n_frames'] = self.n_processed
        stats['sum'] = self.sum_pad
        stats['sum2'] = self.sum_pad2
        stats['min'] = self.min_pad
        stats['max'] = self.max_pad
        stats['beam'] = self.beam
        stats['start'] = self.start
        stats['stop'] = self.stop
        stats['step'] = self.step
        stats['wavelengths'] = self.wavelengths
        if self.histogrammer is not None:
            self.logger.debug(f'Histrogram shape: {np.shape(self.histogrammer.histogram)}')
            stats['histogram'] = self.histogrammer.histogram
            stats['histogram_params'] = self.histogram_params
        if self.gain is not None:
            stats['gain'] = self.gain
            stats['offset'] = self.offset
        return stats

    def from_dict(self, stats):
        self.clear_data()
        if stats.get('sum') is None:
            self.logger.warning("Stats dictionary is empty!")
            return
        self.dataset_id = stats['dataset_id']
        self.pad_geometry = stats['pad_geometry']
        self.mask = stats['mask']
        self.n_processed = stats['n_frames']
        self.sum_pad = stats['sum']
        self.sum_pad2 = stats['sum2']
        self.min_pad = stats['min']
        self.max_pad = stats['max']
        self.beam = stats['beam']
        self.start = stats['start']
        self.stop = stats['stop']
        self.step = stats['step']
        self.wavelengths = stats['wavelengths']
        if stats.get('histogram_params'):
            self.histogram_params = stats['histogram_params']
            self.histogrammer = PixelHistogram(**stats['histogram_params'])
            self.histogrammer.histogram = stats.get('histogram')
        self.initialized = True

    def concatenate(self, stats):
        if not self.initialized:
            self.from_dict(stats)
            self.initialized = True
            return
        if stats['n_frames'] == 0:
            self.logger.debug('No frames to concatentate!')
            return
        self.start = min(self.start, stats['start'])
        self.stop = max(self.stop, stats['stop'])
        self.n_processed += stats['n_frames']
        self.wavelengths = np.concatenate([self.wavelengths, stats['wavelengths']])
        if stats['sum'] is None:
            return
        self.sum_pad += stats['sum']
        self.sum_pad2 += stats['sum2']
        self.min_pad = np.minimum(self.min_pad, stats['min'])
        self.max_pad = np.minimum(self.max_pad, stats['max'])
        if stats.get('histogram') is not None:
            self.histogrammer.histogram += stats['histogram']


def padstats(framegetter=None, start=0, stop=None, step=1, n_processes=1, config=None):
    r""" Gather PAD statistics for a dataset.

    Given a |FrameGetter| subclass instance, fetch the mean intensity, mean squared intensity, minimum,
    and maximum intensities, and optionally a pixel-by-pixel intensity histogram.  The function can run on multiple
    processors via the joblib library.  Logfiles and checkpoint files are created.

    The return of this function is a dictionary with the following keys:

    * 'sum': Sum of PAD intensities
    * 'dataset_id': A unique identifier for the data set (e.g. 'run0154')
    * 'pad_geometry': PADGeometryList
    * 'mask': Bad pixel mask
    * 'n_frames': Number of valid frames contributing to the statistics
    * 'sum': Sum of PAD intensities
    * 'sum2': Sum of squared PAD intensities
    * 'min': Pixel-by-pixel minimum of PAD intensities
    * 'max': Pixel-by-pixel maximum of PAD intensities
    * 'beam': Beam info
    * 'start': Frame at which processing started (global framegetter index)
    * 'stop': Frame at which processing stopped (global framegetter index)
    * 'step': Step size between frames (helpful to speed up processing by sub-sampling)
    * 'wavelengths': An array of wavelengths
    * 'histogram_params': Dictionary with histogram parameters
    * 'histogram': Pixel-by-pixel intensity histogram (MxN array, with M the number of pixels)

    There is a corresponding view_padstats function to view the results in this dictionary.

    padstats needs a configuration dictionary with the following contents:

    * 'log_file': Base path/name for status logging.  Set to None to skip logging.
    * 'checkpoint_file': Base path/name for saving check points.  Set to None to skip checkpoints.
    * 'checkpoint_interval': How many frames between checkpoints.
    * 'message_prefix': Prefix added to all logging messages.  For example: "Run 35" might be helpful
    * 'debug': Set to True for more logging messages.
    * 'reduce_from_checkpoints': True by default, this indicates that data produced by multiple processors should be
      compiled by loading the checkpoints from disk.  Without this, you might have memory problems.  (The need for this
      is due to the joblib package; normally the reduce functions from MPI would be used to avoid hitting the disk.)
    * 'histogram_params': If not None, triggers production of a pixel-by-pixel histogram.  This is a dictionary with
      the following entries: dict(bin_min=-30, bin_max=100, n_bins=100, zero_photon_peak=0, one_photon_peak=30)

    Arguments:
        framegetter (|FrameGetter|): A FrameGetter subclass.  If running in parallel, you should instead pass a
                                     dictionary with keys 'framegetter' (with reference to FrameGetter subclass,
                                     not an actual class instance) and 'kwargs' containing a dictionary of keyword
                                     arguments needed to create a class instance.
        start (int): Which frame to start with.
        stop (int): Which frame to stop at.
        step (int): Step size between frames (default 1).
        n_processes (int): How many processes to run in parallel (if parallel=True).
        config (dict): The configuration dictionary explained above.

    Returns: dict """
    ps = PADStats(framegetter=framegetter, start=start, stop=stop, step=step, n_processes=n_processes, config=config)
    ps.process_frames()
    return ps.to_dict()


def save_padstats(stats: dict, filepath: str):
    r""" Saves the results of the padstats function.

    Arguments:
        stats (dict): Dictionary output of padstats.
        filepath (str): Path to the file you want to save.
    """
    fileio.misc.save_pickle(stats, filepath)


def load_padstats(filepath: str):
    r""" Load the results of padstats from disk.

    Arguments:
        filepath (str): Path to the file you want to load.

    Returns: dict
    """
    stats = fileio.misc.load_pickle(filepath)
    meen = stats['sum']/stats['n_frames']
    meen2 = stats['sum2']/stats['n_frames']
    sdev = np.nan_to_num(meen2-meen**2)
    sdev[sdev < 0] = 0
    sdev = np.sqrt(sdev)
    stats['mean'] = meen
    stats['sdev'] = sdev
    return stats


def padstats_framegetter(stats):
    r""" Make a FrameGetter that can flip through the output of padstats.

    Arguments:
        stats (dict): Output of padstats.

    Returns: ListFrameGetter
    """
    beam = stats['beam']
    geom = stats['pad_geometry']
    geom = detector.PADGeometryList(geom)
    mask = stats['mask']
    meen = stats['sum']/stats['n_frames']
    meen2 = stats['sum2']/stats['n_frames']
    sdev = np.nan_to_num(meen2-meen**2)
    sdev[sdev < 0] = 0
    sdev = np.sqrt(sdev)
    dats = [('mean', meen), ('sdev', sdev), ('min', stats['min']), ('max', stats['max'])]
    if 'gain' in stats:
        dats.append(('gain', stats['gain']))
    if 'offset' in stats:
        dats.append(('offset', stats['offset']))
    dfs = []
    for (a, b) in dats:
        d = DataFrame()
        d.set_dataset_id(stats['dataset_id'])
        d.set_frame_id(a)
        d.set_pad_geometry(geom)
        if beam is not None:
            d.set_beam(beam)
        d.set_mask(mask)
        d.set_raw_data(b)
        dfs.append(d)
    return ListFrameGetter(dfs)


def view_padstats(stats, histogram=False):
    r""" View the output of padstats.

    Arguments:
        stats (dict): Output dictionary from padstats.
    """
    if histogram:
        view_histogram(stats)
    else:
        fg = padstats_framegetter(stats)
        pv = PADView(frame_getter=fg, percentiles=(1, 99))
        pv.start()


def view_histogram(stats):
    r""" View the output of padstats with histogram enabled. """
    geom = stats['pad_geometry']
    mn = stats['histogram_params']['bin_min']
    mx = stats['histogram_params']['bin_max']
    nb = stats['histogram_params']['n_bins']
    c0 = stats['histogram_params'].get('zero_photon_peak', 0)
    c1 = stats['histogram_params'].get('one_photon_peak', 30)
    x = np.linspace(mn, mx, nb)
    histdat = stats['histogram']
    h = np.mean(stats['histogram'], axis=0)
    histplot = pg.plot(x, np.log10(h+1))
    imv = imview(np.log10(histdat+1), fs_lims=[mn, mx], fs_label='Intensity', ss_label='Flat Pixel Index',
           title='Intensity Histogram')
    line = imv.add_line(vertical=True, movable=True)

    def update_histplot(line):
        i = int(np.round(line.value()))
        i = max(i, 0)
        i = min(i, histdat.shape[0]-1)
        histplot.plot(x, np.log10(histdat[i, :]+1), clear=True)
    flat_indices = np.arange(0, geom.n_pixels)
    flat_indices = stats['pad_geometry'].split_data(flat_indices)
    fg = padstats_framegetter(stats)
    pv = PADView(frame_getter=fg, percentiles=(1, 99))

    def set_line_index(evt):
        if evt is None:
            print('Event is None')
            return
        ss, fs, pid = pv.get_pad_coords_from_mouse_pos()
        if pid is None:
            pass
        else:
            fs = int(np.round(fs))
            ss = int(np.round(ss))
            line.setValue(flat_indices[pid][ss, fs])
        pass
    line.sigPositionChanged.connect(update_histplot)
    pv.proxy2 = pg.SignalProxy(pv.viewbox.scene().sigMouseMoved, rateLimit=30, slot=set_line_index)
    pv.start()


# This is too slow.
def analyze_histogram(stats, n_processes=1, debug=0):
    r""" Analyze histogram and attempt to extract offsets and gains from the zero- and one-photon peak.  Experimental.
    Use at your own risk!"""
    def dbgmsg(*args, **kwargs):
        if debug:
            print(*args, **kwargs)
    if n_processes > 1:
        if Parallel is None:
            raise ImportError('You need the joblib package to run in parallel mode.')
        stats_split = [dict(histogram=h) for h in np.array_split(stats['histogram'], n_processes, axis=0)]
        for s in stats_split:
            s['histogram_params'] = stats['histogram_params']
        out = Parallel(n_jobs=n_processes)([delayed(analyze_histogram)(s, debug=debug) for s in stats_split])
        return dict(gain=np.concatenate([out[i]['gain'] for i in range(n_processes)]),
                    offset=np.concatenate([out[i]['offset'] for i in range(n_processes)]))
    mn = stats['histogram_params']['bin_min']
    mx = stats['histogram_params']['bin_max']
    nb = stats['histogram_params']['n_bins']
    c00 = stats['histogram_params'].get('zero_photon_peak', 0)
    c10 = stats['histogram_params'].get('one_photon_peak', 30)
    x = np.linspace(mn, mx, nb)
    histdat = stats['histogram']
    poly = np.polynomial.Polynomial
    n_pixels = histdat.shape[0]
    gain = np.zeros(n_pixels)
    offset = np.zeros(n_pixels)
    for i in range(n_pixels):
        c0 = c00
        c1 = c10
        a = (c1 - c0) / 3
        o = 5
        goodfit = 1
        for j in range(2):
            w0 = np.where((x >= c0-a) * (x <= c0+a))
            w1 = np.where((x >= c1-a) * (x <= c1+a))
            x0 = x[w0]
            x1 = x[w1]
            y0 = histdat[i, :][w0]
            y1 = histdat[i, :][w1]
            if np.sum(y0) < o:
                dbgmsg('skip')
                goodfit = 0
                break
            if np.sum(y1) < o:
                dbgmsg('skip')
                goodfit = 0
                break
            if len(y0) < o:
                dbgmsg('skip')
                goodfit = 0
                break
            if len(y1) < o:
                dbgmsg('skip')
                goodfit = 0
                break
            f0, extra = poly.fit(x0, y0, o, full=True)
            xf0, yf0 = f0.linspace()
            c0 = xf0[np.where(yf0 == np.max(yf0))[0][0]]
            f1, extra = poly.fit(x1, y1, o, full=True)
            xf1, yf1 = f1.linspace()
            c1 = xf1[np.where(yf1 == np.max(yf1))[0][0]]
            a = 5
            o = 3
        if goodfit:
            gain[i] = c1-c0
            offset[i] = c0
        dbgmsg(f"Pixel {i} of {n_pixels} ({i*100/float(n_pixels):0.2f}%), gain={gain[i]}, offset={offset[i]}")
    return dict(gain=gain, offset=offset)


class RadialProfiler(ParallelAnalyzer):
    r"""
    A parallelized class for creating radial profiles from image data.
    Standard profiles are computed using fortran code.
    Bin indices are cached for speed, provided that the |PADGeometry| and |Beam| do not change.
    """
    framegetter = None
    experiment_id = None
    run_id = None
    n_bins = None  # Number of bins in radial profile
    q_range = None  # The range of q magnitudes in the 1D profile.  These correspond to bin centers
    q_edge_range = None  # Same as above, but corresponds to bin edges not centers
    bin_centers = None  # q magnitudes corresponding to 1D profile bin centers
    bin_edges = None  # q magnitudes corresponding to 1D profile bin edges (length is n_bins+1)
    bin_size = None  # The size of the 1D profile bin in q space
    initial_frame = None
    _q_mags = None  # q magnitudes corresponding to diffraction pattern intensities
    _mask = None  # The default mask, in case no mask is provided upon requesting profiles
    _fortran_indices = None  # For speed, pre-index arrays
    _pad_geometry = None  # List of PADGeometry instances
    _beam = None  # Beam instance for creating q magnitudes
    radial_sum = None
    radial_mean = None
    radial_sdev = None
    radial_sum2 = None
    radial_weight_sum = None

    def __init__(self, framegetter=None, **kwargs):
        r"""
        Arguments:
            mask (|ndarray|): Optional.  The arrays will be multiplied by this mask, and the counts per radial bin
                                will come from this (e.g. use values of 0 and 1 if you want a normal average, otherwise
                                you get a weighted average).
            n_bins (int): Number of radial bins you desire.
            q_range (list-like): The minimum and maximum of the *centers* of the q bins.
            pad_geometry (|PADGeometryList|):  Optional.  Will be used to generate q magnitudes.  You must
                                                             provide beam if you provide this.
            beam (|Beam| instance): Optional, unless pad_geometry is provided.  Wavelength and beam direction are
                                     needed in order to calculate q magnitudes.
        """
        start = kwargs.get('start', 0)
        stop = kwargs.get('stop', None)
        step = kwargs.get('step', 1)
        n_processes = kwargs.get('n_processes', 1)
        super().__init__(framegetter=framegetter, start=start, stop=stop,
                         step=step, n_processes=n_processes, **kwargs)
        self.framegetter = framegetter
        self.experiment_id = kwargs.get('experiment_id', 'default')
        self.run_id = kwargs.get('run_id', 0)
        self._pad_geometry, self._beam, self.initial_frame = get_setup_data(framegetter=self.framegetter, **kwargs)
        self._q_mags = self._pad_geometry.q_mags(beam=self._beam).astype(np.float64)
        q_range = kwargs.get('q_range', (0, np.max(self._q_mags)))
        n_bins = kwargs.get('n_bins', int(np.sqrt(self._q_mags.size) / 4.0))
        q_range = np.array(q_range)
        bin_size = (q_range[1] - q_range[0]) / float(n_bins - 1)
        self.bin_centers = np.linspace(q_range[0], q_range[1], n_bins)
        self.bin_centers.flags['WRITEABLE'] = False
        self.bin_edges = np.linspace(q_range[0] - bin_size / 2, q_range[1] + bin_size / 2, n_bins + 1)
        self.bin_edges.flags['WRITEABLE'] = False
        self.q_edge_range = np.array([q_range[0] - bin_size / 2, q_range[1] + bin_size / 2])
        self.q_edge_range.flags['WRITEABLE'] = False
        self.q_range = q_range
        self.q_range.flags['WRITEABLE'] = False
        self.n_bins = n_bins
        self.bin_size = bin_size
        mask = kwargs.get('mask', None)
        if mask is not None:
            mask = self._pad_geometry.concat_data(mask)
        else:
            self._mask = np.ones_like(self._q_mags)
        self._mask = mask.astype(np.float64)  # Because fortran code is involved
        self._mask.flags['WRITEABLE'] = False
        self.radial_sum = []
        self.radial_mean = []
        self.radial_sdev = []
        self.radial_sum2 = []
        self.radial_weight_sum = []

    def add_frame(self, dat: DataFrame):
        if dat.validate():
            data = dat.get_raw_data_flat()
            mask = dat.get_mask_flat()
            data = self._pad_geometry.concat_data(data).astype(np.float64)
            weights = (self._mask * mask).astype(np.float64)
            n_bins = self.n_bins
            indices = self._fortran_indices
            if indices is None:  # Cache for faster computing next time.
                q = self._q_mags
                q_min = self.q_range[0]
                q_max = self.q_range[1]
                indices = np.zeros(len(q), dtype=np.int32)
                scatter_f.profile_indices(q, n_bins, q_min, q_max, indices)
                self._fortran_indices = indices
            sum_ = np.zeros(n_bins, dtype=np.float64)
            sum2 = np.zeros(n_bins, dtype=np.float64)
            wsum = np.zeros(n_bins, dtype=np.float64)
            avg = np.empty(n_bins, dtype=np.float64)
            std = np.empty(n_bins, dtype=np.float64)
            scatter_f.profile_stats_indexed(data, indices, weights, sum_, sum2, wsum)
            scatter_f.profile_stats_avg(sum_, sum2, wsum, avg, std)
            self.radial_sum.append(sum_)
            self.radial_mean.append(avg)
            self.radial_sdev.append(std)
            self.radial_sum2.append(sum2)
            self.radial_weight_sum.append(wsum)

    def concatenate(self, stats):
        self.radial_sum.extend(stats.radial_sum)
        self.radial_mean.extend(stats.radial_mean)
        self.radial_sdev.extend(stats.radial_sdev)
        self.radial_sum2.extend(stats.radial_sum2)
        self.radial_weight_sum.extend(stats.radial_weight_sum)

    def to_dict(self):
        profiler_dict = dict(experiment_id=self.experiment_id,
                             run_id=self.run_id,
                             n_bins=self.n_bins,
                             q_range=self.q_range,
                             mask=self._mask,
                             pad_geometry=self._pad_geometry,
                             beam=self._beam,
                             radial_sum=self.radial_sum,
                             radial_mean=self.radial_mean,
                             radial_sdev=self.radial_sdev,
                             radial_sum2=self.radial_sum2,
                             radial_weights_sum=self.radial_weight_sum)
        return profiler_dict

    def from_dict(self, stats):
        self.experiment_id = stats['experiment_id']
        self.run_id = stats['run_id']
        self.n_bins = stats['n_bins']
        self.q_range = stats['q_range']
        self._mask = stats['mask']
        self._pad_geometry = stats['pad_geometry']
        self._beam = stats['beam']
        self.radial_sum = stats['radial_sum']
        self.radial_mean = stats['radial_mean']
        self.radial_sdev = stats['radial_sdev']
        self.radial_sum2 = stats['radial_sum2']
        self.radial_weight_sum = stats['radial_weights_sum']
