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

import time
from functools import partial
import numpy as np
import pyqtgraph as pg
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None
from .. import source, detector, fileio
from ..dataframe import DataFrame
from ..fileio.getters import ListFrameGetter
from ..source import Beam
from ..viewers.qtviews.padviews import PADView
from ..external.pyqtgraph import imview


class PixelHistogram:

    count = 0  #: Number of frames contributing to the histogram.
    bin_min = None  #: The minimum value corresponding to histogram bin *centers*.
    bin_max = None  #: The maximum value corresponding to histogram bin *centers*.
    n_bins = None  #: The number of histogram bins.

    def __init__(self, bin_min=None, bin_max=None, n_bins=None, n_pixels=None):
        r""" Creates an intensity histogram for each pixel in a PAD.  For a PAD with N pixels in total, this class
        will produce an array of shape (M, N) assuming that you requested M bins in the histogram.

        Arguments:
            bin_min (float): The minimum value corresponding to histogram bin *centers*.
            bin_max (float): The maximum value corresponding to histogram bin *centers*.
            n_bins (int): The number of histogram bins.
            n_pixels (int): How many pixels there are in the detector.
            """
        self.bin_min = float(bin_min)
        self.bin_max = float(bin_max)
        self.n_bins = int(n_bins)
        self.n_pixels = int(n_pixels)
        self.bin_delta = (self.bin_max - self.bin_min) / (self.n_bins - 1)
        self._idx = np.arange(self.n_pixels, dtype=int)
        self.histogram = np.zeros((self.n_pixels, self.n_bins), dtype=int)

    def get_bin_centers(self):
        r""" Returns an 1D array of histogram bin centers. """
        return np.linspace(self.bin_min, self.bin_max, self.n_bins)

    def get_histogram_normalized(self):
        r""" Returns a normalized histogram - an |ndarray| of shape (M, N) where M is the number of pixels and
        N is the number of requested bins per pixel. """
        if isinstance(self.count, int):
            h = self.histogram / self.count
        else:
            h = np.divide(self.histogram, self.count, np.zeros_like(), where=self.count > 0)
        return h

    def get_histogram(self):
        r""" Returns a copy of the histogram - an |ndarray| of shape (M, N) where M is the number of pixels and
        N is the number of requested bins per pixel."""
        return self.histogram.copy()

    def add_frame(self, data, mask=None):
        r""" Add PAD measurement to the histogram."""
        if mask is not None:
            self.count += mask
        else:
            self.count += 1
        bin_index = np.floor((data - self.bin_min) / self.bin_delta).astype(int)
        idx = np.ravel_multi_index((self._idx, bin_index), (self.n_pixels, self.n_bins), mode='clip')
        if mask is not None:
            idx = idx[mask > 0]
        self.histogram.flat[idx] += 1


def padstats(framegetter=None, start=0, stop=None, parallel=False, n_processes=None, process_id=None,
             histogram_params=None, verbose=False):
    r""" Given a |FrameGetter| subclass instance, fetch the mean intensity, mean squared intensity, minimum,
    and maximum intensities, and optionally a pixel-by-pixel intensity histogram.  The function can run in a
    multiprocessing mode through recursion via the joblib library.

    Note:
        If you run the function in parallel mode, you cannot pass a |FrameGetter| instance from the main process to the
        children processes.  Each process must initialize its own instance of the |FrameGetter|.  Instead of
        passing a class instance, you must instead pass in a dictionary with the 'framegetter' key set to the desired
        |FrameGetter| subclass (not an instance of the subclass), and the 'kwargs' key set to the keyword arguments
        needed to instantiate the subclass.

    The return of this function is a dictionary as follows:

    {'sum': sum_pad, 'sum2': sum_pad2, 'min': min_pad, 'max': max_pad, 'n_frames': n_frames,
     'dataset_id': dat.get_dataset_id(), 'pad_geometry': dat.get_pad_geometry(),
     'mask': dat.get_mask_flat(), 'beam': dat.get_beam(), 'histogram': histogram}

    There is a corresponding view_padstats function to view the results in this dictionary.

    Arguments:
        framegetter (|FrameGetter|): A FrameGetter subclass.  If running in parallel, you should instead pass a
                                     dictionary with keys 'framegetter' (with reference to FrameGetter subclass,
                                     not an actual class instance) and 'kwargs' containing a dictionary of keyword
                                     arguments needed to create a class instance.
        start (int): Which frame to start with.
        stop (int): Which frame to stop at.
        parallel (bool): Set to true to use joblib for multiprocessing.
        n_processes (int): How many processes to run in parallel (if parallel=True).
        histogram_params (dict): Optional: A dictionary with the keyword arguments needed for the PixelHistogram class
                                    instance.  The keys should be dict(bin_min=None, bin_max=None, n_bins=None,
                                    n_pixels=None)

    Returns: dict """
    histogram = None
    if framegetter is None:
        raise ValueError('framegetter cannot be None')
    if parallel:
        if Parallel is None:
            raise ImportError('You need the joblib package to run padstats in parallel mode.')
        if not isinstance(framegetter, dict):
            if framegetter.init_params is None:
                raise ValueError('This FrameGetter does not have init_params attribute needed to make a replica')
            framegetter = {'framegetter': type(framegetter), 'kwargs': framegetter.init_params}
        out = Parallel(n_jobs=n_processes)(delayed(padstats)(framegetter=framegetter, start=start, stop=stop,
                                                             parallel=False, n_processes=n_processes, process_id=i,
                                                             histogram_params=histogram_params, verbose=verbose)
                                                             for i in range(n_processes))
        tot = out[0]
        for o in out[1:]:
            if isinstance(o['sum'], np.ndarray):
                tot['sum'] += o['sum']
            if isinstance(o['sum2'], np.ndarray):
                tot['sum2'] += o['sum2']
            if isinstance(o['min'], np.ndarray):
                tot['min'] = np.minimum(tot['min'], o['min'])
            if isinstance(o['max'], np.ndarray):
                tot['max'] = np.minimum(tot['max'], o['max'])
            if histogram_params is not None:
                if isinstance(o['histogram'], np.ndarray):
                    tot['histogram'] += o['histogram']
            tot['start'] = min(tot['start'], o['start'])
            tot['stop'] = max(tot['stop'], o['stop'])
            tot['n_frames'] += o['n_frames']
        return tot
    if isinstance(framegetter, dict):
        framegetter = framegetter['framegetter'](**framegetter['kwargs'])
    if stop is None:
        stop = framegetter.n_frames
    frame_ids = np.arange(start, stop, dtype=int)
    if process_id is not None:
        frame_ids = np.array_split(frame_ids, n_processes)[process_id]
    first = True
    sum_pad = None
    sum_pad2 = None
    min_pad = None
    max_pad = None
    beam_wavelength = 0
    beam_frames = 0
    n_frames = 0
    dataset_id = None
    pad_geometry = None
    mask = None
    t0 = time.time()
    for (n, i) in enumerate(frame_ids):
        if verbose:
            print(f'Frame {i:6d} ({n / len(frame_ids) * 100:0.2g}%, {(time.time()-t0)/60:.1f} minutes)')
        dat = framegetter.get_frame(frame_number=i)
        if dat is None:
            print(f'Frame {i:6d} is None!!!')
            continue
        rdat = dat.get_raw_data_flat()
        if rdat is None:
            continue
        if dat.validate():
            beam_data = dat.get_beam()
            beam_wavelength += beam_data.wavelength
            beam_frames += 1
        if first:
            s = rdat.size
            sum_pad = np.zeros(s)
            sum_pad2 = np.zeros(s)
            max_pad = rdat
            min_pad = rdat
            n_frames = np.zeros(s)
            if histogram_params is not None:
                if histogram_params.get('n_pixels', None) is None:
                    histogram_params['n_pixels'] = rdat.size
                histogram = PixelHistogram(**histogram_params)
            first = False
        sum_pad += rdat
        sum_pad2 += rdat ** 2
        min_pad = np.minimum(min_pad, rdat)
        max_pad = np.maximum(max_pad, rdat)
        if dataset_id is None:
            dataset_id = dat.get_dataset_id()
        if pad_geometry is None:
            pad_geometry = dat.get_pad_geometry()
        if mask is None:
            mask = dat.get_mask_flat()
        if histogram is not None:
            histogram.add_frame(rdat, mask=dat.get_mask_flat())
        n_frames += 1
    if beam_frames == 0:
        beam = None
    else:
        avg_wavelength = beam_wavelength / beam_frames
        beam = Beam(wavelength=avg_wavelength)
    run_stats_keys = ['dataset_id', 'pad_geometry', 'mask',
                      'n_frames', 'sum', 'min',
                      'max', 'sum2', 'beam',
                      'start', 'stop']
    run_stats_vals = [dataset_id, pad_geometry, mask,
                      n_frames, sum_pad, min_pad,
                      max_pad, sum_pad2, beam,
                      start, stop]
    runstats = dict(zip(run_stats_keys, run_stats_vals))
    if histogram is not None:
        runstats['histogram'] = histogram.histogram
    runstats['histogram_params'] = histogram_params
    return runstats


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
    dats = (('mean', meen), ('sdev', sdev), ('min', stats['min']), ('max', stats['max']))
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
    histplot = pg.plot(x, np.log10(h))
    imv = imview(np.log10(histdat+1), fs_lims=[mn, mx], fs_label='Intensity', ss_label='Flat Pixel Index',
           title='Intensity Histogram')
    line = imv.add_line(vertical=True, movable=True)
    def update_histplot(line, c0, c1, a=10):
        i = int(np.round(line.value()))
        i = max(i, 0)
        i = min(i, histdat.shape[0]-1)
        histplot.plot(x, np.log10(histdat[i, :]+1), clear=True)
        if 1:  # TODO: Some hard-coded numbers below should instead be grabbed from histogram_params
            poly = np.polynomial.Polynomial
            o = 5
            for j in range(2):
                w0 = np.where((x >= c0-a) * (x <= c0+a))
                w1 = np.where((x >= c1-a) * (x <= c1+a))
                x0 = x[w0]
                x1 = x[w1]
                y0 = histdat[i, :][w0]
                y1 = histdat[i, :][w1]
                f0 = poly.fit(x0, y0, o)
                xf0, yf0 = f0.linspace()
                c0 = xf0[np.where(yf0 == np.max(yf0))]
                f1 = poly.fit(x1, y1, o)
                xf1, yf1 = f1.linspace()
                c1 = xf1[np.where(yf1 == np.max(yf1))]
                a = 5
                o = 3
            histplot.plot(xf0, np.log10(yf0 + 1), pen='g')
            histplot.plot(xf1, np.log10(yf1 + 1), pen='g')
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
    line.sigPositionChanged.connect(partial(update_histplot, c0=c0, c1=c1, a=(c1-c0)/3))
    pv.proxy2 = pg.SignalProxy(pv.viewbox.scene().sigMouseMoved, rateLimit=30, slot=set_line_index)
    pv.start()
