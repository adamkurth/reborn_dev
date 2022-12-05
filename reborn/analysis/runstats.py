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
import os
import sys
import logging
import time
import glob
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


def default_padstats_config():
    config = dict(log_file=None, checkpoint_file=None, checkpoint_interval=500)
    return config


def default_histogram_config():
    return dict(bin_min=-30, bin_max=100, n_bins=100)


def get_padstats_logger(filename=None, n_processes=1, process_id=0):
    logger = logging.getLogger(name='padstats')
    logger.setLevel(logging.INFO)
    pid = ""
    if process_id >= 0:
        pid = f" process {process_id+1} of {n_processes} -"
    formatter = logging.Formatter(f'%(asctime)s - %(name)s -{pid} %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level=logging.INFO)
    logger.addHandler(console_handler)
    if filename is not None:
        if len(filename) < 4 or filename[-4:] != '.log':
            filename += '.log'
        if process_id > 0:
            filename = filename.replace('.log', f'_{process_id:02d}.log')
        file_handler = logging.FileHandler(filename=filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level=logging.INFO)
        logger.addHandler(file_handler)
    return logger


def padstats(framegetter=None, start=0, stop=None, parallel=False, n_processes=1, _process_id=0,
             histogram_params=None, verbose=True, config=None):
    r""" Gather PAD statistics for a dataset.

    Given a |FrameGetter| subclass instance, fetch the mean intensity, mean squared intensity, minimum,
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
    # Config dictionary for some advanced settings
    if config is None:
        config = default_padstats_config()
    # For logging status to a file
    log_file = config.get('log_file', None)
    # Checkpoints to resume data compilation in case of a crash or timeout
    checkpoint_file = config.get('checkpoint_file', None)
    if checkpoint_file is not None:
        checkpoint_file += f'_checkpoint_{n_processes}_{_process_id}'
    checkpoint_interval = config.get('checkpoint_interval', 500)
    # Using python's fancy logger package
    logger = get_padstats_logger(log_file, n_processes, _process_id)
    logger.info('Begin processing')
    logger.info(f"framegetter={framegetter}, start={start}, stop={stop}, parallel={parallel}, n_processes="
                f"{n_processes}, _process_id={_process_id}, histogram_params={histogram_params}, config={config}")
    histogram = None
    if framegetter is None:
        raise ValueError('framegetter cannot be None')
    if parallel:
        # If parallel set to true, then this process will simply launch the other processes and compile results
        if Parallel is None:
            raise ImportError('You need the joblib package to run padstats in parallel mode.')
        if not isinstance(framegetter, dict):
            if framegetter.init_params is None:
                raise ValueError('This FrameGetter does not have init_params attribute needed to make a replica')
            framegetter = {'framegetter': type(framegetter), 'kwargs': framegetter.init_params}
        out = Parallel(n_jobs=n_processes)(delayed(padstats)(framegetter=framegetter, start=start, stop=stop,
                parallel=False, n_processes=n_processes, _process_id=i+1, histogram_params=histogram_params,
                                                             verbose=verbose, config=config)
                                                             for i in range(n_processes))
        tot = out[0]
        tot['wavelengths'] = np.concatenate([o['wavelengths'] for o in out])
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
    if _process_id is not None:
        frame_ids = np.array_split(frame_ids, n_processes)[_process_id-1]
    else:
        _process_id = 1
    first = True
    t0 = time.time()
    tot_frames = len(frame_ids)
    jumpstart = 0
    pcpf = None
    checkpoint = dict()
    # Check if there is an existing checkpoint file that we can start from
    if checkpoint_file:
        cpfs = sorted(glob.glob(checkpoint_file + '*'))
        if len(cpfs) > 0:
            c = cpfs[-1]
            jumpstart = int(c.split('_')[-1])
            checkpoint = load_padstats(c)
            first = False
    cpstart = checkpoint.get('start', start)
    cpstop = checkpoint.get('stop', stop)
    if (cpstart != start) or (cpstop != stop):
        logger.warning('Checkpoint start/stop does not match - Rejecting!')
        checkpoint = dict()
    dataset_id = checkpoint.get('dataset_id', None)
    pad_geometry = checkpoint.get('pad_geometry', None)
    mask = checkpoint.get('mask', None)
    n_frames = checkpoint.get('n_frames', 0)
    sum_pad = checkpoint.get('sum', None)
    min_pad = checkpoint.get('min', None)
    max_pad = checkpoint.get('max', None)
    sum_pad2 = checkpoint.get('sum2', None)
    beam = checkpoint.get('beam', None)
    wavelengths = checkpoint.get('wavelengths', None)
    run_stats_keys = ['dataset_id', 'pad_geometry', 'mask',
                      'n_frames', 'sum', 'min',
                      'max', 'sum2', 'beam',
                      'start', 'stop', 'wavelengths']
    for n in range(jumpstart, tot_frames):
        i = frame_ids[n]
        ts = time.ctime()
        dt = time.time() - t0
        tr = dt*(tot_frames-n)/(n+1)
        logger.info(f"Frame {i} ({n} of {tot_frames}): {n/tot_frames*100}% @ {dt/60:.1f} min. => {tr/60:.1f} min. "
                    f"remaining'")
        # ==========================================================================
        # This is the actual processing.  Very simple.
        # ==========================================================================
        dat = framegetter.get_frame(frame_number=i)
        if dat is None:
            print(f'{ts}: Frame {i:6d} is None!!!')
            continue
        rdat = dat.get_raw_data_flat()
        if rdat is None:
            logger.warning(f'Raw data is None')
            continue
        if first:
            s = rdat.size
            wavelengths = np.zeros(tot_frames)
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
        if dat.validate():
            beam_data = dat.get_beam()
            wavelengths[n] = beam_data.wavelength
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
        # End processing ========================================================
        if (checkpoint_file is not None) and ((n+1) % checkpoint_interval == 0):
            beam = None
            w = np.where(wavelengths > 0)
            if len(w) > 0:
                avg_wavelength = np.sum(wavelengths[w]) / len(w[0])
                beam = Beam(wavelength=avg_wavelength)
            else:
                logger.warning('No beam info found!')
            run_stats_vals = [dataset_id, pad_geometry, mask,
                              n_frames, sum_pad, min_pad,
                              max_pad, sum_pad2, beam,
                              start, stop, wavelengths]
            runstats = dict(zip(run_stats_keys, run_stats_vals))
            if histogram is not None:
                runstats['histogram'] = histogram.histogram
            runstats['histogram_params'] = histogram_params
            if not os.path.exists(checkpoint_file):
                cpf = checkpoint_file + f'_{n+1:07d}'
                logger.info(f'Saving checkpoint file {cpf}')
                save_padstats(runstats, cpf)
                if (pcpf is not None):
                    logger.info(f'Removing previous checkpoint file {pcpf}')
                    os.remove(pcpf)
                pcpf = cpf
    run_stats_vals = [dataset_id, pad_geometry, mask,
                      n_frames, sum_pad, min_pad,
                      max_pad, sum_pad2, beam,
                      start, stop, wavelengths]
    runstats = dict(zip(run_stats_keys, run_stats_vals))
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
