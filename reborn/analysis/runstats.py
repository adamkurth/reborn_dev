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

    def __init__(self, bin_min=None, bin_max=None, n_bins=None, n_pixels=None, **kwargs):
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
        self.histogram = np.zeros((self.n_pixels, self.n_bins), dtype=np.uint16)

    def get_bin_centers(self):
        r""" Returns an 1D array of histogram bin centers. """
        return np.linspace(self.bin_min, self.bin_max, self.n_bins)

    def get_histogram_normalized(self):
        r""" Returns a normalized histogram - an |ndarray| of shape (M, N) where M is the number of pixels and
        N is the number of requested bins per pixel. """
        count = np.sum(self.histogram, axis=1)
        return np.divide(self.histogram, count, np.zeros_like(), where=count > 0)

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


def default_padstats_config(histogram=False):
    r""" Get a default padstats config dictionary to specify logging, checkpoints, messaging, debugging, and the method
    by which the results from multiple processes are reduced.
    """
    hp = None
    if histogram:
        hp = dict(bin_min=-30, bin_max=100, n_bins=100, zero_photon_peak=0, one_photon_peak=30)
    config = dict(log_file=None, checkpoint_file=None, checkpoint_interval=500, message_prefix="", debug=True,
                  reduce_from_checkpoints=True, histogram_params=hp)
    return config


def default_histogram_config():
    r""" Get a default dictionary for the creation of PAD histogram.  These numbers are probably no good for you!"""
    return dict(bin_min=-30, bin_max=100, n_bins=100)


class PADStats:
    def __init__(self, framegetter=None, start=0, stop=None, step=1, parallel=False, n_processes=1, process_id=0,
                       config=None):
        r""" A class for gathering PAD statistics from a framegetter. """
        # ================================================================================================
        # The basic function that this class provides (creating averages, histogram, etc) is very simple.  However, the
        # class itself is rather complicated because of our efforts to
        # (1) Systematically log progress with log files and stdout.
        # (2) Create checkpoint files in order to deal with timeouts or other cases in which a run cannot be completed.
        # (3) Enable parallel processing via the joblib package.
        # =================================================================================================
        self.start = start  # Global start for the full run/framegetter
        self.stop = stop  # Global stop point for the full run/framegetter
        self.step = step  # Global step size for the full run/framegetter
        self.parallel = parallel  # If parallel is true, the processing will be divided into multiple processes
        self.n_processes = n_processes
        self.process_id = process_id  # Don't set this manually; it is handled internally
        self.config = config
        if config is None:
            self.config = default_padstats_config()
        self.logger = None
        self.setup_logger()
        self.framegetter = None
        self.framegetter_dict = None  # Needed to create replicas of the framegetter in sub-processes
        self.setup_framegetter(framegetter)
        self.current_checkpoint_number = 0
        self.previous_checkpoint_file = None
        self.checkpoint_interval = None
        self.checkpoint_file_base = None
        self.reduce_from_checkpoints = None  # Reduce/concatenate data by first saving to disk (minimize memory use)
        self.setup_checkpoints()
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
        self.n_chunk = None  # Total frames expected for this chunk of the run (with possible bad frames)
        self.n_processed = 0  # Number of frames actually processed contributing to the stats (not counting bad frames)
        if self.stop is None:
            self.stop = self.framegetter.n_frames
        self.stop = min(self.stop, self.framegetter.n_frames)
    @staticmethod
    def get_default_config(histogram=True):
        hp = None
        if histogram:
            hp = dict(bin_min=-30, bin_max=100, n_bins=100, zero_photon_peak=0, one_photon_peak=30)
        return dict(log_file=None, checkpoint_file=None, checkpoint_interval=500, message_prefix="", debug=True,
                  reduce_from_checkpoints=True, histogram_params=hp)
    def setup_logger(self):
        r""" Setup padstats logging using the python logging package.  Helps maintain consistent form of logs in both
        stdout and log files.  Specifies which process is running and so on. """
        # Sometimes we want to prefix a run number or experiment ID (for example).
        message_prefix = self.config.get("message_prefix", "")
        # Where to put the log file.
        logger = logging.getLogger(name='padstats')
        self.logger = logger
        if len(logger.handlers) > 0:
            return
        logger.propagate = False
        if self.config.get('debug'):
            level = logging.DEBUG
        else:
            level = logging.INFO
        logger.setLevel(level)
        pid = f"Process {self.process_id} of {self.n_processes}"
        if self.process_id == 0:
            pid = f"Process 0 (main)"
        formatter = " - ".join(["%(asctime)s", "%(levelname)s", "%(name)s", f"{pid}", f"{message_prefix} %(message)s"])
        formatter = logging.Formatter(formatter)
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level=level)
        logger.addHandler(console_handler)
        filename = self.config.get('log_file')
        if filename is not None:
            if len(filename) < 4 or filename[-4:] != '.log':
                filename += '.log'
            if self.process_id > 0:
                filename = filename.replace('.log', f'_{self.process_id:02d}.log')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            file_handler = logging.FileHandler(filename=filename)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level=level)
            logger.addHandler(file_handler)
            logger.info("\n"+"="*40+f"\nNew run, process {self.process_id} of {self.n_processes}\n"+"="*40)
            logger.info(f"Logging to file {filename}")
        else:
            logger.info(f"No logfile specified.")
    def setup_framegetter(self, framegetter):
        r""" Setup the framegetter.  If running in parallel then we need to prepare a dictionary that allows the
        framegetter to be created within another process.  If not, then we might need to utilize said dictionary
        to create a framegetter instance. """
        if isinstance(framegetter, dict):
            self.framegetter_dict = framegetter
            self.logger.info('Creating framegetter')
            self.framegetter = framegetter['framegetter'](**framegetter['kwargs'])
            self.logger.debug("Created framegetter")
        else:
            self.framegetter = framegetter
        if self.parallel:
            if Parallel is None:
                raise ImportError('You need the joblib package to run padstats in parallel mode.')
            if self.framegetter_dict is None:
                if framegetter.init_params is None:
                    raise ValueError('This FrameGetter does not have init_params attribute needed to make a replica')
                self.framegetter_dict = {'framegetter': type(framegetter), 'kwargs': framegetter.init_params}
    def setup_checkpoints(self):
        # Checkpoints to resume in case of a crash or timeout
        checkpoint_file = self.config.get('checkpoint_file', None)
        if checkpoint_file is not None:
            checkpoint_file += f'_checkpoint_{self.n_processes}_{self.process_id}'
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            logging.info(f"Checkpoint file base: {checkpoint_file}")
        self.reduce_from_checkpoints = self.config.get("reduce_from_checkpoints", True)
        if checkpoint_file is None:
            logging.warning(f"There will be no checkpoint files!")
            self.reduce_from_checkpoints = False
        checkpoint_interval = self.config.get('checkpoint_interval', 500)
        if checkpoint_file:
            logging.info(f"Checkpoint file base: {checkpoint_file}, Interval: {checkpoint_interval}")
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_file_base = checkpoint_file
    def setup_histogram(self):
        if self.histogram_params is not None:
            if self.histogram_params.get('n_pixels', None) is None:
                self.histogram_params['n_pixels'] = self.sum_pad.shape[0]
            self.histogrammer = PixelHistogram(**self.histogram_params)
    def save_checkpoint(self):
        if self.checkpoint_file_base is None:
            return
        cframe = self.processing_index
        if not (((cframe+1) % self.checkpoint_interval == 0) or (cframe == self.n_chunk - 1)):
            return
        self.logger.debug("Processing checkpoint")
        self.logger.debug(f'Previous checkpoint file: {self.previous_checkpoint_file}')
        cpf = self.checkpoint_file_base + f'_{cframe + 1:07d}'
        self.logger.info(f'Saving checkpoint file {cpf}')
        self.save(cpf)
        if self.previous_checkpoint_file is not None:
            self.logger.info(f'Removing previous checkpoint file {self.previous_checkpoint_file}')
            os.remove(self.previous_checkpoint_file)
        self.previous_checkpoint_file = cpf
    def load_checkpoint(self):
        if self.checkpoint_file_base:
            self.logger.info(f"Seeking checkpoint files {self.checkpoint_file_base}*")
            cpfs = sorted(glob.glob(self.checkpoint_file_base + '*'))
            self.logger.info(f"Found {len(cpfs)} possible checkpoints")
            while len(cpfs) > 0:
                c = cpfs.pop()
                try:
                    self.logger.info(f'Loading checkpoint file {c}')
                    stats = load_padstats(c)
                    if self.start != stats['start'] or self.stop != stats['stop'] or self.step != stats['step']:
                        self.logger.warning('The start/stop/step of the checkpoint are mismatched with this job')
                    idx = int(c.split('_')[-1])
                    self.from_dict(stats)
                    self.logger.info(f'Starting at frame {idx}')
                    self.processing_index = idx
                    break
                except Exception as e:
                    self.logger.warning(f"Problem loading file {c}")
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
            self.histogrammer.add_frame(rdat)  #, mask=dat.get_mask_flat())
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
    def save(self, filename):
        d = self.to_dict()
        save_padstats(self.to_dict(), filename)
    def load(self, filename):
        d = load_padstats(filename)
        self.from_dict(d)
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
    def process_frames(self):
        if self.parallel:
            return self.process_parallel()
        self.logger.info(f"Global start frame: {self.start}")
        self.logger.info(f"Global stop frame: {self.stop}")
        self.logger.info(f"Global step size: {self.step}")
        frame_ids = np.arange(self.start, self.stop, self.step, dtype=int)
        frame_ids = np.array_split(frame_ids, self.n_processes)[self.process_id - 1]
        t0 = time.time()
        self.n_processed = 0
        self.processing_index = 0
        self.framegetter_index = 0
        self.n_chunk = len(frame_ids)
        self.logger.info(f"Total frames for this process: {self.n_chunk}")
        if self.n_chunk == 0:
            return None
        self.load_checkpoint()  # This will fast forward if possible.  Affects processing indices
        fpsf = 0  # Frames processed so far (not counting those restored from checkpoint)
        ftp = self.n_chunk - self.processing_index  # Total frames to process (not counting
        for n in range(self.processing_index, self.n_chunk):
            self.processing_index = n
            self.framegetter_index = frame_ids[n]
            fpsf += 1
            fg_idx = frame_ids[n]
            dt = time.time() - t0  # Total processing time so far
            atpf = dt / fpsf  # Average time per frame
            tr = atpf*(ftp - fpsf)  # Time remaining
            freq = 1/atpf if atpf > 0 else 0
            self.logger.info(f"Frame ID {fg_idx} (# {n+1} of {self.n_chunk}) - {freq:.2f} Hz => {tr / 60:.1f} min. "
                             f"remaining")
            dat = self.framegetter.get_frame(frame_number=fg_idx)
            if dat is None:
                self.logger.warning('Frame is None')
            self.add_frame(dat)
            if self.histogrammer is not None:
                self.logger.debug('Adding frame to histogram')
                self.histogrammer.add_frame(dat.get_raw_data_flat())
            self.n_processed += 1
            self.save_checkpoint()
        if self.beam is not None:
            self.logger.info("Setting nominal beam wavelength to the median of the dataset")
            self.beam.wavelength = np.median(self.wavelengths[self.wavelengths > 0])
        self.logger.info('Processing completed')
        if self.reduce_from_checkpoints:
            self.logger.info('Returning checkpoint file path')
            return self.previous_checkpoint_file
        self.logger.info('Returning dictionary')
        return self.to_dict()
    def process_parallel(self):
        self.logger.info(f"Launching {self.n_processes} parallel processes")
        n = self.n_processes
        fg = self.framegetter_dict
        conf = self.config
        out = Parallel(n_jobs=n)(delayed(padstats)(framegetter=fg, start=self.start, stop=self.stop, step=self.step,
                parallel=False, n_processes=n, process_id=i+1, config=conf) for i in range(n))
        self.logger.info(f"Compiling results from {self.n_processes} processes")
        self.clear_data()
        for i in range(self.n_processes):
            stats = out[i]
            if stats is None:
                self.logger.info(f"No results from process {i}")
                continue
            if isinstance(stats, str):
                self.logger.info(f"Loading checkpoint file {stats}")
                stats = load_padstats(stats)
            if stats is None:
                self.logger.info(f"No results from process {i}")
                continue
            if stats['n_frames'] == 0:
                self.logger.info(f"No results from process {i}")
                continue
            self.logger.info(f"Concatenating results from process {i} ({stats['n_frames']} frames).")
            self.concatenate(stats)
        d = self.to_dict()
        self.logger.info(f"{d['n_frames']} frames combined.")
        self.logger.info(f"Done")
        return d


def padstats(framegetter=None, start=0, stop=None, step=None, parallel=False, n_processes=1, process_id=0, config=None):
    r""" Gather PAD statistics for a dataset.

    Given a |FrameGetter| subclass instance, fetch the mean intensity, mean squared intensity, minimum,
    and maximum intensities, and optionally a pixel-by-pixel intensity histogram.  The function can run on multiple
    processors via the joblib library.  Logfiles and checkpoint files are created.

    Note:
        If you run the function in parallel mode, you cannot pass a |FrameGetter| instance from the main process to the
        children processes.  Each process must initialize its own instance of the |FrameGetter|.  Instead of
        passing a class instance, you must instead pass in a dictionary with the 'framegetter' key set to the desired
        |FrameGetter| subclass (not an instance of the subclass), and the 'kwargs' key set to the keyword arguments
        needed to instantiate the subclass.
        For example: dict(framegetter=LCLSFrameGetter, kwargs=dict(experiment_id='cxilu1817', run_number=150))

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
        parallel (bool): Set to true to use joblib for parallel processing.
        n_processes (int): How many processes to run in parallel (if parallel=True).
        config (dict): The configuration dictionary explained above.

    Returns: dict """
    ps = PADStats(framegetter=framegetter, start=start, stop=stop, step=step, parallel=parallel,
                  n_processes=n_processes, process_id=process_id, config=config)
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
    def update_histplot(line, c0, c1, a=10):
        i = int(np.round(line.value()))
        i = max(i, 0)
        i = min(i, histdat.shape[0]-1)
        histplot.plot(x, np.log10(histdat[i, :]+1), clear=True)
        if 1:  # TODO: Some hard-coded numbers below should instead be grabbed from histogram_params
            poly = np.polynomial.Polynomial
            o = 5
            for j in range(2):
                have_fit = False
                w0 = np.where((x >= c0-a) * (x <= c0+a))
                w1 = np.where((x >= c1-a) * (x <= c1+a))
                x0 = x[w0]
                x1 = x[w1]
                y0 = histdat[i, :][w0]
                y1 = histdat[i, :][w1]
                if len(x0) < o or len(x1) < o:
                    break
                f0 = poly.fit(x0, y0, o)
                xf0, yf0 = f0.linspace()
                c0 = xf0[np.where(yf0 == np.max(yf0))]
                f1 = poly.fit(x1, y1, o)
                xf1, yf1 = f1.linspace()
                c1 = xf1[np.where(yf1 == np.max(yf1))]
                a = 5
                o = 3
                have_fit = True
            if have_fit:
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


class ParallelAnalyzer:

    def __init__(self, framegetter=None, start=0, stop=None, step=1, parallel=False, n_processes=1, process_id=0,
                       config=None):
        r""" A skeleton for parallel processing with logging and checkpoints.  Do the following to make this worK:

        Put all needed parameters in a single config dictionary and provide the dictionary at startup.

        Define the add_frame method.  It should handle the DataFrames that come from the FrameGetter.  (Make sure it
        can handle a None type.)  This should build up results.

        Define the save and load methods, which should save/load the current state of the data that you are compiling.

        Define the concatenate method, which should combine data from different chunks when handled by different
        processors.

        Define the from_dict and to_dict methods, which should convert all the relevant data into a dictionary.

        config["message_prefix"] = Prefix to log messages


        """
        self.analyzer_name = None  # Re-define this to something more sensible
        self.start = start  # Global start for the full run/framegetter
        self.stop = stop  # Global stop point for the full run/framegetter
        self.step = step  # Global step size for the full run/framegetter
        self.parallel = parallel  # If parallel is true, the processing will be divided into multiple processes
        self.n_processes = n_processes
        self.process_id = process_id  # Don't set this manually; it is handled internally
        self.config = config
        self.logger = None
        self.setup_logger()
        self.framegetter = None
        self.framegetter_dict = None  # Needed to create replicas of the framegetter in sub-processes
        self.setup_framegetter(framegetter)
        self.current_checkpoint_number = 0
        self.previous_checkpoint_file = None
        self.checkpoint_interval = None
        self.checkpoint_file_base = None
        self.reduce_from_checkpoints = None  # Reduce/concatenate data by first saving to disk (minimize memory use)
        self.setup_checkpoints()
        self.initialized = False
        self.n_chunk = None  # Total frames expected for this chunk of the run (with possible bad frames)
        self.n_processed = 0  # Number of frames actually processed contributing to the stats (not counting bad frames)
        if self.stop is None:
            self.stop = self.framegetter.n_frames
        self.stop = min(self.stop, self.framegetter.n_frames)
        self.processing_index = 0
        self.framegetter_index = 0

    def setup_logger(self):
        r""" Setup logger.  This is affected by the config dictionary keys 'debug', 'message_prefix', 'log_file' """
        # Sometimes we want to prefix a run number or experiment ID (for example).
        message_prefix = self.config.get("message_prefix", "")
        # Where to put the log file.
        logger = logging.getLogger(name=self.analyzer_name)
        self.logger = logger
        if len(logger.handlers) > 0:
            return
        logger.propagate = False
        if self.config.get('debug'):
            level = logging.DEBUG
        else:
            level = logging.INFO
        logger.setLevel(level)
        pid = f"Process {self.process_id} of {self.n_processes}"
        if self.process_id == 0:
            pid = f"Process 0 (main)"
        formatter = " - ".join(["%(asctime)s", "%(levelname)s", "%(name)s", f"{pid}", f"{message_prefix} %(message)s"])
        formatter = logging.Formatter(formatter)
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level=level)
        logger.addHandler(console_handler)
        filename = self.config.get('log_file')
        if filename is not None:
            if len(filename) < 4 or filename[-4:] != '.log':
                filename += '.log'
            if self.process_id > 0:
                filename = filename.replace('.log', f'_{self.process_id:02d}.log')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if self.config.get('clear_logs', False):
                if os.path.exists(filename):
                    self.logger.info(f'Removing log file {filename}')
                    os.remove(filename)
            file_handler = logging.FileHandler(filename=filename)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level=level)
            logger.addHandler(file_handler)
            logger.info("\n"+"="*40+f"\nNew run, process {self.process_id} of {self.n_processes}\n"+"="*40)
            logger.info(f"Logging to file {filename}")
        else:
            logger.info(f"No logfile specified.")

    def setup_framegetter(self, framegetter):
        r""" Setup the framegetter.  If running in parallel then we need to prepare a dictionary that allows the
        framegetter to be created within another process.  If not, then we might need to utilize said dictionary
        to create a framegetter instance. """
        if isinstance(framegetter, dict):
            self.framegetter_dict = framegetter
            self.logger.info('Creating framegetter')
            self.framegetter = framegetter['framegetter'](**framegetter['kwargs'])
            self.logger.debug("Created framegetter")
        else:
            self.framegetter = framegetter
        if self.parallel:
            if Parallel is None:
                raise ImportError('You need the joblib package to run padstats in parallel mode.')
            if self.framegetter_dict is None:
                if framegetter.init_params is None:
                    raise ValueError('This FrameGetter does not have init_params attribute needed to make a replica')
                self.framegetter_dict = {'framegetter': type(framegetter), 'kwargs': framegetter.init_params}

    def setup_checkpoints(self):
        r""" Setup checkpoints in the case of timeouts.  Affected by config keys 'checkpoint_file' and
        'checkpoint_interval' """
        checkpoint_file = self.config.get('checkpoint_file', None)
        if checkpoint_file is not None:
            checkpoint_file += f'_checkpoint_{self.n_processes}_{self.process_id}'
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            logging.info(f"Checkpoint file base: {checkpoint_file}")
        self.reduce_from_checkpoints = self.config.get("reduce_from_checkpoints", True)
        if checkpoint_file is None:
            logging.warning(f"There will be no checkpoint files!")
            self.reduce_from_checkpoints = False
        checkpoint_interval = self.config.get('checkpoint_interval', 500)
        if checkpoint_file:
            logging.info(f"Checkpoint file base: {checkpoint_file}, Interval: {checkpoint_interval}")
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_file_base = checkpoint_file
        if self.config.get('clear_checkpoints', False):
            cpfs = sorted(glob.glob(self.checkpoint_file_base + '*'))
            for f in cpfs:
                self.logger.info(f'Removing checkpoint file {f}')
                os.remove(f)

    def save_checkpoint(self):
        r""" Saves a checkpoint file.  Uses the save method, which a user can override.  User should not override
        this method. """
        if self.checkpoint_file_base is None:
            return
        cframe = self.processing_index
        if not (((cframe+1) % self.checkpoint_interval == 0) or (cframe == self.n_chunk - 1)):
            return
        self.logger.debug("Processing checkpoint")
        self.logger.debug(f'Previous checkpoint file: {self.previous_checkpoint_file}')
        cpf = self.checkpoint_file_base + f'_{cframe + 1:07d}'
        self.logger.info(f'Saving checkpoint file {cpf}')
        self.save(cpf)
        if self.previous_checkpoint_file is not None:
            self.logger.info(f'Removing previous checkpoint file {self.previous_checkpoint_file}')
            if os.path.exists(self.previous_checkpoint_file):
                os.remove(self.previous_checkpoint_file)
        self.previous_checkpoint_file = cpf

    def load_checkpoint(self):
        r""" Loads a checkpoint file.  Uses the load method, which a user can override.  User should not override
        this method. """
        if self.checkpoint_file_base:
            self.logger.info(f"Seeking checkpoint files {self.checkpoint_file_base}*")
            cpfs = sorted(glob.glob(self.checkpoint_file_base + '*'))
            self.logger.info(f"Found {len(cpfs)} possible checkpoints")
            while len(cpfs) > 0:
                c = cpfs.pop()
                try:
                    self.logger.info(f'Loading checkpoint file {c}')
                    stats = self.load_file(c)
                    if self.start != stats['start'] or self.stop != stats['stop'] or self.step != stats['step']:
                        self.logger.warning('The start/stop/step of the checkpoint are mismatched with this job')
                    idx = int(c.split('_')[-1])
                    self.from_dict(stats)
                    self.logger.info(f'Starting at frame {idx}')
                    self.processing_index = idx
                    break
                except Exception as e:
                    self.logger.warning(f"Problem loading file {c}")

    def add_frame(self, dat):
        pass

    def clear_data(self):
        pass

    def initialize_data(self):
        pass

    def to_dict(self):
        return dict()

    def from_dict(self, stats):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def concatenate(self, stats):
        pass

    def process_frames(self):
        if self.parallel:
            return self.process_parallel()
        self.logger.info(f"Global start frame: {self.start}")
        self.logger.info(f"Global stop frame: {self.stop}")
        self.logger.info(f"Global step size: {self.step}")
        frame_ids = np.arange(self.start, self.stop, self.step, dtype=int)
        frame_ids = np.array_split(frame_ids, self.n_processes)[self.process_id - 1]
        t0 = time.time()
        self.n_processed = 0
        self.processing_index = 0
        self.framegetter_index = 0
        self.n_chunk = len(frame_ids)
        self.logger.info(f"Total frames for this process: {self.n_chunk}")
        if self.n_chunk == 0:
            return None
        self.load_checkpoint()  # This will fast forward if possible.  Affects processing indices
        fpsf = 0  # Frames processed so far (not counting those restored from checkpoint)
        ftp = self.n_chunk - self.processing_index  # Total frames to process (not counting
        for n in range(self.processing_index, self.n_chunk):
            self.processing_index = n
            self.framegetter_index = frame_ids[n]
            fpsf += 1
            fg_idx = frame_ids[n]
            dt = time.time() - t0  # Total processing time so far
            atpf = dt / fpsf  # Average time per frame
            tr = atpf*(ftp - fpsf)  # Time remaining
            freq = 1/atpf if atpf > 0 else 0
            self.logger.info(f"Frame ID {fg_idx} (# {n+1} of {self.n_chunk}) - {freq:.2f} Hz => {tr / 60:.1f} min. "
                             f"remaining")
            dat = self.framegetter.get_frame(frame_number=fg_idx)
            if dat is None:
                self.logger.warning('Frame is None')
            self.add_frame(dat)
            self.n_processed += 1
            self.save_checkpoint()
        self.logger.info('Processing completed')
        if self.reduce_from_checkpoints:
            return self.previous_checkpoint_file
        return self.to_dict()

    @staticmethod
    def process(framegetter=None, start=0, stop=None, step=1, parallel=False, n_processes=1, process_id=0, config=None):
        ps = cls(framegetter=framegetter, start=start, stop=stop, step=step, parallel=parallel,
                  n_processes=n_processes, process_id=process_id, config=config)
        ps.process_frames()
        return ps.to_dict()

    def process_parallel(self):
        self.logger.info(f"Launching {self.n_processes} parallel processes")
        n = self.n_processes
        fg = self.framegetter_dict
        conf = self.config
        out = Parallel(n_jobs=n)(delayed(self.process)(framegetter=fg, start=self.start, stop=self.stop, step=self.step,
                parallel=False, n_processes=n, process_id=i+1, config=conf) for i in range(n))
        self.logger.info(f"Compiling results from {self.n_processes} processes")
        self.clear_data()
        for i in range(self.n_processes):
            stats = out[i]
            if stats is None:
                self.logger.info(f"No results from process {i}")
                continue
            if isinstance(stats, str):
                self.logger.info(f"Loading checkpoint file {stats}")
                stats = self.load_dictionary(stats)
            if stats is None:
                self.logger.info(f"No results from process {i}")
                continue
            # if stats['n_frames'] == 0:
            #     self.logger.info(f"No results from process {i}")
            #     continue
            self.logger.info(f"Concatenating results from process {i} ({stats['n_frames']} frames).")
            self.concatenate(stats)
        d = self.to_dict()
        self.logger.info(f"{d['n_frames']} frames combined.")
        self.logger.info(f"Done")
        return d
