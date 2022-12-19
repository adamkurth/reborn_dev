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
from abc import ABC, abstractmethod
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


class ParallelAnalyzer(ABC):

    super_initialized = False

    def __init__(self, framegetter=None, config=None, **kwargs):
        r""" A skeleton for parallel processing of datasets with logging and checkpoints.  This class is only useful
        if each frame is processed independently.  The normal use case is to accumulate results from many frames in a
        run.  You must create a subclass as follows:

        - Put all needed configuration parameters into a single dictionary and provide the dictionary on instantiation.
        - Define the **to_dict** method, which puts all information needed to restore the state of analysis into a
          dictionary.  See method docs for more details.
        - Define the **from_dict** method, which restores the state of analysis.  See method docs for more detail.
        - Define the **add_frame** method.  This is the core of the processing pipeline; it does all needed actions
          associated with the addition of a |DataFrame| to the compiled results.  See method docs for more detail.
        - Define the **concatenate** method, which combines data from different chunks handled by different processors.
        - Optionally define the **finalize** method that will run after all processing/concatenation is complete.
        - At the beginning of your __init__, include the following line:
          `super().__init__(framegetter=framegetter, config=config, **kwargs)`  This is essential -- your subclass will
          not function properly without this line.  The initialization of the base class handles all the configurations
          associated with logging, checkpoints, and parallel processing.

        The following config dictionary entries affect parallelization, logging and checkpoints:

        - **debug**: Set to true to produce debug messages in log files.
        - **log_file**: The base filepath for log files (e.g. "/results/logs/run0010.log").  Processor IDs
          will be appended as necessary.
        - **message_prefix**: Prefix to log messages.  For example: "Run 7: "
        - **clear_logs**: Log files keep growing by default.  Set this to true if you want them to be cleared.
        - **checkpoint_file**: The base filepath for checkpoint files (e.g.
          "/results/checkpoints/run0010.pkl").  Processor IDs will be appended as necessary.  If this is set to None,
          no checkpoints will be created, and parallel processing might fail.  Be careful to ensure that you will not
          be processing different runs with the same checkpoint files!!!
        - **checkpoint_interval**: How often to save checkpoints.
        - **clear_checkpoints**: Set this to true if you want to remove all stale checkpoints.  Be very careful
          with paths if you do this.  You might wipe out something important...

        Important numbers that you might use but should not modify:

        - **self.n_processed**: Counts the number of good frames processed (for which |DataFrame| is not None)
        - **self.processing_index**: Indicates the current raw index for the given analysis chunk.  Starts at zero for
          each process.
        - **self.framegetter_index**: Indicates the current framegetter index.  E.g. if you start at frame 10,
          this starts at 10 for worker process 1.  It will start elsewhere for the other worker processes.
        """
        self.super_initialized = True
        self.analyzer_name = None  # Re-define this to something more sensible
        self.start = kwargs.get('start', 0)  # Global start for the full run/framegetter
        self.stop = kwargs.get('stop', None)  # Global stop point for the full run/framegetter
        self.step = kwargs.get('step', 1)  # Global step size for the full run/framegetter
        self.parallel = kwargs.get('parallel', True)
        self.n_processes = kwargs.get('n_processes', 1)
        if self.n_processes < 2:
            self.parallel = False
        self.process_id = kwargs.get('process_id', 0)
        self.config = config
        self.logger = None
        self._setup_logger()
        self.framegetter = None
        self.framegetter_dict = None  # Needed to create replicas of the framegetter in sub-processes
        self._setup_framegetter(framegetter)
        self.current_checkpoint_number = 0
        self.previous_checkpoint_file = None
        self.checkpoint_interval = None
        self.checkpoint_file_base = None
        self.reduce_from_checkpoints = None  # Reduce/concatenate data by first saving to disk (minimize memory use)
        self._setup_checkpoints()
        self.initialized = False
        self.n_chunk = None  # Total frames expected for this chunk of the run (with possible bad frames)
        self.n_processed = 0  # Number of frames actually processed contributing to the stats (not counting bad frames)
        if self.stop is None:
            self.stop = self.framegetter.n_frames
        self.stop = min(self.stop, self.framegetter.n_frames)
        self.processing_index = 0
        self.framegetter_index = 0

    @abstractmethod
    def add_frame(self, dat: DataFrame):
        r""" User-defined method that does all actions associated with the addition of one |DataFrame| to the
        results.  You should probably add a test to determine if any initializations are needed, e.g. in
        the event that empty arrays must be pre-allocated or counters need to be set to zero.  """
        pass

    @abstractmethod
    def to_dict(self):
        r""" User-defined method that compiles all relevant data into a dictionary.  This will be used to save the
        state of the analysis in checkpoint files, so be sure that all information that is needed to fast-forward the
        processing to an intermediate state is provided.  You should probably include the config dictionary in this
        file so you know how the analysis was configured.  """
        return dict()

    @abstractmethod
    def from_dict(self, stats):
        r""" User-defined method that is complementary to the to_dict method.  Given a dictionary produced by
        from_dict, this method must take all necessary actions to restore a given analysis state. """
        pass

    @abstractmethod
    def concatenate(self, stats):
        r""" User-defined method that combines an existing instance of ParallelAnalyzer with the results of another
        ParallelAnalyzer that has operated on a different chunk of DataFrames.   """
        pass

    def finalize(self):
        r""" Optional user-defined method that will be called at the end of all processing, after concatenation and
        immediately before a final results dictionary is returned or saved to disk. """
        pass

    def _setup_logger(self):
        r""" Setup logger.  This is affected by the config dictionary keys 'debug', 'message_prefix', 'log_file' """
        # Sometimes we want to prefix a run number or experiment ID (for example).
        message_prefix = self.config.get("message_prefix", None)
        if message_prefix is None:
            print("Provide a message prefix by adding the 'message_prefix' to the config dictionary.")
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
            logger.info(f"No logfile specified.  Specify it by adding 'log_file' to the config dictionary.")

    def _setup_framegetter(self, framegetter):
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

    def _setup_checkpoints(self):
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
        checkpoint_interval = self.config.get('checkpoint_interval', None)
        if checkpoint_interval is None:
            checkpoint_interval = 250
            self.logger.info(f"Checkpoint interval will be set to {checkpoint_interval}.  You may choose a different "
                             f"value by setting the 'checkpoint_interval' key in the config dictionary.")
        if checkpoint_file:
            logging.info(f"Checkpoint file base: {checkpoint_file}, Interval: {checkpoint_interval}")
        else:
            logging.info(f"No checkpoint file name is specified.  Checkpoints are disabled.  Enable by setting the "
                         f"'checkpoint_file' key in the config dictionary.")
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_file_base = checkpoint_file
        if self.config.get('clear_checkpoints', False) and checkpoint_file:
            cpfs = sorted(glob.glob(self.checkpoint_file_base + '*'))
            for f in cpfs:
                self.logger.info(f'Removing checkpoint file {f}')
                os.remove(f)

    def _save_checkpoint(self):
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

    def _load_checkpoint(self):
        r""" Loads a checkpoint file.  Uses the load method, which a user can override.  User should not override
        this method. """
        if self.checkpoint_file_base:
            self.logger.info(f"Seeking checkpoint files {self.checkpoint_file_base}*")
            cpfs = sorted(glob.glob(self.checkpoint_file_base + '*'))
            self.logger.info(f"Found {len(cpfs)} possible checkpoints")
            while len(cpfs) > 0:
                c = cpfs.pop()
                # try:
                self.logger.info(f'Loading checkpoint file {c}')
                stats = self.load(c)
                if self.start != stats['start'] or self.stop != stats['stop'] or self.step != stats['step']:
                    self.logger.warning('The start/stop/step of the checkpoint are mismatched with this job')
                idx = int(c.split('_')[-1])
                self.from_dict(stats)
                self.logger.info(f'Starting at frame {idx}')
                self.processing_index = idx
                break
                # except Exception as e:
                #     print(e)
                #     self.logger.warning(f"Problem loading file {c}")

    def save(self, filepath):
        r""" Saves dictionary (produced by the to_dict method) as a pickle file.  You may wish to override this
        method if you prefer to save in a different format (e.g. hdf5).  """
        d = self.to_dict()
        d['start'] = self.start
        d['stop'] = self.stop
        d['step'] = self.step
        d['n_processed'] = self.n_processed
        fileio.misc.save_pickle(d, filepath)

    def load(self, filepath):
        r""" Loads pickled dictionary, as defined by the to_dict method.  If you override this method, be sure that
        it matches with the save method. """
        return fileio.misc.load_pickle(filepath)

    def process_frames(self):
        r""" Process all dataframes.  Will launch parallel processes if n_processes is greater than 1.

        Returns either a dictionary of results, or a string that indicates the path to the results file.  In the
        case of parallel processing, each worker process creates a file containing results, and the main process then
        combines all of the worker results vie the concatenate method."""
        if not self.super_initialized:
            raise Exception('Super was not initialized.  Subclass needs the line super().__init__('
                            'framegetter=framegetter, config=config, **kwargs) at the beginning of __init__')
        if self.parallel:
            return self._process_parallel()
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
        self._load_checkpoint()  # This will fast forward if possible.  Affects processing indices
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
            self._save_checkpoint()
        self.logger.info('Processing completed')
        if self.reduce_from_checkpoints:
            return self.previous_checkpoint_file
        self.finalize()
        return self.to_dict()

    @staticmethod
    def _worker(ana, **kwargs):
        ps = ana(**kwargs)
        ps.process_frames()
        return ps.to_dict()

    def _process_parallel(self):
        self.logger.info(f"Launching {self.n_processes} parallel processes")
        n = self.n_processes
        kwargs = dict(framegetter=self.framegetter_dict, start=self.start, stop=self.stop, step=self.step,
                      n_processes=self.n_processes, config=self.config, parallel=False)
        out = Parallel(n_jobs=n)(delayed(self._worker)(type(self), process_id=i+1, **kwargs) for i in range(n))
        self.logger.info(f"Compiling results from {self.n_processes} processes")
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
        self.finalize()
        d = self.to_dict()
        self.logger.info(f"{d['n_frames']} frames combined.")
        self.logger.info(f"Done")
        return d


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
    # @staticmethod
    # def get_default_config(histogram=True):
    #     hp = None
    #     if histogram:
    #         hp = dict(bin_min=-30, bin_max=100, n_bins=100, zero_photon_peak=0, one_photon_peak=30)
    #     return dict(log_file=None, checkpoint_file=None, checkpoint_interval=500, message_prefix="", debug=True,
    #               reduce_from_checkpoints=True, histogram_params=hp)
    def setup_histogram(self):
        if self.histogram_params is not None:
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
            self.histogrammer.add_frame(rdat)  #, mask=dat.get_mask_flat())
    def finalize(self):
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


def padstats(framegetter=None, start=0, stop=None, step=None, n_processes=1, config=None):
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
# def analyze_histogram(stats, n_processes=1, debug=0):
#     r""" Analyze histogram and attempt to extract offsets and gains from the zero- and one-photon peak.  Experimental.
#     Use at your own risk!"""
#     def dbgmsg(*args, **kwargs):
#         if debug:
#             print(*args, **kwargs)
#     if n_processes > 1:
#         if Parallel is None:
#             raise ImportError('You need the joblib package to run in parallel mode.')
#         stats_split = [dict(histogram=h) for h in np.array_split(stats['histogram'], n_processes, axis=0)]
#         for s in stats_split:
#             s['histogram_params'] = stats['histogram_params']
#         out = Parallel(n_jobs=n_processes)([delayed(analyze_histogram)(s, debug=debug) for s in stats_split])
#         return dict(gain=np.concatenate([out[i]['gain'] for i in range(n_processes)]),
#                     offset=np.concatenate([out[i]['offset'] for i in range(n_processes)]))
#     mn = stats['histogram_params']['bin_min']
#     mx = stats['histogram_params']['bin_max']
#     nb = stats['histogram_params']['n_bins']
#     c00 = stats['histogram_params'].get('zero_photon_peak', 0)
#     c10 = stats['histogram_params'].get('one_photon_peak', 30)
#     x = np.linspace(mn, mx, nb)
#     histdat = stats['histogram']
#     poly = np.polynomial.Polynomial
#     n_pixels = histdat.shape[0]
#     gain = np.zeros(n_pixels)
#     offset = np.zeros(n_pixels)
#     for i in range(n_pixels):
#         c0 = c00
#         c1 = c10
#         a = (c1 - c0) / 3
#         o = 5
#         goodfit = 1
#         for j in range(2):
#             w0 = np.where((x >= c0-a) * (x <= c0+a))
#             w1 = np.where((x >= c1-a) * (x <= c1+a))
#             x0 = x[w0]
#             x1 = x[w1]
#             y0 = histdat[i, :][w0]
#             y1 = histdat[i, :][w1]
#             if np.sum(y0) < o:
#                 dbgmsg('skip')
#                 goodfit = 0
#                 break
#             if np.sum(y1) < o:
#                 dbgmsg('skip')
#                 goodfit = 0
#                 break
#             if len(y0) < o:
#                 dbgmsg('skip')
#                 goodfit = 0
#                 break
#             if len(y1) < o:
#                 dbgmsg('skip')
#                 goodfit = 0
#                 break
#             f0, extra = poly.fit(x0, y0, o, full=True)
#             xf0, yf0 = f0.linspace()
#             c0 = xf0[np.where(yf0 == np.max(yf0))[0][0]]
#             f1, extra = poly.fit(x1, y1, o, full=True)
#             xf1, yf1 = f1.linspace()
#             c1 = xf1[np.where(yf1 == np.max(yf1))[0][0]]
#             a = 5
#             o = 3
#         if goodfit:
#             gain[i] = c1-c0
#             offset[i] = c0
#         dbgmsg(f"Pixel {i} of {n_pixels} ({i*100/float(n_pixels):0.2f}%), gain={gain[i]}, offset={offset[i]}")
#     return dict(gain=gain, offset=offset)
