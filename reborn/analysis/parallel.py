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
import glob
import logging
import numpy as np
import os
import random
import sys
import time
from abc import ABC, abstractmethod
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None
from .. import fileio
from ..dataframe import DataFrame


def get_setup_data(**kwargs):
    beam = kwargs.get('beam', None)
    pad_geometry = kwargs.get('pad_geometry', None)
    max_iterations = kwargs.get('max_iterations', 1e4)
    framegetter = kwargs.get('framegetter', None)
    initial_frame = None
    if framegetter is None:
        raise ValueError('get_setup_data requires a framegetter')
    if (pad_geometry is None) or (beam is None):
        frames = random.sample(range(max_iterations), max_iterations)
        for i in frames:
            data = framegetter.get_frame(frame_number=i)
            if data.validate():
                initial_frame = i
                pad_geometry = data.get_pad_geometry()
                beam = data.get_beam()
                break
    return pad_geometry, beam, initial_frame


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
        r""" Set up the framegetter.  If running in parallel then we need to prepare a dictionary that allows the
        framegetter to be created within another process.  If not, then we might need to utilize said dictionary
        to create a framegetter instance. """
        if callable(framegetter):
            framegetter = framegetter()
        self.framegetter = framegetter

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
        if Parallel is None:
            raise ImportError('You need the joblib package to run padstats in parallel mode.')
        framegetter = self.framegetter.factory()
        self.logger.info(f"Launching {self.n_processes} parallel processes")
        n = self.n_processes
        kwargs = dict(framegetter=framegetter, start=self.start, stop=self.stop, step=self.step,
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
