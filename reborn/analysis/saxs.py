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
import os
import sys
import glob
import time
import logging
import numpy as np
try:
    from joblib import delayed
    from joblib import Parallel
except ImportError:
    delayed = None
    Parallel = None
from ..detector import RadialProfiler
from ..external import pyqtgraph as pg
from .. import utils, fileio

debug = False


def debug_message(*args, caller=True, **kwargs):
    r""" Standard debug message, which includes the function called. """
    if debug:
        s = ''
        if caller:
            s = utils.get_caller(1)
        print('DEBUG:'+s+':', *args, **kwargs)


def get_profile_stats(dataframe, n_bins, q_range, include_median=False):
    r"""
    Operates on one raw diffraction pattern and returns a dictionary with the following:
    
        mean :   Mean of unmasked intensities
        sdev :   Standard deviation of unmasked intensities
        median : Median of unmasked intensities (only if requested; this is slow)
        sum :    Sum of unmasked intensities
        sum2 :   Sum of squared unmasked intensities
        count :  Number of unmasked pixels in the q bin
        q_bins : The centers of the q bins

    Prior to computing the above, the following steps are followed:

        1) The pattern is divided by the polarization factor.
        2) The pattern is divided by relative solid angles

    It is assumed that the mask in the dataframe is good -- i.e. that you have removed
    outliers already with e.g. an SNR transform or other outlier rejection scheme.

    Arguments:
        dataframe (DataFrame): A reborn dataframe instance. Has raw data, geometry, beam, etc.
        n_bins (float): Number of q bin in radial profile.
        q_range (list-like): The minimum and maximum of the centers of the q bins.
        include_median (bool): Toggle the inclusion of median profiles (default: False)
    
    Returns:
        dict
    """
    beam = dataframe.get_beam()
    geom = dataframe.get_pad_geometry().copy()
    debug_message('gathering data')
    data = dataframe.get_raw_data_flat()
    mask = dataframe.get_mask_flat()
    pfac = dataframe.get_polarization_factors_flat()
    sa = dataframe.get_solid_angles_flat()
    data /= pfac * sa * 1e6  # normalize our the polarization factors, microsteradian units for solid angles
    debug_message('computing profiles')
    profiler = RadialProfiler(pad_geometry=geom, mask=mask, beam=beam,
                              n_bins=n_bins, q_range=q_range)
    stats = profiler.quickstats(data)
    out = dict()
    out['mean'] = stats['mean']
    out['sdev'] = stats['sdev']
    out['sum'] = stats['sum']
    out['sum2'] = stats['sum2']
    out['counts'] = stats['weight_sum']
    out['q_bins'] = profiler.q_bin_centers
    out['frame_id'] = dataframe.get_frame_id()
    if include_median:
        out['median'] = profiler.get_median_profile(data)
    return out


def get_profile_runstats(framegetter=None, n_bins=1000, q_range=None,
                         start=0, stop=None, parallel=False,
                         n_processes=None, process_id=None,
                         include_median=False, verbose=1):
    r""" 
    Parallelized version of get_profile_stats.

    You should be able to pass in any FrameGetter subclass.
    You can try the parallel flag if you have joblib package installed... but if you parallelize, please understand that you
    cannot pass a framegetter from the main process to the children processes (because, for example, the framegetter
    might have a reference to a file handle object). Therefore, in order to parallelize, we use the convention in
    which the framegetter is passed in as a dictionary with the 'framegetter' key set to the desired FrameGetter
    subclass, and the 'kwargs' key set to the keyword arguments needed to create a new class instance.

    Returns:
        dict
    """
    def message(*args, **kwargs):
        if verbose >= 1:
            print(*args, **kwargs)
    if framegetter is None:
        raise ValueError('framegetter cannot be None')
    if parallel and n_processes > 1:
        debug_message('Begin parallelized processing.')
        if Parallel is None:
            raise ImportError('You need the joblib package to run in parallel mode.')
        if not isinstance(framegetter, dict):
            if framegetter.init_params is None:
                raise ValueError('This FrameGetter does not have init_params attribute needed to make a replica')
            framegetter = {'framegetter': type(framegetter), 'kwargs': framegetter.init_params}
        pout = Parallel(n_jobs=n_processes)(delayed(get_profile_runstats)(framegetter=framegetter,
                                                                         start=start,
                                                                         stop=stop,
                                                                         parallel=False,
                                                                         n_processes=n_processes,
                                                                         process_id=i,
                                                                         include_median=include_median)
                                                                         for i in range(n_processes))
        out = dict()
        out['mean'] = np.concatenate([o['mean'] for o in pout])
        out['sdev'] = np.concatenate([o['sdev'] for o in pout])
        out['sum'] = np.concatenate([o['sum'] for o in pout])
        out['sum2'] = np.concatenate([o['sum2'] for o in pout])
        out['counts'] = np.concatenate([o['counts'] for o in pout])
        out['q_bins'] = np.concatenate([o['q_bins'] for o in pout])
        out['frame_ids'] = np.concatenate([o['frame_ids'] for o in pout])
        if include_median:
            out['median'] = np.concatenate([o['median'] for o in pout])
        return out
    if isinstance(framegetter, dict):
        framegetter = framegetter['framegetter'](**framegetter['kwargs'])
    if stop is None:
        stop = framegetter.n_frames
    stop = min(stop, framegetter.n_frames)
    start = max(0, start)
    frame_numbers = np.arange(start, stop, dtype=int)
    if process_id is not None:
        frame_numbers = np.array_split(frame_numbers, n_processes)[process_id]
    pmean = np.zeros((frame_numbers.size, n_bins))
    psdev = np.zeros((frame_numbers.size, n_bins))
    psum = np.zeros((frame_numbers.size, n_bins))
    psum2 = np.zeros((frame_numbers.size, n_bins))
    pcounts = np.zeros((frame_numbers.size, n_bins))
    pq_bin = np.zeros((frame_numbers.size, n_bins))
    frame_ids = []
    if include_median:
        pmedian = np.zeros((frame_numbers.size, n_bins))
    nf = len(frame_numbers)
    t0 = time.time()
    for (n, i) in enumerate(frame_numbers):
        ts = time.ctime()
        dt = time.time() - t0
        tr = dt*(nf-n)/(n+1)
        print(f'{ts}: Process {process_id:2d}: Frame {i:6d} ({n:6d} of {nf:6d}): {n/nf*100:0.2g}% @'
              f' {dt/60:.1f} min. => {tr/60:.1f} min. remaining')
        # message(f'Process {process_id:2d}; Frame {i:6d}; {n:6d} of {nf:6d}; {n / nf * 100:0.2g}% complete.')
        dat = framegetter.get_frame(frame_number=i)
        if dat is None:
            print(f'{ts}: Frame {i:6d} is None!!!')
            continue
        pstats = get_profile_stats(dataframe=dat, n_bins=n_bins, q_range=q_range, include_median=include_median)
        pmean[n, :] = pstats['mean']
        psdev[n, :] = pstats['sdev']
        psum[n, :] = pstats['sum']
        psum2[n, :] = pstats['sum2']
        pcounts[n, :] = pstats['counts']
        pq_bin[n, :] = pstats['q_bins']
        frame_ids.append(dat.get_frame_id())
        if include_median:
            pmedian[n, :] = pstats['median']
    out = dict()
    out['mean'] = pmean
    out['sdev'] = psdev
    out['sum'] = psum
    out['sum2'] = psum2
    out['counts'] = pcounts
    out['q_bins'] = pq_bin
    out['frame_ids'] = np.array(frame_ids)
    if include_median:
        out['median'] = pmedian
    return out


def save_profiles(stats, filepath):
    fileio.misc.save_pickle(stats, filepath)


def load_profiles(filepath):
    stats = fileio.misc.load_pickle(filepath)
    with np.errstate(divide='ignore', invalid='ignore'):
        meen = stats['sum']/stats['counts']
        meen2 = stats['sum2']/stats['counts']
        sdev = np.nan_to_num(meen2-meen**2, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        sdev[sdev < 0] = 0
        sdev = np.sqrt(sdev)
    stats['mean'] = meen
    stats['sdev'] = sdev
    return stats


def view_profile_runstats(stats):
    q = stats['q_bins'][0]
    labels = ['Mean', 'Standard Deviation']
    images = np.array([stats['mean'], stats['sdev']])
    pg.imview(images, fs_label='q [A]', ss_label='frame #', fs_lims=[q[0]/1e10, q[-1]/1e10], hold=True,
              frame_names=labels)


def normalize_profile_stats(stats, q_range=None):
    r"""
    Given a stats dictionary, normalize by setting a particular q range to an average of 1.

    Arguments:
        stats (dict): A stats dictionary created by get_profile_stats_from_pandas()
        q_range (tuple): Optionally specify the normalization range with a tuple containing (q_min, q_max)

    Returns:
        dict: An updates stats dictionary
    """
    
    out_keys = ['median', 'mean', 'sum', 'sum2', 'counts', 'q_bins']
    q = stats['q_bins'][0, :]
    if q_range is None:
        q_range = (np.min(q), np.max(q))
    run_pmedian = stats['median'].copy()
    run_pmean = stats['mean'].copy()
    run_psdev = stats['sdev'].copy()
    run_psum = stats['sum'].copy()
    run_psum2 = stats['sum2'].copy()
    qmin = q_range[0]
    qmax = q_range[1]
    w = np.where((q > qmin) * (q < qmax))
    s = np.mean(run_pmean[:, w[0]], axis=1)
    out_vals = [(run_pmedian.T / s).T, (run_pmean.T / s).T,
                (run_psdev.T / s).T, (run_psum.T / s).T,
                (run_psum2.T / s ** 2).T, stats["counts"].copy(),
                stats["q_bins"].copy()]
    return dict(zip(out_keys, out_vals))


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





        """
        self.analyzer_name = 'ParallelAnalyzer'  # Re-define this to something more sensible
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
    def initialize_data(self, rdat):
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
            return self.previous_checkpoint_file
        return self.to_dict()
    @staticmethod
    def process(framegetter=None, start=0, stop=None, step=1, parallel=False, n_processes=1, process_id=0, config=None):
        cls = self.type
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
