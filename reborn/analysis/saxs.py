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
