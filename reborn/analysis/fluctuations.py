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

import numpy as np
import reborn.dataframe
from joblib import delayed
from joblib import Parallel


def correlate(s1, s2=None, cached=False):
    r"""
    Computes correlation function.
    If two signals are provided computes cross correlation.
    If one singal is provided computes auto correlation.
    If cached, assumes Fourier transforms are already computed.

    Computed via Fourier transforms:
        cf = iFT(FT(s1) FT(s2)*)

    Arguments:
        s1 (|ndarray|): signal 1
        s2 (|ndarray|): signal 2
        cached (bool): provide ffts instead of computing
    Returns:
        correlation (|ndarray|): correlation of s1 and s2
    """
    if not cached:
        s1 = np.fft.fft(s1, axis=1)
        if s2 is not None:
            s2 = np.fft.fft(s2, axis=1)
    if s2 is None:
        s2 = s1.copy()
    cor_fft = s1 * s2.conj()
    correlation = np.fft.ifft(cor_fft, axis=1)
    return np.real(correlation)


def subtract_masked_data_mean(data, mask):
    r"""
    Subtract average q for each q ring in data, ignores masked pixels.
    This normalizes and centers the data around 0.

    Arguments:
        data (|ndarray|): data
        mask (|ndarray|): mask (i.e. data to ignore)
    Returns:
        data (|ndarray|): data - <data>_q
    """
    data[mask == 0] = 0
    d_sum = np.sum(data, axis=1)
    count = np.sum(mask, axis=1)
    zs = np.zeros_like(d_sum, dtype=float)
    data_avg = np.divide(d_sum, count, out=zs, where=count != 0)
    d = (data.T - data_avg).T
    d[mask == 0] = 0  # re-zero masked pixels
    return d


def data_correlation(n, data, mask, cached=False):
    r"""
    Computes cross correlation of data with data shifted by n.

    Note: For n = 0 this returns the auto correlation of the data.

    Arguments:
        n (int): number of q rings to shift
        data (|ndarray|): data
        mask (|ndarray|): mask (i.e. data to ignore)
        cached (bool): provide ffts instead of computing
    Returns:
        ccf (|ndarray|): cross correlation of data
    """
    data[mask == 0] = 0
    d_roll = None
    m_roll = None
    if not cached:
        data = subtract_masked_data_mean(data=data, mask=mask)
    if n > 0:
        d_roll = np.roll(data, shift=n, axis=0)
        m_roll = np.roll(mask, shift=n, axis=0)
    d_cf = correlate(s1=data, s2=d_roll, cached=cached)
    m_cf = correlate(s1=mask, s2=m_roll, cached=cached)
    zs = np.zeros_like(data, dtype=float)
    return np.divide(d_cf, m_cf, out=zs, where=m_cf != 0)


def compute_data_correlations(data, mask):
    r"""
    Computes cross correlation of data with data shifted by n.

    Note: For n = 0 this returns the auto correlation of the data.

    Arguments:
        data (|ndarray|): data
        mask (|ndarray|): mask (i.e. data to ignore)
    Returns:
        correlations (dict): correlations of data
    """
    data[mask == 0] = 0
    q_range = data.shape[0]
    data = subtract_masked_data_mean(data=data, mask=mask)
    d = np.fft.fft(data, axis=1)
    m = np.fft.fft(mask, axis=1)
    return [data_correlation(n=n, data=d, mask=m, cached=True) for n in range(q_range)]


class FXS:
    _experiment_id = None
    _frame_beam = None
    _frame_data = None
    _frame_geom = None
    _frame_mask = None
    _frame_correlations = None
    _frame_polar_data = None
    _frame_polar_mask = None
    _n_patterns = None
    _polar_assembler = None
    _radial_profiler = None
    _run_id = None
    _run_max = None
    _run_min = None
    _run_sum = None
    _run_sum2 = None
    _run_sum_correlations = None

    def __init__(self, experiment_id=None, run_id=None, polar_assembler=None, radial_profiler=None):
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._polar_assembler = polar_assembler
        self._radial_profiler = radial_profiler
        self._n_patterns = 0

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def run_id(self):
        return self._run_id

    @property
    def n_patterns(self):
        return self._n_patterns

    @property
    def run_sum_correlations(self):
        return self._run_sum_correlations

    @property
    def run_max(self):
        return self._run_max

    @property
    def run_min(self):
        return self._run_min

    @property
    def run_sum(self):
        return self._run_sum

    @property
    def run_sum2(self):
        return self._run_sum2

    @property
    def run_var(self):
        return np.array([rs2 - rs ** 2 for rs, rs2 in zip(self._run_sum, self._run_sum2)])

    def __str__(self):
        out = "Fluctuation X-ray Scattering\n"
        out += f"    Experiment ID: {self.experiment_id}\n"
        out += f"           Run ID: {self.run_id}\n"
        if self._polar_assembler is None:
            out += f"  Polar Assembler: None\n"
        else:
            out += f"  Polar Assembler: Assigned\n"
        if self._radial_profiler is None:
            out += f"  Radial Profiler: None\n"
        else:
            out += f"  Radial Profiler: Assigned\n"
        out += f"Averaged Patterns: {self.n_patterns}\n"
        return out

    def add_frame(self, dataframe):
        if isinstance(dataframe, reborn.dataframe.DataFrame):
            if dataframe.validate():
                print("Dataframe is valid, adding to analysis ...")
                self._n_patterns += 1
                self._frame_geom = dataframe.get_pad_geometry()
                self._frame_beam = dataframe.get_beam()
                self._frame_data = dataframe.get_raw_data_list()
                self._frame_mask = dataframe.get_mask_list()

                self._frame_polar_data, self._frame_polar_mask = self._polar_assembler.get_mean(data=self._frame_data,
                                                                                                mask=self._frame_mask)
                self._frame_correlations = compute_data_correlations(self._frame_polar_data,
                                                                     self._frame_polar_mask)
                if self._run_sum_correlations is None:
                    self._run_sum_correlations = self._frame_correlations
                else:
                    for i, fc in enumerate(self._frame_correlations):
                        self._run_sum_correlations[i] += fc

                if self._run_max is None:
                    self._run_max = self._frame_data
                else:
                    self._run_max = np.maximum(self._run_max, self._frame_data)

                if self._run_min is None:
                    self._run_min = self._frame_data
                else:
                    self._run_min = np.minimum(self._run_min, self._frame_data)

                if self._run_sum is None:
                    self._run_sum = self._frame_data
                else:
                    for i, fd in enumerate(self._frame_data):
                        self._run_sum[i] += fd

                if self._run_sum2 is None:
                    self._run_sum2 = [fd ** 2 for fd in self._frame_data]
                else:
                    for i, fd in enumerate(self._frame_data):
                        self._run_sum2[i] += fd ** 2
            else:
                print("Dataframe not valid, skipping ...")
        else:
            print("Only dataframe addition is supported at this time.")

    def merge_fxs(self, fxso):
        if isinstance(fxso, FXS):
            self._n_patterns += fxso.n_patterns

            if self._run_sum_correlations is None:
                self._run_sum_correlations = fxso.run_sum_correlations
            else:
                for i, c in enumerate(fxso.run_sum_correlations):
                    self._run_sum_correlations[i] += c

            if self._run_max is None:
                self._run_max = fxso.run_max
            else:
                self._run_max = np.maximum(self._run_max, fxso.run_max)

            if self._run_min is None:
                self._run_min = fxso.run_min
            else:
                self._run_min = np.minimum(self._run_min, fxso.run_min)

            if self._run_sum is None:
                self._run_sum = fxso.run_sum
            else:
                for i, fd in enumerate(fxso.run_sum):
                    self._run_sum[i] += fd

            if self._run_sum2 is None:
                self._run_sum2 = fxso.run_sum2
            else:
                for i, fd in enumerate(fxso.run_sum2):
                    self._run_sum2[i] += fd
        else:
            print("Only merging FXS objects is supported at this time.")

    def get_run_correlations(self):
        return [c / self._n_patterns for c in self._run_sum_correlations]

    def get_auto_correlations(self):
        return self._run_sum_correlations[0] / self._n_patterns

    def get_saxs(self, pattern, mask=None, statistic=None):
        if self._radial_profiler is None:
            p = None
        else:
            p = self._radial_profiler.quickstats(data=pattern, weights=mask)
            if statistic is not None:
                p = p[statistic]
        return p

    def to_dict(self):
        fxs_dict = {'experiment_id': self.experiment_id,
                    'run_id': self.run_id,
                    'analysis/n_patterns': self.n_patterns,
                    'analysis/run_max': self.run_max,
                    'analysis/run_min': self.run_min,
                    'analysis/run_sum': self.run_sum,
                    'analysis/run_sum2': self._run_sum2,
                    'analysis/geometry/n_q_bins': self._polar_assembler.n_q_bins,
                    'analysis/geometry/n_phi_bins': self._polar_assembler.n_phi_bins}
        qs = self._polar_assembler.q_mags
        phis = self._polar_assembler.phis
        fxs_dict.update({'analysis/geometry/q_max': qs[-1],
                         'analysis/geometry/q_min': qs[0],
                         'analysis/geometry/phi_max': phis[-1],
                         'analysis/geometry/phi_min': phis[0]})
        kam_cor = self.get_run_correlations()
        base_keys = [f'analysis/kam_correlations/{k}' for k in range(len(kam_cor))]
        base_vals = kam_cor
        rsum = np.array(self._run_sum)
        sxs = self.get_saxs(pattern=rsum / self.n_patterns, statistic=None)
        if isinstance(sxs, list):
            radial_stats = ['mean', 'sdev', 'sum', 'sum2', 'weight_sum']
            base_keys += [f'analysis/saxs/{s}' for s in radial_stats]
            base_vals += [s for s in sxs]
        fxs_dict.update(dict(zip(base_keys, base_vals)))
        return fxs_dict


def kam_analysis(framegetter=None, polar_assembler=None, radial_profiler=None,
                 start=0, stop=None, nq_bins=100,
                 np_bins=100, parallel=False, n_processes=None,
                 process_id=None, verbose=False):
    if framegetter is None:
        raise ValueError('framegetter cannot be None')
    if polar_assembler is None:
        raise ValueError('polar assembler cannot be None')
    if radial_profiler is None:
        raise ValueError('radial profiler cannot be None')
    if parallel:
        if Parallel is None:
            raise ImportError('You need the joblib package to run analysis in parallel mode.')
        if not isinstance(framegetter, dict):
            if framegetter.init_params is None:
                raise ValueError('This FrameGetter does not have init_params attribute needed to make a replica')
            framegetter = {'framegetter': type(framegetter),
                           'kwargs': framegetter.init_params}
        fsa = Parallel(n_jobs=n_processes)(delayed(kam_analysis)(framegetter=framegetter,
                                                                 polar_assembler=polar_assembler,
                                                                 radial_profiler=radial_profiler,
                                                                 start=0,
                                                                 stop=None,
                                                                 nq_bins=nq_bins,
                                                                 np_bins=np_bins,
                                                                 parallel=False,
                                                                 n_processes=n_processes,
                                                                 process_id=i,
                                                                 verbose=verbose)
                                           for i in range(n_processes))
        main_fxs_analysis = fsa[0]
        for f in fsa[1:]:
            main_fxs_analysis.merge_fxs(f)
        return main_fxs_analysis
    if isinstance(framegetter, dict):
        framegetter = framegetter['framegetter'](**framegetter['kwargs'])
    fxs_analysis = FXS(experiment_id=framegetter.experiment_id,
                       run_id=framegetter.run_id,
                       polar_assembler=polar_assembler,
                       radial_profiler=radial_profiler)
    if stop is None:
        stop = framegetter.n_frames
    frame_ids = np.arange(start, stop, dtype=int)
    if process_id is not None:
        frame_ids = np.array_split(frame_ids, n_processes)[process_id]
    for (n, i) in enumerate(frame_ids):
        if verbose:
            print(f'Frame {i:3d} ({n / len(frame_ids) * 100:0.2g})')
            print(f'{i} of {stop} ({i / stop * 100} %)')
        dataframe = framegetter.get_frame(i)
        if dataframe is None:
            continue
        if dataframe.validate() is False:
            continue
        fxs_analysis.add_frame(dataframe)
    return fxs_analysis
