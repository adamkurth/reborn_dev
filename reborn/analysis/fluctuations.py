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
from reborn import detector
from .parallel import get_setup_data
from .parallel import ParallelAnalyzer
try:
    from joblib import delayed
    from joblib import Parallel
except ImportError:
    Parallel = None
    delayed = None


def correlate(s1, s2=None, cached=False):
    r"""
    Computes correlation function.
    If two signals are provided computes cross correlation.
    If one signal is provided computes auto correlation.
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
    d_avg = np.zeros_like(d_sum, dtype=float)
    np.divide(d_sum, count, out=d_avg, where=count != 0)
    d = (data.T - d_avg).T
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


class FXSError(Exception):
    def __int__(self, message):
        super.__init__(message)


class FXS(ParallelAnalyzer):

    beam = None
    initial_frame = None
    n_patterns = None
    np_bins = None
    nq_bins = None
    pad_geometry = None
    polar_assembler = None
    run_sum_correlations = None

    def __init__(self, framegetter=None, **kwargs):
        r"""
        Class to compute angular cross correlations.

        Arguments:
            experiment_id (str): Experiment identifier (optional, default='default').
            run_id (int): Run identifier (optional, default=0).
            n_q_bins (int): Number of q bins for polar binning (optional, default=100).
            n_phi_bins (int): Number of phi bins for polar binning (optional, default=100).
            q_range (tuple): Polar binning q range (optional, default=(qmin, qmax) from geometry).
            phi_range (tuple): Polar binning phi range (optional, default=(0,2pi)).
            max_iterations (int): Number of shots in data to check for beam and geometry (optional, default=1e4).
            beam (|Beam|): X-ray beam (optional, default is read from data).
            pad_geometry (|PADGeometryList|): Detector geometry (optional, default is read from data).
        """
        super().__init__(framegetter=framegetter, **kwargs)
        self.experiment_id = kwargs.get('experiment_id', 'default')
        self.run_id = kwargs.get('run_id', 0)
        self.polar_assembler = self.setup_polar_pad_assembler(**kwargs)
        self.n_patterns = 0

    def __str__(self):
        out = f"    Experiment ID: {self.experiment_id}\n"
        out += f"           Run ID: {self.run_id}\n"
        out += f"Patterns Averaged: {self.n_patterns}\n"
        return out

    def setup_polar_pad_assembler(self, **kwargs):
        self.logger.info('Setting up PolarPADAssembler')
        self.nq_bins = kwargs.get('n_q_bins', 100)
        self.logger.info(f'n_q_bins: {self.nq_bins}')
        self.np_bins = kwargs.get('n_phi_bins', 100)
        self.logger.info(f'n_phi_bins: {self.np_bins}')
        q_range = kwargs.get('q_range', None)
        phi_range = kwargs.get('phi_range', None)
        self.pad_geometry, self.beam, self.initial_frame = get_setup_data(framegetter=self.framegetter, **kwargs)
        if (self.pad_geometry is None) or (self.beam is None):
            msg = 'PADGeometry or Beam are None; likely not enough iterations or bad run.'
            self.logger.warning(msg)
            raise FXSError(msg)
        else:
            self.logger.info(f'beam found')
            self.logger.info(f'pad_geometry found')
        assembler = detector.PolarPADAssembler(pad_geometry=self.pad_geometry,
                                               beam=self.beam,
                                               n_q_bins=self.nq_bins,
                                               q_range=q_range,
                                               n_phi_bins=self.np_bins,
                                               phi_range=phi_range)
        return assembler

    def add_frame(self, dat):
        if isinstance(dat, reborn.dataframe.DataFrame):
            if dat.validate():
                self.logger.warning("Dataframe is valid, adding to analysis ...")
                frame_data = dat.get_raw_data_list()
                frame_mask = dat.get_mask_list()
                frame_polar_data, frame_polar_mask = self.polar_assembler.get_mean(data=frame_data,
                                                                                   mask=frame_mask)
                frame_correlations = compute_data_correlations(frame_polar_data, frame_polar_mask)
                if self.run_sum_correlations is None:
                    self.run_sum_correlations = frame_correlations
                else:
                    for i, fc in enumerate(frame_correlations):
                        self.run_sum_correlations[i] += fc
                self.n_patterns += 1
            else:
                self.logger.warning("Dataframe not valid, skipping ...")
        else:
            self.logger.warning("Only dataframe addition is supported at this time.")

    def to_dict(self):
        fxs_dict = dict(experiment_id=self.experiment_id,
                        run_id=self.run_id,
                        n_patterns=self.n_patterns,
                        initial_frame=self.initial_frame)
        qr = self.polar_assembler.q_range
        pr = self.polar_assembler.phi_range
        asm_dict = dict(n_q_bins=self.nq_bins,
                        q_min=qr[0],
                        q_max=qr[1],
                        n_phi_bins=self.np_bins,
                        phi_min=pr[0],
                        phi_max=pr[1])
        fxs_dict.update(asm_dict)
        base_keys = [f'run_sum_correlations/{k}' for k in range(len(self.run_sum_correlations))]
        base_vals = self.run_sum_correlations
        fxs_dict.update(dict(zip(base_keys, base_vals)))
        return fxs_dict

    def clear_data(self):
        self.initial_frame = None
        self.n_patterns = 0
        self.np_bins = None
        self.nq_bins = None
        self.polar_assembler = None
        self.run_sum_correlations = None

    def from_dict(self, stats):
        self.clear_data()
        self.experiment_id = stats['experiment_id']
        self.run_id = stats['run_id']
        self.initial_frame = stats['initial_frame']
        self.n_patterns = stats['n_patterns']
        self.nq_bins = stats['n_q_bins']
        self.np_bins = stats['n_phi_bins']
        self.polar_assembler = self.setup_polar_pad_assembler(q_range=(stats['q_min'], stats['q_max']),
                                                              phi_range=(stats['phi_min'], stats['phi_max']),
                                                              max_iterations=stats['max_iterations'])
        self.run_sum_correlations = [f'run_sum_correlations/{k}' for k in range(len(self.run_sum_correlations))]

    def concatenate(self, stats):
        self.logger.warning('Merging')
        self.n_patterns += stats['n_patterns']
        self.np_bins = stats['n_phi_bins']
        self.nq_bins = stats['n_q_bins']
        if self.run_sum_correlations is None:
            self.run_sum_correlations = stats['run_sum_correlations']
        else:
            for i, c in enumerate(stats.run_sum_correlations):
                self.run_sum_correlations[i] += c
