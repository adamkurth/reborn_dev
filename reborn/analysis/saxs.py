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
from joblib import delayed
from joblib import Parallel
from reborn.detector import RadialProfiler
from reborn import utils

debug = True


def debug_message(*args, caller=True, **kwargs):
    r""" Standard debug message, which includes the function called. """
    if debug:
        s = ''
        if caller:
            s = utils.get_caller(1)
        print('DEBUG:'+s+':', *args, **kwargs)


def get_profile_stats(dataframe, n_bins, q_range):
    r"""
    Operates on one raw diffraction pattern and returns a dictionary with the following:
    
        mean :   Mean of unmasked intensities
        median : Median of unmasked intensities
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
    sa *= 1e6  # Set the units to micro steradian solid angles
    data /= pfac * sa   # normalize our the polarization factors
    debug_message('computing profiles')
    profiler = RadialProfiler(pad_geometry=geom, mask=mask, beam=beam,
                              n_bins=n_bins, q_range=q_range)
    stats = profiler.quickstats(data)
    out_keys = ['median', 'mean', 'sdev', 'sum', 'sum2', 'counts', 'q_bins']
    out_vals = [profiler.get_median_profile(data), stats['mean'],
                stats['sdev'], stats['sum'], stats['sum2'],
                stats['weight_sum'], profiler.q_bin_centers]
    return dict(zip(out_keys, out_vals))


def get_profile_runstats(framegetter=None, n_bins=1000, q_range=None,
                         start=0, stop=None, parallel=False,
                         n_processes=None, process_id=None):
    r""" 
    Parallelized version of get_profile_stats.

    You should be able to pass in any FrameGetter subclass.
    You can try the parallel flag if you have joblib package installed... but if you parallelize, please understand that you
    cannot pass a framegetter from the main process to the children processes (because, for example, the framegetter
    might have a reference to a file handle object). Therefore, in order to parallelize, we use the convention in
    which the framegetter is passed in as a dictionary with the 'framegetter' key set to the desired FrameGetter
    subclass, and the 'kwargs' key set to the keyword arguments needed to create a new class instance.

    The return of this function is a dictionary as follows:
    
        mean :   Mean of unmasked intensities
        median : Median of unmasked intensities
        sum :    Sum of unmasked intensities
        sum2 :   Sum of squared unmasked intensities
        count :  Number of unmasked pixels in the q bin
        q_bins : The centers of the q bins

    each value is a 2D array with the computed radials for each frame.

    Returns:
        dict
    """
    if framegetter is None:
        raise ValueError('framegetter cannot be None')
    out_keys = ['median', 'mean', 'sum', 'sum2', 'counts', 'q_bins']
    if parallel:
        if Parallel is None:
            raise ImportError('You need the joblib package to run padstats in parallel mode.')
        if not isinstance(framegetter, dict):
            if framegetter.init_params is None:
                raise ValueError('This FrameGetter does not have init_params attribute needed to make a replica')
            framegetter = {'framegetter': type(framegetter), 'kwargs': framegetter.init_params}
        out = Parallel(n_jobs=n_processes)(delayed(get_profile_runstats)(framegetter=framegetter,
                                                                         start=start,
                                                                         stop=stop,
                                                                         parallel=False,
                                                                         n_processes=n_processes,
                                                                         process_id=i)
                                                                         for i in range(n_processes))
        pmedian = np.concatenate([o['median'] for o in out])
        pmean = np.concatenate([o['mean'] for o in out])
        psdev = np.concatenate([o['sdev'] for o in out])
        psum = np.concatenate([o['sum'] for o in out])
        psum2 = np.concatenate([o['sum2'] for o in out])
        pcounts = np.concatenate([o['counts'] for o in out])
        pq_bin = np.concatenate([o['q_bins'] for o in out])
        out_vals = [pmedian, pmean, psum, psum2, pcounts, pq_bin]
        return dict(zip(out_keys, out_vals))
    if isinstance(framegetter, dict):
        framegetter = framegetter['framegetter'](**framegetter['kwargs'])
    if stop is None:
        stop = framegetter.n_frames
    frame_ids = np.arange(start, stop, dtype=int)
    if process_id is not None:
        frame_ids = np.array_split(frame_ids, n_processes)[process_id]
    pmedian = np.zeros((frame_ids.size, n_bins))
    pmean = np.zeros((frame_ids.size, n_bins))
    psdev = np.zeros((frame_ids.size, n_bins))
    psum = np.zeros((frame_ids.size, n_bins))
    psum2 = np.zeros((frame_ids.size, n_bins))
    pcounts = np.zeros((frame_ids.size, n_bins))
    pq_bin = np.zeros((frame_ids.size, n_bins))
    for (n, i) in enumerate(frame_ids):
        debug_message(f'Frame {i:6d} ({n / len(frame_ids) * 100:0.2g})', end='\r')
        dat = framegetter.get_frame(frame_number=i)
        if dat is None:
            debug_message(f'Frame {i:6d} is None!!!')
            continue
        pstats = get_profile_stats(dataframe=dat,
                                   n_bins=n_bins,
                                   q_range=q_range)
        pmedian[n, :] = pstats['median'].copy()
        pmean[n, :] = pstats['mean'].copy()
        psdev[n, :] = pstats['sdev'].copy()
        psum[n, :] = pstats['sum'].copy()
        psum2[n, :] = pstats['sum2'].copy()
        pcounts[n, :] = pstats['counts'].copy()
        pq_bin[n, :] = pstats['q_bins'].copy()
    out_vals = [pmedian, pmean, psdev, psum, psum2, pcounts, pq_bin]
    return dict(zip(out_keys, out_vals))


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
