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

import h5py
import numpy as np
from ..detector import concat_pad_data, PADGeometry, PADGeometryList
from ..source import Beam


def save_pad_geometry_as_h5(pad_geometry, h5_file_path):
    r"""
    Save |PADGeometryList| in an HDF5 file format in a standardized way.

    FIXME: Missing the parent_data_slice and parent_data_shape info.

    Arguments:
        pad_geometry (|PADGeometryList|): pad geometry to save
        h5_file_path (str): filename
    """
    with h5py.File(h5_file_path, 'a') as hf:
        for i, pad in enumerate(pad_geometry):
            p = f'geometry/pad_{i:03n}'
            hf.create_dataset(f'{p}/t_vec', data=pad.t_vec)
            hf.create_dataset(f'{p}/fs_vec', data=pad.fs_vec)
            hf.create_dataset(f'{p}/ss_vec', data=pad.ss_vec)
            hf.create_dataset(f'{p}/n_fs', data=pad.n_fs)
            hf.create_dataset(f'{p}/n_ss', data=pad.n_ss)
    print(f'Saved PADGeometryList: {h5_file_path}')


def load_pad_geometry_from_h5(h5_file_path):
    r"""
    Load |PADGeometryList| from HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename

    Returns:
        pad_geometry (|PADGeometryList|): pad geometry saved in hdf5 file
    """
    geometry = PADGeometryList()
    with h5py.File(h5_file_path, 'r') as hf:
        pads = list(hf['geometry'].keys())
        if 'mask' in pads:
            pads.remove('mask')
        for pad in pads:
            p = f'geometry/{pad}'
            geom = PADGeometry()
            geom.t_vec = hf[f'{p}/t_vec'][:]
            geom.fs_vec = hf[f'{p}/fs_vec'][:]
            geom.ss_vec = hf[f'{p}/ss_vec'][:]
            geom.n_fs = hf[f'{p}/n_fs'][()]
            geom.n_ss = hf[f'{p}/n_ss'][()]
            geometry.append(geom)
    return geometry


def save_mask_as_h5(mask, h5_file_path):
    r"""
    Save mask in an HDF5 file format in a standardized way.

    Arguments:
        mask (list or |ndarray|): mask to save
        h5_file_path (str): filename
    """
    if isinstance(mask, list):
        mask = concat_pad_data(mask)
    with h5py.File(h5_file_path, 'a') as hf:
        hf.create_dataset('geometry/mask', data=mask)
    print(f'Saved mask: {h5_file_path}')


def load_mask_from_h5(h5_file_path):
    r"""
    Load mask from HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename

    Returns:
        mask (|ndarray|): mask saved in hdf5 file
    """
    with h5py.File(h5_file_path, 'r') as hf:
        geom_keys = list(hf['geometry'].keys())
        if 'mask' in geom_keys:
            mask = hf[f'geometry/mask'][:]
        else:
            mask = None
    return mask


def save_beam_as_h5(beam, h5_file_path):
    r"""
    Save |Beam| in an HDF5 file format in a standardized way.

    Arguments:
        beam (|Beam|): beam to save
        h5_file_path (str): filename
    """
    beam_dict = beam.to_dict()
    with h5py.File(h5_file_path, 'a') as hf:
        hf.create_dataset('data/beam/beam_profile', data=beam_dict['beam_profile'])
        del beam_dict['beam_profile']
        for k, v in beam_dict.items():
            hf.create_dataset(f'data/beam/{k}', data=np.array(v))
    print(f'Saved beam: {h5_file_path}')


def load_beam_from_h5(h5_file_path):
    r"""
    Load |Beam| from HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename

    Returns:
        beam (|Beam|): beam saved in hdf5 file
    """
    beam = Beam()
    beam_dict = dict()
    with h5py.File(h5_file_path, 'r') as hf:
        profile = hf['data/beam/beam_profile'][()]
        beam_dict['beam_profile'] = profile.decode('utf-8')
        beam_keys = list(hf['data/beam'].keys())
        beam_keys.remove('beam_profile')
        for k in beam_keys:
            beam_data = hf[f'data/beam/{k}'][()]
            if hasattr(beam_data, '__iter__'):
                beam_dict[k] = tuple([d for d in beam_data])
            else:
                beam_dict[k] = beam_data
    beam.from_dict(beam_dict)
    return beam


def save_padstats_as_h5(experiment_id, run, stats, h5_file_path):
    r"""
    Save padstats in an HDF5 file format in a dictionary with the keys following keys:

    FIXME: There is a sphinx warning caused by this doc string.

    Arguments:
        stats (dict): dict to save
                      with keys: 'dataset_id'
                                 'pad_geometry'
                                 'mask'
                                 'n_frames'
                                 'sum'
                                 'min'
                                 'max'
                                 'sum2'
                                 'beam'
                                 'start'
                                 'stop'
        h5_file_path (str): filename
    """
    save_stats = ['n_frames', 'max', 'min', 'sum', 'sum2', 'start', 'stop']
    save_pad_geometry_as_h5(stats['pad_geometry'], h5_file_path)
    save_mask_as_h5(stats['mask'], h5_file_path)
    save_beam_as_h5(stats['beam'], h5_file_path)
    with h5py.File(h5_file_path, 'a') as hf:
        hf.create_dataset('meta/experiment_id', data=f'{experiment_id}')
        hf.create_dataset('meta/run_number', data=run)
        hf.create_dataset('meta/dataset_id', data=stats['dataset_id'])
        for ss in save_stats:
            hf.create_dataset(f'padstats/{ss}', data=stats[ss])
    print(f'Saved padstats: {h5_file_path}')


def load_padstats_from_h5(h5_file_path):
    r"""
    Load padstats from HDF5 file format in a standardized way.

    Arguments:
        h5_file_path (str): filename

    Returns:
        stats (dict): dict to save
                      with keys: 'dataset_id'
                                 'pad_geometry'
                                 'mask'
                                 'n_frames'
                                 'sum'
                                 'min'
                                 'max'
                                 'sum2'
                                 'beam'
                                 'start'
                                 'stop'
    """
    save_stats_scalar = ['n_frames', 'start', 'stop']
    save_stats_arrays = ['max', 'min', 'sum', 'sum2']
    stats = {'pad_geometry': load_pad_geometry_from_h5(h5_file_path),
             'mask': load_mask_from_h5(h5_file_path),
             'beam': load_beam_from_h5(h5_file_path)}
    with h5py.File(h5_file_path) as hf:
        stats['experiment_id'] = hf['meta/experiment_id'][()].decode('utf-8')
        stats['run_number'] = hf['meta/run_number'][()]
        stats['dataset_id'] = hf['meta/dataset_id'][()].decode('utf-8')
        for ss in save_stats_scalar:
            stats[ss] = hf[f'padstats/{ss}'][()]
        for ss in save_stats_arrays:
            stats[ss] = hf[f'padstats/{ss}'][:]
    return stats


def save_analysis_as_h5(dataset_name, data, h5_file_path):
    r"""
    Save analysis result in an HDF5 file format in a standardized way.

    Arguments:
        dataset_name (str): dataset name for hdf5
        data (|ndarray|): data to store
        h5_file_path (str): filename
    """
    with h5py.File(h5_file_path, 'a') as hf:
        hf.create_dataset(f'analysis/{dataset_name}', data=data)
    print(f'Saved analysis: {h5_file_path}!')


def get_analysis_h5_keys(h5_file_path):
    r"""
    Retrieve analysis keys saved in an HDF5 file format.

    Arguments:
        h5_file_path (str): filename
    """
    with h5py.File(h5_file_path, 'a') as hf:
        ks = list(hf['analysis/'])
    return ks


def load_analysis_from_h5(analysis_key, h5_file_path):
    r"""
    Load analysis results from an HDF5 file format in a standardized way.

    Arguments:
        analysis_key (list): analysis keys for data to retrieve
        h5_file_path (str): filename

    Returns:
        data (dict): data stored in HDF5 file
    """
    data = {}
    with h5py.File(h5_file_path, 'a') as hf:
        for k in analysis_key:
            data[k] = hf[f'analysis/{k}'][:]
    return data


def save_fxs_as_h5(fxs, filename, **kwargs):
    analysis_dict = fxs.to_dict()
    analysis_dict.update(kwargs)
    if filename[-4:] == 'hdf5':
        with h5py.File(filename, 'a') as hf:
            for k, v in analysis_dict.items():
                if v is None:
                    continue
                else:
                    hf.create_dataset(k, data=v)
    else:
        print('Only hdf5 files can be saved at this time.')
    print(f'Saved : {filename}', end='\r')


def load_fxs_from_h5(filename):
    fxs_dict = dict()
    bkeys = ['meta', 'analysis/kam_correlations/', 'analysis/geometry/']
    okeys = ['analysis/saxs', 'analysis/n_patterns',
             'analysis/run_max', 'analysis/run_min',
             'analysis/run_sum', 'analysis/run_sum2']
    if filename[-4:] == 'hdf5':
        with h5py.File(filename, 'r') as hf:
            fxs_dict.update({k: hf[f'{k}'][()] for k in okeys})
            for sec in bkeys:
                sec_keys = list(hf[f'{sec}'])
                fxs_dict.update({k: hf[f'{sec}/{k}'][()] for k in sec_keys})
    else:
        print('Only hdf5 files can be loaded at this time.')
    return fxs_dict


def save_run_profile_stats_as_h5(pstats, h5_file_path):
    r"""
    Save profile runstats in an HDF5 file format in a standardized way.

    Arguments:
        beam (|Beam|): beam to save
        h5_file_path (str): filename
    """
    with h5py.File(h5_file_path, 'a') as hf:
        for k, v in pstats.items():
            hf.create_dataset(f'pstats/{k}', data=np.array(v))
    print(f'Saved run profile stats: {h5_file_path}')


def load_run_profile_stats_from_h5(h5_file_path):
    stat_keys = ['median', 'mean', 'sum', 'sum2', 'counts', 'q_bins']
    pstats = dict()
    with h5py.File(h5_file_path, 'a') as hf:
        for k in stat_keys:
            pstats[k] = hf[f'pstats/{k}'][:]
    return pstats
