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


def _allkeys(obj):
    # https://stackoverflow.com/questions/59897093/get-all-keys-and-its-hierarchy-in-h5-file-using-python-library-h5py
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + _allkeys(value)
            else:
                keys = keys + (value.name,)
    return keys


def all_keys_in_file(h5_file_path):
    r"""
    Gather all keys an HDF5 file.

    Arguments:
        h5_file_path (str): filename
    """
    with h5py.File(h5_file_path, 'a') as hf:
        keys = _allkeys(hf)
    return keys


def save_metadata_as_h5(h5_file_path, experiment_id, run_id):
    r"""
    Save metadata in an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename
        experiment_id (str): experiment id
        run_id (int): experiment run id
    """
    metadata = {'meta/experiment_id': f'{experiment_id}',
                'meta/run_id': run_id}
    file_keys = all_keys_in_file(h5_file_path)
    with h5py.File(h5_file_path, 'a') as hf:
        for k, v in metadata.items():
            if k not in file_keys:
                hf.create_dataset(k, data=v)
    print(f'Saved metadata: {h5_file_path}')


def load_metadata_from_h5(h5_file_path):
    r"""
    Load metadata from an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename
    """
    metadata = dict()
    with h5py.File(h5_file_path, 'a') as hf:
        metadata['experiment_id'] = hf['meta/experiment_id'][()].decode('utf-8')
        metadata['run_id'] = hf['meta/run_id']
    return metadata


def save_pad_geometry_as_h5(h5_file_path, pad_geometry):
    r"""
    Save |PADGeometryList| in an HDF5 file in a standardized way.

    FIXME: Missing the parent_data_slice and parent_data_shape info.

    Arguments:
        h5_file_path (str): filename
        pad_geometry (|PADGeometryList|): pad geometry to save
    """
    file_keys = all_keys_in_file(h5_file_path)
    with h5py.File(h5_file_path, 'a') as hf:
        for i, pad in enumerate(pad_geometry):
            p = f'geometry/pad_{i:03n}'
            if p in file_keys:
                continue
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


def save_mask_as_h5(h5_file_path, mask):
    r"""
    Save mask in an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename
        mask (list or |ndarray|): mask to save
    """
    file_keys = all_keys_in_file(h5_file_path)
    if isinstance(mask, list):
        mask = concat_pad_data(mask)
    with h5py.File(h5_file_path, 'a') as hf:
        m = 'geometry/mask'
        if m not in file_keys:
            hf.create_dataset(m, data=mask)
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


def save_beam_as_h5(h5_file_path, beam):
    r"""
    Save |Beam| in an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename
        beam (|Beam|): beam to save
    """
    file_keys = all_keys_in_file(h5_file_path)
    beam_dict = beam.to_dict()
    with h5py.File(h5_file_path, 'a') as hf:
        bpk = 'data/beam/beam_profile'
        if bpk not in file_keys:
            hf.create_dataset(bpk, data=beam_dict['beam_profile'])
        del beam_dict['beam_profile']
        for k, v in beam_dict.items():
            bk = f'data/beam/{k}'
            if bk not in file_keys:
                hf.create_dataset(bk, data=np.array(v))
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


def save_padstats_as_h5(h5_file_path, experiment_id, run_id, stats):
    r"""
    Save padstats in an HDF5 file in a dictionary with the keys following keys:

    Arguments:
        h5_file_path (str): filename
        experiment_id (str): experiment id
        run_id (int): experiment run id
        stats (dict): dict to save
                      with keys: dataset_id
                                 pad_geometry
                                 mask
                                 n_frames
                                 sum
                                 min
                                 max
                                 sum2
                                 beam
                                 start
                                 stop
    """
    file_keys = all_keys_in_file(h5_file_path)
    save_metadata_as_h5(h5_file_path=h5_file_path, experiment_id=experiment_id, run_id=run_id)
    save_pad_geometry_as_h5( h5_file_path=h5_file_path, pad_geometry=stats['pad_geometry'])
    save_mask_as_h5(h5_file_path=h5_file_path, mask=stats['mask'])
    save_beam_as_h5(h5_file_path=h5_file_path, beam=stats['beam'])
    del stats['pad_geometry']
    del stats['mask']
    del stats['beam']
    with h5py.File(h5_file_path, 'a') as hf:
        md = 'meta/dataset_id'
        if md not in file_keys:
            hf.create_dataset(md, data=stats['dataset_id'])
        del stats['dataset_id']
        for k, v in stats.items():
            psk = f'padstats/{k}'
            if psk not in file_keys:
                hf.create_dataset(psk, data=v)
    print(f'Saved padstats: {h5_file_path}')


def load_padstats_from_h5(h5_file_path):
    r"""
    Load padstats from HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename

    Returns:
        stats (dict): dict to load
                      with keys: dataset_id
                                 pad_geometry
                                 mask
                                 n_frames
                                 sum
                                 min
                                 max
                                 sum2
                                 beam
                                 start
                                 stop
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


def save_analysis_as_h5(h5_file_path, **kwargs):
    r"""
    Save analysis result in an HDF5 file format in a standardized way.

    Arguments:
        h5_file_path (str): filename
        kwargs: the data you would like to save (format: key=data)
    """
    file_keys = all_keys_in_file(h5_file_path)
    with h5py.File(h5_file_path, 'a') as hf:
        for k, v in kwargs.items():
            key_name = f'analysis/{k}'
            if key_name not in file_keys:
                hf.create_dataset(f'analysis/{k}', data=v)
    print(f'Saved analysis: {h5_file_path}!')


def load_analysis_from_h5(h5_file_path):
    r"""
    Load analysis results from an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename

    Returns:
        data (dict): data stored in HDF5 file
    """
    data = dict()
    file_keys = all_keys_in_file(h5_file_path)
    analysis_keys = [k for k in file_keys if 'analysis' in k]
    with h5py.File(h5_file_path, 'a') as hf:
        for k in analysis_keys:
            data[k] = hf[f'{k}'][:]
    return data


def get_analysis_h5_keys(h5_file_path):
    r"""
    Retrieve analysis keys saved in an HDF5 file format.

    Arguments:
        h5_file_path (str): filename
    """
    with h5py.File(h5_file_path, 'a') as hf:
        ks = list(hf['analysis/'])
    return ks


def load_analysis_key_from_h5(h5_file_path, *analysis_key):
    r"""
    Load analysis results from an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename
        analysis_key (str): analysis keys for data to retrieve

    Returns:
        data (dict): data stored in HDF5 file
    """
    data = dict()
    with h5py.File(h5_file_path, 'a') as hf:
        for k in analysis_key:
            data[k] = hf[f'analysis/{k}'][:]
    return data


def save_fxs_as_h5(h5_file_path, fxs, experiment_id, run_id, **kwargs):
    file_keys = all_keys_in_file(h5_file_path)
    analysis_dict = fxs.to_dict()
    analysis_dict.update(kwargs)
    if h5_file_path[-4:] == 'hdf5':
        save_metadata_as_h5(h5_file_path=h5_file_path, experiment_id=experiment_id, run_id=run_id)
        with h5py.File(h5_file_path, 'a') as hf:
            for k, v in analysis_dict.items():
                if k in file_keys:
                    continue
                if v is None:
                    continue
                else:
                    hf.create_dataset(k, data=v)
    else:
        print('Only hdf5 files can be saved at this time.')
    print(f'Saved : {h5_file_path}', end='\r')


def load_fxs_from_h5(h5_file_path):
    fxs_dict = dict()
    bkeys = ['meta', 'analysis/kam_correlations/', 'analysis/geometry/']
    okeys = ['analysis/saxs', 'analysis/n_patterns',
             'analysis/run_max', 'analysis/run_min',
             'analysis/run_sum', 'analysis/run_sum2']
    if h5_file_path[-4:] == 'hdf5':
        with h5py.File(h5_file_path, 'r') as hf:
            fxs_dict.update({k: hf[f'{k}'][()] for k in okeys})
            for sec in bkeys:
                sec_keys = list(hf[f'{sec}'])
                fxs_dict.update({k: hf[f'{sec}/{k}'][()] for k in sec_keys})
    else:
        print('Only hdf5 files can be loaded at this time.')
    return fxs_dict


def save_run_profile_stats_as_h5(h5_file_path, pstats):
    r"""
    Save profile runstats in an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename
        pstats (dict): profile stats
                       with keys: mean
                                  sdev
                                  sum
                                  sum2
                                  weight_sum
    """
    file_keys = all_keys_in_file(h5_file_path)
    with h5py.File(h5_file_path, 'a') as hf:
        for k, v in pstats.items():
            if k in file_keys:
                continue
            hf.create_dataset(f'profile_stats/{k}', data=np.array(v))
    print(f'Saved run profile stats: {h5_file_path}')


def load_run_profile_stats_from_h5(h5_file_path):
    r"""
    Load profile runstats from an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename

    Returns:
        pstats (dict): profile stats
                       with keys: mean
                                  sdev
                                  sum
                                  sum2
                                  weight_sum
    """
    keys = ['median', 'mean', 'sum', 'sum2', 'counts', 'q_bins']
    pstats = dict()
    with h5py.File(h5_file_path, 'a') as hf:
        for k in keys:
            pstats[k] = hf[f'profile_stats/{k}'][:]
    return pstats
