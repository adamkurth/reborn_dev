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
import os
from ..detector import PADGeometry, PADGeometryList
from ..source import Beam
from ..dataframe import DataFrame


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


def save_metadata_as_h5(h5_file_path, experiment_id, run_id, **meta):
    r"""
    Save metadata in an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename
        experiment_id (str): experiment id
        run_id (int): experiment run id
        meta: other metadata to save (format: key=value)
    """
    metadata = {'experiment_id': f'{experiment_id}',
                'run_id': run_id}
    metadata.update(meta)
    file_keys = all_keys_in_file(h5_file_path)
    with h5py.File(h5_file_path, 'a') as hf:
        for k, v in metadata.items():
            mk = f'meta/{k}'
            if mk in file_keys:
                continue
            hf.create_dataset(mk, data=v)
    print(f'Saved metadata: {h5_file_path}')


def load_metadata_from_h5(h5_file_path):
    r"""
    Load metadata from an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename

    Returns:
        metadata (dict): dictionary with everything stored in
                         hdf5 file under the 'meta' group.
    """
    with h5py.File(h5_file_path, 'a') as hf:
        metadata = {k: hf[f'meta/{k}'][()].decode('utf-8') for k in hf['meta'].keys()}
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
        for pad in pads:
            p = f'geometry/{pad}'
            g = PADGeometry()
            g.t_vec = hf[f'{p}/t_vec'][:]
            g.fs_vec = hf[f'{p}/fs_vec'][:]
            g.ss_vec = hf[f'{p}/ss_vec'][:]
            g.n_fs = hf[f'{p}/n_fs'][()]
            g.n_ss = hf[f'{p}/n_ss'][()]
            geometry.append(g)
    return geometry


def save_pad_data_as_h5(h5_file_path, data_key, data_list):
    r"""
    Save data in an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename
        data_key (str): root key for data
        data_list (list): data to save
    """
    file_keys = all_keys_in_file(h5_file_path)
    with h5py.File(h5_file_path, 'a') as hf:
        for i, data in enumerate(data_list):
            d = f'{data_key}/pad_{i:03n}'
            if d in file_keys:
                continue
            hf.create_dataset(d, data=data)
    print(f'Saved PAD {data_key}: {h5_file_path}')


def load_pad_data_from_h5(h5_file_path, data_key):
    with h5py.File(h5_file_path, 'a') as hf:
        data_list = [hf[f'{data_key}/{k}'][:] for k in hf[data_key].keys()]
    return data_list


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
        bpk = 'beam/beam_profile'
        if bpk not in file_keys:
            hf.create_dataset(bpk, data=beam_dict['beam_profile'])
        del beam_dict['beam_profile']
        for k, v in beam_dict.items():
            bk = f'beam/{k}'
            if bk in file_keys:
                continue
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
        if 'beam' in hf.keys():
            profile = hf['beam/beam_profile'][()]
            beam_dict['beam_profile'] = profile.decode('utf-8')
            beam_keys = list(hf['beam'].keys())
            beam_keys.remove('beam_profile')
            for k in beam_keys:
                beam_data = hf[f'beam/{k}'][()]
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
        h5_file_path (str): Filename
        experiment_id (str): Experiment ID
        run_id (int): Experiment run ID
        stats (dict): Dictionary to save with keys

                        - dataset_id
                        - pad_geometry
                        - mask
                        - n_frames
                        - sum
                        - min
                        - max
                        - sum2
                        - beam
                        - start
                        - stop
    """
    file_keys = all_keys_in_file(h5_file_path)
    save_metadata_as_h5(h5_file_path=h5_file_path, experiment_id=experiment_id, run_id=run_id)
    save_pad_geometry_as_h5( h5_file_path=h5_file_path, pad_geometry=stats['pad_geometry'])
    mask = stats['pad_geometry'].split_data(stats['mask'])
    save_pad_data_as_h5(h5_file_path=h5_file_path, data_key='mask', data_list=mask)
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
             'mask': load_pad_data_from_h5(h5_file_path, data_key='mask'),
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


def save_analysis_as_h5(h5_file_path, **analysis):
    r"""
    Save analysis result in an HDF5 file format in a standardized way.

    Arguments:
        h5_file_path (str): filename
        analysis: data you would like to save (format: key=data)
    """
    file_keys = all_keys_in_file(h5_file_path)
    with h5py.File(h5_file_path, 'a') as hf:
        for k, v in analysis.items():
            key_name = f'analysis/{k}'
            if key_name in file_keys:
                continue
            if v is None:
                continue
            hf.create_dataset(key_name, data=v)
    print(f'Saved analysis: {h5_file_path}!')


def load_analysis_from_h5(h5_file_path):
    r"""
    Load analysis results from an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename

    Returns:
        data (dict): data stored in HDF5 file
    """
    with h5py.File(h5_file_path, 'a') as hf:
        data = {k: hf[f'analysis/{k}'][:] for k in hf['analysis'].keys()}
    return data


def load_analysis_key_from_h5(h5_file_path, *analysis_key):
    r"""
    Load analysis results from a specific key in an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename
        analysis_key (str): analysis keys for data to retrieve

    Returns:
        data (dict): data stored in HDF5 file
    """
    if isinstance(analysis_key, str):
        analysis_key = [analysis_key]
    with h5py.File(h5_file_path, 'a') as hf:
        data = {k: hf[f'analysis/{k}'][:] for k in analysis_key}
    return data


def save_fxs_as_h5(h5_file_path, fxs, **kwargs):
    r"""
    Save FXS object in an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): Filename
        fxs (|FXS|): reborn.analysis.fluctionations.FXS object to save
        **kwargs: anything else you would like to save (format: key=data)
    """
    analysis_dict = fxs.to_dict()
    analysis_dict.update(kwargs)
    save_metadata_as_h5(h5_file_path=h5_file_path,
                        experiment_id=analysis_dict['experiment_id'],
                        run_id=analysis_dict['run_id'])
    del analysis_dict['experiment_id']
    del analysis_dict['run_id']
    save_analysis_as_h5(h5_file_path, **analysis_dict)
    print(f'Saved : {h5_file_path}')


def load_fxs_from_h5(h5_file_path):
    r"""
    Load FXS object from an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): Filename

    Returns:
        fxs_dict (dict): dictionary to instantiate an FXS object
    """
    fxs_dict = load_metadata_from_h5(h5_file_path)
    fxs_dict.update(load_analysis_from_h5(h5_file_path))
    return fxs_dict


def save_run_profile_stats_as_h5(h5_file_path, pstats):
    r"""  FIXME: Add link to the function that generates profile runstats.
    Save profile runstats in an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): Filename
        pstats (dict): Profile stats with keys:

                        - mean: Radially binned mean of pixel values.
                        - sdev: Radially binned standard deviation of pixel values.
                        - sum: Radially binned sum of pixel values
                        - sum2: Radially binned sum of squared pixel values.
                        - weight_sum: Radially binned sum of pixel weights.
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
    with h5py.File(h5_file_path, 'a') as hf:
        pstats = {k: hf[f'profile_stats/{k}'][:] for k in keys}
    return pstats


def save_dataframe_as_h5(h5_base_path, dataframe, **kwargs):
    r"""
    Save dataframe to HDF5 file in a standardized way.

    Arguments:
        h5_base_path (str): path to where file will be saved
        dataframe (|DataFrame|): frame data
        **kwargs : anything else to save (needs manual loading)
    """
    experiment_id, run_id = dataframe.get_dataset_id().split(':')
    h5_base = f'{h5_base_path}/{experiment_id}/{run_id}'
    os.makedirs(h5_base, exist_ok=True)
    h5_file_path = f'{h5_base}/{dataframe.get_frame_index()}.h5'
    with h5py.File(h5_file_path, 'a') as hf:
        save_metadata_as_h5(h5_file_path, experiment_id, run_id, **kwargs)
        for i, j in enumerate(dataframe.get_frame_id()):
            hf.create_dataset(f'frame_id/{i}', data=j)
        for k, v in kwargs.values():
            hf.create_dataset(f'{k}', data=v)
    b = dataframe.get_beam()
    if b is not None:
        save_beam_as_h5(h5_file_path, b)
    g = dataframe.get_pad_geometry()
    if g is not None:
        save_pad_geometry_as_h5(h5_file_path, g)
    d = dataframe.get_raw_data_list()
    if d is not None:
        save_pad_data_as_h5(h5_file_path, data_key='data', data_list=d)
    m = dataframe.get_mask_list()
    if m is not None:
        save_pad_data_as_h5(h5_file_path, data_key='mask', data_list=m)


def load_dataframe_from_h5(h5_file_path):
    r"""
    Load dataframe from an HDF5 file in a standardized way.

    Arguments:
        h5_file_path (str): filename

    Returns:
        dataframe (|DataFrame|): frame data
    """
    dataframe = DataFrame()
    meta = load_metadata_from_h5(h5_file_path)
    exp = meta['experiment_id']
    run = meta['run_id']
    dataframe.set_dataset_id(f'{exp}:{run}')
    dataframe.set_frame_index(h5_file_path.split('/')[-1].split('.')[0])
    b = load_beam_from_h5(h5_file_path)
    if b is not None:
        dataframe.set_beam(b)
    g = load_pad_geometry_from_h5(h5_file_path)
    if g is not None:
        dataframe.set_pad_geometry(g)
    d = load_pad_data_from_h5(h5_file_path, 'data')
    if d is not None:
        dataframe.set_raw_data(d)
    m = load_pad_data_from_h5(h5_file_path, 'mask')
    if m is not None:
        dataframe.set_mask(m)
    with h5py.File(h5_file_path, 'a') as hf:
        dataframe.set_frame_id(tuple(hf[f'frame_id/{i}'][()] for i in hf['frame_id'].keys()))
    return dataframe
