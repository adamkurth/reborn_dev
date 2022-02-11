import h5py
import numpy as np
from ..detector import concat_pad_data, PADGeometry, PADGeometryList
from ..source import Beam


def save_pad_geometry_as_h5(pad_geometry, h5_file_path):
    with h5py.File(h5_file_path, 'a') as hf:
        for i, pad in enumerate(pad_geometry):
            p = f'geometry/pad_{i:03n}'
            hf.create_dataset(f'{p}/t_vec', data=pad.t_vec)
            hf.create_dataset(f'{p}/fs_vec', data=pad.fs_vec)
            hf.create_dataset(f'{p}/ss_vec', data=pad.ss_vec)
            hf.create_dataset(f'{p}/n_fs', data=pad.n_fs)
            hf.create_dataset(f'{p}/n_ss', data=pad.n_ss)


def load_pad_geometry_from_h5(h5_file_path):
    geometry = PADGeometryList()
    with h5py.File(h5_file_path, 'r') as hf:
        pads = list(hf['geometry'].keys())
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
    if isinstance(mask, list):
        mask = concat_pad_data(mask)
    with h5py.File(h5_file_path, 'a') as hf:
        hf.create_dataset('geometry/mask', data=mask)


def load_mask_from_h5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as hf:
        mask = hf[f'geometry/mask'][:]
    return mask


def save_beam_as_h5(beam, h5_file_path):
    beam_dict = beam.to_dict()
    with h5py.File(h5_file_path, 'a') as hf:
        hf.create_dataset('data/beam/beam_profile', data=beam_dict['beam_profile'])
        del beam_dict['beam_profile']
        for k, v in beam_dict.items():
            hf.create_dataset(f'data/beam/{k}', data=np.array(v))


def load_beam_from_h5(h5_file_path):
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
    save_pad_geometry_as_h5(stats['pad_geometry'], h5_file_path)
    save_mask_as_h5(stats['mask'], h5_file_path)
    save_beam_as_h5(stats['beam'], h5_file_path)
    with h5py.File(h5_file_path, 'a') as hf:
        hf.create_dataset('meta/experiment_id', data=experiment_id)
        hf.create_dataset('meta/run_number', data=run)
        hf.create_dataset('meta/dataset_id', data=stats['dataset_id'])
        hf.create_dataset('data/n_frames', data=stats['n_frames'])
        hf.create_dataset('data/max', data=stats['max'])
        hf.create_dataset('data/min', data=stats['min'])
        hf.create_dataset('data/sum', data=stats['sum'])
        hf.create_dataset('data/sum2', data=stats['sum2'])
        hf.create_dataset('data/start', data=stats['start'])
        hf.create_dataset('data/stop', data=stats['stop'])
    print(f'Saved {h5_file_path}!')


def load_padstats_from_h5(h5_file_path):
    stats = {'pad_geometry': load_pad_geometry_from_h5(h5_file_path),
             'mask': load_mask_from_h5(h5_file_path),
             'beam': load_beam_from_h5(h5_file_path)}
    with h5py.File(h5_file_path) as hf:
        stats['experiment_id'] = hf['meta/experiment_id'][()].decode('utf-8')
        stats['run_number'] = hf['meta/run_number'][()]
        stats['dataset_id'] = hf['meta/dataset_id'][()].decode('utf-8')
        stats['n_frames'] = hf['data/n_frames'][()]
        stats['max'] = hf['data/max']
        stats['min'] = hf['data/min']
        stats['sum'] = hf['data/sum']
        stats['sum2'] = hf['data/sum2']
        stats['start'] = hf['data/start']
        stats['stop'] = hf['data/stop']
    return stats
