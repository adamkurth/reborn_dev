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
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None
from .. import detector
from ..dataframe import DataFrame
from ..fileio.getters import ListFrameGetter
from ..source import Beam
from ..viewers.qtviews.padviews import PADView

def padstats(framegetter=None, start=0, stop=None, parallel=False, n_processes=None, process_id=None, verbose=False):
    r""" EXPERIMENTAL!!!  Since we often need to loop over a series of dataframes and fetch the mean, variance, min
    max values, this function will do that for you.  You should be able to pass in any FrameGetter subclass.  You can
    try the parallel flag if you have joblib package installed... but if you parallelize, please understand that you
    cannot pass a framegetter from the main process to the children processes (because, for example, the framegetter
    might have a reference to a file handle object).  Therefore, in order to parallelize, we use the convention in
    which the framegetter is passed in as a dictionary with the 'framegetter' key set to the desired FrameGetter
    subclass, and the 'kwargs' key set to the keyword arguments needed to create a new class instance.

    The return of this function is a dictionary as follows:

    {'sum': sum_pad, 'sum2': sum_pad2, 'min': min_pad, 'max': max_pad, 'n_frames': n_frames,
            'dataset_id': dat.get_dataset_id(), 'pad_geometry': dat.get_pad_geometry(),
            'mask': dat.get_mask_flat(), 'beam': dat.get_beam()}

    There is a corresponding view_padstats function to view the results in this dictionary.

    Returns: dict """
    if framegetter is None:
        raise ValueError('framegetter cannot be None')
    if parallel:
        if Parallel is None:
            raise ImportError('You need the joblib package to run padstats in parallel mode.')
        if not isinstance(framegetter, dict):
            if framegetter.init_params is None:
                raise ValueError('This FrameGetter does not have init_params attribute needed to make a replica')
            framegetter = {'framegetter': type(framegetter), 'kwargs': framegetter.init_params}
        out = Parallel(n_jobs=n_processes)(delayed(padstats)(framegetter=framegetter, start=start, stop=stop,
                                                             parallel=False, n_processes=n_processes, process_id=i,
                                                             verbose=verbose)
                                                             for i in range(n_processes))
        tot = out[0]
        for o in out[1:]:
            if o['sum'] is None:
                continue
            tot['sum'] += o['sum']
            tot['sum2'] += o['sum2']
            tot['min'] = np.minimum(tot['min'], o['min'])
            tot['max'] = np.minimum(tot['max'], o['max'])
            tot['n_frames'] += o['n_frames']
        return tot
    if isinstance(framegetter, dict):
        framegetter = framegetter['framegetter'](**framegetter['kwargs'])
    if stop is None:
        stop = framegetter.n_frames
    frame_ids = np.arange(start, stop, dtype=int)
    if process_id is not None:
        frame_ids = np.array_split(frame_ids, n_processes)[process_id]
    first = True
    sum_pad = None
    sum_pad2 = None
    min_pad = None
    max_pad = None
    beam_wavelength = 0
    beam_frames = 0
    n_frames = 0
    dataset_id = None
    pad_geometry = None
    mask = None
    for (n, i) in enumerate(frame_ids):
        if verbose:
            print(f'Frame {i:6d} ({n / len(frame_ids) * 100:0.2g})')
        dat = framegetter.get_frame(frame_number=i)
        if dat is None:
            print(f'Frame {i:6d} is None!!!')
            continue
        rdat = dat.get_raw_data_flat()
        if rdat is None:
            continue
        if dat.validate():
            beam_data = dat.get_beam()
            beam_wavelength += beam_data.wavelength
            beam_frames += 1
        if first:
            s = rdat.shape
            sum_pad = np.zeros(s)
            sum_pad2 = np.zeros(s)
            max_pad = rdat
            min_pad = rdat
            n_frames = np.zeros(s)
            first = False
        sum_pad += rdat
        sum_pad2 += rdat ** 2
        min_pad = np.minimum(min_pad, rdat)
        max_pad = np.maximum(max_pad, rdat)
        if dataset_id is None:
            dataset_id = dat.get_dataset_id()
        if pad_geometry is None:
            pad_geometry = dat.get_pad_geometry()
        if mask is None:
            mask = dat.get_mask_flat()
        n_frames += 1
    if beam_frames == 0:
        beam = None
    else:
        avg_wavelength = beam_wavelength / beam_frames
        beam = Beam(wavelength=avg_wavelength)
    return {'dataset_id': dataset_id,
            'pad_geometry': pad_geometry,
            'mask': mask,
            'n_frames': n_frames,
            'sum': sum_pad,
            'min': min_pad,
            'max': max_pad,
            'sum2': sum_pad2,
            'beam': beam,
            'start': start,
            'stop': stop}


def save_padstats(stats, filepath):
    np.savez(filepath, allow_pickle=True, **stats)


def load_padstats(filepath):
    return np.load(filepath)


def padstats_framegetter(stats, geom=None, mask=None):
    r""" Make a FrameGetter that can flip through the padstats result. """
    beam = stats['beam']
    if geom is None:
        geom = stats['pad_geometry']
    geom = detector.PADGeometryList(geom)
    if mask is None:
        mask = stats['mask']
    meen = stats['sum']/stats['n_frames']
    meen2 = stats['sum2']/stats['n_frames']
    sdev = np.nan_to_num(meen2-meen**2)
    sdev[sdev < 0] = 0
    sdev = np.sqrt(sdev)
    dats = (('mean', meen), ('sdev', sdev), ('min', stats['min']), ('max', stats['max']))
    dfs = []
    for (a, b) in dats:
        d = DataFrame()
        d.set_dataset_id(stats['dataset_id'])
        d.set_frame_id(a)
        d.set_pad_geometry(geom)
        if beam is not None:
            d.set_beam(beam)
        d.set_mask(mask)
        d.set_raw_data(b)
        dfs.append(d)
    return ListFrameGetter(dfs)


def view_padstats(stats, geom=None, mask=None):
    fg = padstats_framegetter(stats, geom=geom, mask=mask)
    pv = PADView(frame_getter=fg, percentiles=[1, 99])
    pv.start()
    # fg.view()
