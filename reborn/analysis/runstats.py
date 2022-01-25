import numpy as np
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None
from ..dataframe import DataFrame
from ..fileio.getters import ListFrameGetter


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
        for i in np.linspace(1, len(out), 1, dtype=int):
            if out[i]['sum'] is None:
                continue
            tot['sum'] += out[i]['sum']
            tot['sum2'] += out[i]['sum2']
            tot['min'] = np.minimum(tot['min'], out[i]['min'])
            tot['max'] = np.minimum(tot['max'], out[i]['max'])
            tot['n_frames'] += out[i]['n_frames']
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
    n_frames = None
    for (n, i) in enumerate(frame_ids):
        if verbose:
            print('Frame %6d (%0.2g%%)' % (i, n / len(frame_ids) * 100))
        dat = framegetter.get_frame(frame_number=i)
        if dat is None:
            continue
        rdat = dat.get_raw_data_flat()
        if rdat is None:
            continue
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
        n_frames += 1
    return {'sum': sum_pad, 'sum2': sum_pad2, 'min': min_pad, 'max': max_pad, 'n_frames': n_frames,
            'dataset_id': dat.get_dataset_id(), 'pad_geometry': dat.get_pad_geometry(),
            'mask': dat.get_mask_flat(), 'beam': dat.get_beam(), 'start': start, 'stop': stop}


def save_padstats(stats, filepath):
    np.savez(filepath, **stats)


def load_padstats(filepath):
    return np.load(filepath)


def padstats_framegetter(stats):
    r""" Make a FrameGetter that can flip through the padstats result. """
    beam = stats['beam']
    geom = stats['pad_geometry']
    mask = stats['mask']
    meen = stats['sum']/stats['n_frames']
    meen2 = stats['sum2']/stats['n_frames']
    sdev = np.sqrt(meen2-meen**2)
    dats = (('mean', meen), ('sdev', sdev), ('min', stats['min']), ('max', stats['max']))
    dfs = []
    for (a, b) in dats:
        d = DataFrame()
        d.set_dataset_id(stats['dataset_id'])
        d.set_frame_id(a)
        d.set_pad_geometry(geom)
        d.set_beam(beam)
        d.set_mask(mask)
        d.set_raw_data(b)
        dfs.append(d)
    return ListFrameGetter(dfs)


def view_padstats(stats):
    fg = padstats_framegetter(stats)
    fg.view()
