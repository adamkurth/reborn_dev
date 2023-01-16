#!/usr/bin/env python
import os
import argparse
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = delayed = None
from reborn.external.lcls import LCLSFrameGetter
from reborn.analysis.runstats import padstats, save_padstats, load_padstats, view_padstats
from config import get_config


def get_runstats(run_number=1, n_processes=1, max_frames=None, overwrite=False):
    r""" Fetches some PAD statistics for a run.  See reborn docs. """
    config = get_config(run_number=run_number)
    savefile = config['runstats']['results_directory'] + f"r{run_number:04d}.pkl"
    if os.path.exists(savefile) and not overwrite:
        print('Loading', savefile)
        return load_padstats(savefile)
    framegetter = LCLSFrameGetter(run_number=run_number, max_events=max_frames, experiment_id=config['experiment_id'],
                                  pad_detectors=config['pad_detectors'], cachedir=config['cachedir'])
    stats = padstats(framegetter=framegetter, stop=max_frames, n_processes=n_processes, config=config['runstats'])
    print('Saving', savefile)
    save_padstats(stats, savefile)
    return stats


def view_runstats(stats=None, geom=None, mask=None, **kwargs):
    """ Convenience viewer for get_runstats. Accepts same arguments as get_runstats, along with a couple more:

    Arguments:
        geom (PADGeometryList): PAD geometry.
        mask (ndarray): PAD mask.
    """
    if stats is None:
        stats = get_runstats(**kwargs)
    if mask is not None:
        stats['mask'] = mask
    if geom is not None:
        stats['pad_geometry'] = geom
    view_padstats(stats)

#
# def analyze_histogram(run_number, n_processes=1, debug=0, overwrite=False):
#     config = get_config(run_number=run_number)
#     savefile = config['results_directory']+f'/runstats/{run_number:04d}/histogram_{run_number:04d}.pkl'
#     os.makedirs(os.path.dirname(savefile), exist_ok=True)
#     if os.path.exists(savefile) and not overwrite:
#         print('loading', savefile)
#         return reborn.fileio.misc.load_pickle(savefile)
#     stats = get_runstats(run_number=run_number)
#     out = analysis.runstats.analyze_histogram(stats, n_processes=n_processes, debug=debug)
#     reborn.fileio.misc.save_pickle(out, savefile)
#     return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', type=int, required=True, help='Run number')
    parser.add_argument('--view', action='store_true', help='View stats')
    parser.add_argument('--max_events', type=int, default=1e7, help='Maximum number of events to process')
    parser.add_argument('-j', '--n_processes', type=int, default=8, help='Number of parallel processes')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite previous results')
    args = parser.parse_args()
    print(f'Fetching runstats...')
    stats = get_runstats(run_number=args.run, n_processes=args.n_processes, max_frames=args.max_events,
                         overwrite=args.overwrite)
    if args.view:
        print('Viewing runstats...')
        # hist = analyze_histogram(run_number=args.run, n_processes=args.n_processes, debug=1, overwrite=False)
        # stats['gain'] = hist['gain']
        # stats['offset'] = hist['offset']
        view_padstats(stats, histogram=True)