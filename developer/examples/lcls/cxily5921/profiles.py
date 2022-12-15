#!/usr/bin/env python3
import os
import argparse
from reborn.external.lcls import LCLSFrameGetter
from reborn import analysis
from config import get_config


def get_profiles(run_number=1, n_processes=1, max_frames=None, overwrite=False):
    config = get_config(run_number=run_number)
    os.makedirs(f"{config['results_directory']}/profiles", exist_ok=True)
    savefile = f"{config['results_directory']}/profiles/{run_number:04d}.pkl"
    if os.path.exists(savefile) and not overwrite:
        print('Loading', savefile)
        return analysis.saxs.load_profiles(savefile)
    framegetter = {'framegetter': LCLSFrameGetter, 
                   'kwargs': {'run_number': run_number,
                              'max_events': max_frames,
                              'experiment_id': config['experiment_id'], 
                              'pad_detectors': config['pad_detectors']}}
    # Reborn does the standard processing pipeline, parallelized
    stats = analysis.saxs.get_profile_runstats(framegetter=framegetter, 
                                               n_bins=config['n_q_bins'],
                                               q_range=config['q_range'],
                                               start=0,
                                               stop=max_frames,
                                               n_processes=n_processes,
                                               parallel=True)
    print('Saving', savefile)
    analysis.saxs.save_profiles(stats, savefile)
    return stats


def get_mean_dark(run_number):
    stats = get_profiles(run_number=run_number)
    print(stats['frame_ids'])


def view_profiles(run_number, *args, **kwargs):
    """ Convenience viewer for get_runstats. """
    stats = get_profiles(run_number=run_number, *args, **kwargs)
    analysis.saxs.view_profile_runstats(stats)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=int, default=154, help='Run number')
    parser.add_argument('-n', type=int, default=1e6, help='Max number of events')
    parser.add_argument('-j', type=int, default=12, help='Number of processes')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing save file')
    parser.add_argument('--view', action='store_true', help='View stats')
    args = parser.parse_args()
    kwargs = dict(run_number=args.r, max_frames=args.n, overwrite=args.overwrite, n_processes=args.j)
    if args.view:
        view_profiles(**kwargs)
        get_mean_dark(run_number=args.r)
    else:
        get_profiles(**kwargs)
