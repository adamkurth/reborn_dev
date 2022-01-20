#!/usr/bin/env python3
import sys
from joblib import Memory
from reborn.external.lcls import LCLSFrameGetter
from reborn import analysis
from config import config

# This sets up caching of results without needing to think about format and naming conventions.  Interesting feature
# of the joblib package.  I don't know how efficient it really is...
memory = Memory(config['results_directory'], verbose=1)


@memory.cache(ignore=['n_processes'])  # Results cached to disk, restored if same arguments passed again.
def get_runstats(run_number=1, n_processes=1, max_frames=None):
    """ Attempts to fetch the PAD data sum, sum^2, min, max of a run. """
    # We provide the FrameGetter subclass (not instance) and the arguments to initialize the FrameGetter
    framegetter = {'framegetter': LCLSFrameGetter, 
                   'kwargs': {'run_number': run_number,
                              'max_events': max_frames,
                              'experiment_id': config['experiment_id'], 
                              'pad_detectors': config['pad_detectors']}}
    # Reborn does the standard processing pipeline, parallelized
    return analysis.runstats.padstats(n_processes=n_processes, parallel=True, framegetter=framegetter, verbose=True)


def view_runstats(run_number=1, n_processes=1, max_frames=None):
    """ Convenience viewer for get_runstats. """
    stats = get_runstats(run_number=run_number, n_processes=n_processes, max_frames=max_frames)
    analysis.runstats.view_padstats(stats)
    
    
if __name__ == '__main__':
    from config import config
    run_number = 125
    if len(sys.argv) > 1:
        run_number = int(sys.argv[1])
    view_runstats(run_number=run_number, n_processes=16, max_frames=1000)
