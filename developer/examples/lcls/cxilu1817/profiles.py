#!/usr/bin/env python3
import argparse
import numpy as np
import pyqtgraph as pg
from reborn.external.lcls import LCLSFrameGetter
from reborn.analysis.saxs import RadialProfiler, view_profile_runstats
from config import config

from time import perf_counter


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=int, default=154, help='Run number')
    parser.add_argument('-n', type=int, default=1e6, help='Max number of events')
    parser.add_argument('-j', type=int, default=12, help='Number of processes')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing save file')
    args = parser.parse_args()

    tik = perf_counter()
    expid = config['experiment_id']
    rnum = args.r
    pads = config['pad_detectors']
    nevents = args.n

    lclsfg = LCLSFrameGetter(experiment_id=expid,
                             run_number=rnum,
                             pad_detectors=pads,
                             max_events=nevents)

    conf = dict(debug=True,
                log_file=f'./results/logs/run{rnum:04}_profiler.log',
                message_prefix=f'run {rnum} (profiler): ',
                clear_logs=False,
                checkpoint_file=f'./results/checkpoints/run{rnum:04}.pkl')

    rp = RadialProfiler(framegetter=lclsfg, config=conf, max_frames=nevents, n_processes=args.j)
    rp.process_frames()
    tok = perf_counter()

    t = tok - tik
    print(f'processing frames took {t} s ({t % 60 } m)')

    out = rp.to_dict()

    q = out['q_range']
    labels = ['Mean', 'Standard Deviation']
    images = np.array([out['radial_mean'], out['radial_sdev']])
    pg.image(images)
