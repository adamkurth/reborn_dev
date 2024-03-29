#!/usr/bin/env python3
import sys
from reborn.external.lcls import LCLSFrameGetter
from config import get_run_config
run_number = 117
if len(sys.argv) > 1:
    run_number = int(sys.argv[1])
config = get_run_config(run_number)
fg = LCLSFrameGetter(experiment_id=config['experiment_id'], run_number=run_number,
                     pad_detectors=config['pad_detectors'])
fg.view(debug_level=0)
