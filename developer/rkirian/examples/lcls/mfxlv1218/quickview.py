#!/usr/bin/env python3
import sys
import numpy as np
from reborn.external.lcls import LCLSFrameGetter
from reborn.viewers.qtviews import PADView
from config import config
run_number = 312
if len(sys.argv) > 1:
    run_number = int(sys.argv[1])
fg = LCLSFrameGetter(experiment_id=config['experiment_id'], run_number=run_number,
                     pad_detectors=config['pad_detectors'])
pv = PADView(frame_getter=fg, debug_level=1)
pv.add_rings(d_spacings=58.38e-10/np.arange(1, 6))
pv.start()
# fg.view(debug_level=1)
