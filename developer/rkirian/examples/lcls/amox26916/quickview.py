#!/usr/bin/env python
import argparse
from reborn.external.lcls import LCLSFrameGetter
from reborn.viewers.qtviews import PADView
from config import get_config
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_number", type=int, default=35, help="Run number")
    args = parser.parse_args()
    config = get_config(run_number=args.run_number)
    fg = LCLSFrameGetter(
        experiment_id=config["experiment_id"],
        run_number=args.run_number,
        pad_detectors=config["pad_detectors"],
        cachedir=config["cachedir"],
        # idx=False,
    )
    pv = PADView(frame_getter=fg, debug_level=1, percentiles=(10, 90))
    pv.set_mask_color([128, 0, 0, 75])
    pv.start()
