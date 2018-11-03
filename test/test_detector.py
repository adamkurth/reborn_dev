

import sys

sys.path.append('..')
import bornagain as ba
import numpy as np


def test_PADGeometry():

    # TODO: actually test something...

    pad = ba.detector.PADGeometry()
    assert(pad.t_vec is None)

def test_PADAssembler():

    # TODO: check that layout is actually correct.

    pad_geom = []
    pad = ba.detector.PADGeometry()
    pad.t_vec = [0, .01, .5]
    pad.fs_vec = [100e-6, 0, 0]
    pad.ss_vec = [0, 100e-6, 0]
    pad.n_fs = 100
    pad.n_ss = 150
    pad_geom.append(pad)
    pad = ba.detector.PADGeometry()
    pad.t_vec = [0, -.01, .5]
    pad.fs_vec = [100e-6, 0, 0]
    pad.ss_vec = [0, 100e-6, 0]
    pad.n_fs = 100
    pad.n_ss = 150
    pad_geom.append(pad)

    assembler = ba.detector.PADAssembler(pad_geom)
    dat = [p.ones() for p in pad_geom]

    ass = assembler.assemble_data(dat)

    assert(np.min(ass) == 0)
    assert(np.max(ass) == 1)

