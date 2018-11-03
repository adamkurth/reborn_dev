

import sys

sys.path.append('..')
import bornagain as ba
from bornagain import detector
import numpy as np


def make_pad_list():

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

    return pad_geom

def test_PADGeometry():

    # TODO: actually test something...

    pad = ba.detector.PADGeometry()
    assert(pad.t_vec is None)

def test_PADAssembler():

    # TODO: check that layout is actually correct.

    pad_geom = make_pad_list()

    assembler = ba.detector.PADAssembler(pad_geom)
    dat = [p.ones() for p in pad_geom]

    ass = assembler.assemble_data(dat)

    assert(np.min(ass) == 0)
    assert(np.max(ass) == 1)


def test_radial_profiler():

    pad_geom = make_pad_list()

    beam = [0, 0, 1]
    wav = 1.5e-10

    q_mags = np.ravel([p.q_mags(beam_vec=beam, wavelength=wav) for p in pad_geom])

    rad = detector.RadialProfiler()
    rad.make_plan(q_mags, mask=None, n_bins=100, q_range=None)

    data = np.ravel([np.random.rand(p.n_ss, p.n_fs) for p in pad_geom])

    prof = rad.get_profile(data, average=True)
    assert(np.max(prof) <= 1)
    assert(np.min(prof) >= 0)