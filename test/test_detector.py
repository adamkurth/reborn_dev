

import sys

sys.path.append('..')
import bornagain as ba


def test_PADGeometry():

    pad = ba.detector.PADGeometry()
    assert(pad.t_vec is None)
