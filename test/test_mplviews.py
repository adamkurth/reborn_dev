# coding: utf-8
import pytest
import numpy as np

from bornagain.simulate import clcore
from bornagain.viewers.mplviews import SimplePAD

@pytest.mark.gui
def test_simpleview():

    cl = clcore.ClCore()
    pad = SimplePAD(wavelen=1.)

    rs = 3*np.random.random((1000, 3))

    amps = cl.phase_factor_qrf(pad.Q_vectors, rs, np.ones(rs.shape[0])*(1 + 0j))
    pad.readout(amps)
    #pad.display()
