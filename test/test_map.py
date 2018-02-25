from __future__ import division

import sys

import numpy as np
from numpy.fft import fftn, ifftn, fftshift

sys.path.append("..")
import bornagain as ba
from bornagain.target import crystal, map



def test_transforms():

    cryst = crystal.structure()
    cryst.set_spacegroup('P 63')
    cryst.set_cell(1e-9, 1e-9, 1e-9, 90*np.pi/180, 90*np.pi/180, 120*np.pi/180)

    mt = map.CrystalMeshTool(cryst, 0.1e-9, 1)
    dat0 = mt.reshape(np.arange(0, mt.N**3))

    dat1 = mt.symmetry_transform(0, 1, dat0)
    dat2 = mt.symmetry_transform(1, 0, dat1)

    print(dat0.flat[0:5])
    print(dat1.flat[0:5])
    print(dat2.flat[0:5])

    assert(np.array_equal(dat0, dat2))




