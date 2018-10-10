from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys

import numpy as np

sys.path.append("..")
from bornagain.target import crystal, density


def test_transforms():

    cryst = crystal.Structure()
    cryst.set_spacegroup('P 63')
    cryst.set_cell(1e-9, 1e-9, 1e-9, 90*np.pi/180, 90*np.pi/180, 120*np.pi/180)

    for d in [0.2, 0.3, 0.4, 0.5]:

        mt = density.CrystalMeshTool(cryst, d, 1)
        dat0 = mt.reshape(np.arange(0, mt.N**3)).astype(np.float)
        dat1 = mt.symmetry_transform(0, 1, dat0)
        dat2 = mt.symmetry_transform(1, 0, dat1)

        assert(np.allclose(dat0, dat2))
