# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

from bornagain import simulate
import bornagain as ba
ba.set_global('debug', 0)
from bornagain.simulate import examples
from bornagain.viewers.qtviews import PADView
from bornagain import detector
from bornagain.fileio.getters import FrameGetter
import numpy as np
import pyqtgraph as pg

pads = examples.cspad_pads()
# pads = examples.pnccd_pads()

for pad in pads:
    pad.t_vec.flat[2] = 0.3

sim = examples.lysozyme_molecule(pads=pads)


class MyFrameGetter(FrameGetter):

    def __init__(self, pads):

        FrameGetter.__init__(self)

        self.n_frames = 1
        self.current_frame = 0

        self.pads = pads
        self.simulator = examples.PDBMoleculeSimulator(pdb_file=None, pads=pads, wavelength=None, random_rotation=True)

    def get_frame(self, frame_number=0):

        self.current_frame = frame_number

        I = np.double(self.simulator.next())
        tot = np.sum(I.ravel())
        I = I*(1e6/tot)
        I = np.random.poisson(I)
        I += 1
        I = detector.split_pad_data(pads, I)

        dat = {'pad_data': I}

        return dat


frame_getter = MyFrameGetter(pads)

pad_data = frame_getter.get_next_frame()['pad_data']

padgui = PADView(pad_data=pad_data, pad_geometry=pads, logscale=True)
padgui.frame_getter = frame_getter
padgui.show_all_geom_info()
# padgui.show_pad_frames()
# x = (np.random.rand(1000, 2)-0.5)*1000
# padgui.add_scatter_plot(x[:, 0], x[:, 1], pen=pg.mkPen('g'), brush=None, width=2, pxMode=False, size=10)
# padgui.show_coordinate_axes()
# padgui.show_grid()
# padgui.show_pad_labels()
# padgui.add_rings([200, 400, 600, 800], pens=[pg.mkPen([255, 0, 0], width=2)]*4)

padgui.start()
