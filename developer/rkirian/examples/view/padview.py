# -*- coding: utf-8 -*-

import sys
from reborn.simulate import examples
from reborn.viewers.qtviews import PADView
from reborn import detector
from reborn.fileio.getters import FrameGetter
import numpy as np

pad_geometry = detector.pnccd_pad_geometry_list()

for pad in pad_geometry:
    pad.t_vec[2] = 0.3


class MyFrameGetter(FrameGetter):

    def __init__(self, pad_geometry):

        FrameGetter.__init__(self)

        self.n_frames = 1
        self.current_frame = 0

        self.pads = pad_geometry
        self.simulator = examples.PDBMoleculeSimulator(pdb_file=None, pad_geometry=pad_geometry, random_rotation=True)

    def get_frame(self, frame_number=0):

        self.current_frame = frame_number

        I = self.simulator.next()
        tot = np.sum(I.ravel())
        I *= 1e5/tot
        I = np.random.poisson(I)

        I = detector.split_pad_data(pad_geometry, I)

        dat = {'pad_data': I}

        return dat


frame_getter = MyFrameGetter(pad_geometry)
padview = PADView(frame_getter=frame_getter, pad_geometry=pad_geometry)
# padgui.frame_getter = frame_getter
# padgui.show_all_geom_info()
# padgui.show_pad_frames()
# x = (np.random.rand(1000, 2)-0.5)*1000
# padgui.add_scatter_plot(x[:, 0], x[:, 1], pen=pg.mkPen('g'), brush=None, width=2, pxMode=False, size=10)
# padgui.show_coordinate_axes()
# padgui.show_grid()
# padgui.show_pad_labels()
# padgui.add_rings([200, 400, 600, 800], pens=[pg.mkPen([255, 0, 0], width=2)]*4)
padview.start()
