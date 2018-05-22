#!/usr/bin/env python

import sys
import numpy as np
# import pyqtgraph as pg
from PyQt5 import uic

sys.path.append('..')
import bornagain.external.pyqtgraph as pg
from bornagain import detector
from bornagain.simulate.clcore import ClCore
from bornagain.simulate import atoms
from bornagain.target.crystal import structure
from bornagain.units import hc, keV
import bornagain.external.pyqtgraph as bpg
from bornagain.external import crystfel


photon_energy = 6/keV
wavelength = hc/photon_energy
pdb_file = '../examples/data/pdb/2LYZ.pdb'
geom_file = '../examples/data/crystfel/geom/pnccd_front.geom'

sim = ClCore(group_size=32, double_precision=False)

pads = crystfel.geometry_file_to_pad_geometry_list(geom_file)

cryst = structure(pdb_file)
r = cryst.r
f = atoms.get_scattering_factors(cryst.Z, photon_energy=photon_energy)
q = [pad.q_vecs(beam_vec=[0, 0, 1], wavelength=wavelength) for pad in pads]
q = np.ravel(q)

A = sim.phase_factor_qrf(q, r, f)
I = np.abs(A)**2

data_list = detector.split_pad_data(pads, I)

im = data_list[0]
mx = np.max(im)
for i in np.arange(0, 9):
    im[2**i, :] = mx
    im[:, 2**i] = mx


class PADGui(object):

    pad_geometry = []
    data = []
    rois = []
    images = []
    rings = []

    def __init__(self, pad_geometry=[], data=[]):

        self.pad_geometry = pad_geometry
        self.data = data

        self.app = pg.mkQApp()
        self.main_window = uic.loadUi('gwiz.ui')
        self.viewbox = pg.ViewBox()
        self.viewbox.setAspectLocked()
        self.main_window.graphics_view.setCentralItem(self.viewbox)
        self.setup_pads()
        self.add_grid()
        # self.remove_grid()
        self.setup_histogram_tool()
        # self.enable_geometry_adjustment()
        #
        # self.main_window.setWindowState(self.main_window.windowState() & ~pg.QtCore.Qt.WindowMinimized
        #                                 | pg.QtCore.Qt.WindowActive)
        # self.main_window.activateWindow()
        self.main_window.show()
        # self.main_window.showMaximized()


    def setup_histogram_tool(self):

        # self.lut = bpg.MultiHistogramLUTWidget()
        self.main_window.histogram.setImageItems(self.images)
        # self.viewbox2 = pg.ViewBox()
        # self.main_window.graphics_view2.setCentralItem(self.lut)


    def setup_pads(self):

        self.overall_levels = np.array([np.min(np.ravel(self.data)), np.max(np.ravel(self.data))])

        for i in range(0, len(self.pad_geometry)):
            p = self.pad_geometry[i]
            d = self.data[i]
            # d[0:10, 0:100] = np.max(d)
            s = d.shape
            t = p.t_vec.flat[0:2]/p.pixel_size()
            t[1] *= -1
            r = pg.ROI(t, s)
            r.translatable = False
            r.rotateAllowed = False
            r.setPen(None)
            im = bpg.ImageItem(d) #, levels=self.overall_levels)
            im.setAutoDownsample(2)
            im.setParentItem(r)
            self.viewbox.addItem(r)
            self.rois.append(r)
            self.images.append(im)


    def enable_geometry_adjustment(self):

        for roi in self.rois:

            roi.translatable = True
            roi.rotateAllowed = True
            roi.setPen(pg.mkPen('g'))
            roi.addRotateHandle([0, 0], [0.5, 0.5])
            roi.addRotateHandle([0, .5], [0.5, 0.5])


    def disable_geometry_adjustment(self):

        for roi in self.rois:

            roi.translatable = False
            roi.rotateAllowed = False
            roi.setPen(None)
            for handle in roi.handles:
                roi.removeHandle(handle)


    def add_rings(self, radii=[]):

        # TODO: does not work

        for r in radii:
            circ = pg.CircleROI(pos=[-r*0.5, -r*0.5], size=r)
            circ.translatable = False
            circ.removable = True
            # circ.removeHandle(circ.handles[0])
            self.rings.append(circ)
            self.viewbox.addItem(circ)

    def add_grid(self):

        self.grid = pg.GridItem()
        self.viewbox.addItem(self.grid)

    def remove_grid(self):

        self.viewbox.removeItem(self.grid)


    def init_ui(self):

        mw = self.main_window
        mw.statusBar()
        mw.center()


    def start(self):

        self.app.exec_()


padgui = PADGui(data=data_list, pad_geometry=pads)
padgui.add_rings([100])
padgui.start()
