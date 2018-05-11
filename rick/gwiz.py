#!/usr/bin/env python

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import uic

sys.path.append('..')
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
        # self.setup_app()
        # self.setup_statusbar()
        # self.setup_menubar()
        # self.setup_layout()
        self.setup_graphics_view()
        self.setup_pads()
        # self.add_grid()
        # self.remove_grid()
        # self.setup_histogram_tool()
        # self.enable_geometry_adjustment()
        #
        # self.main_window.setWindowState(self.main_window.windowState() & ~pg.QtCore.Qt.WindowMinimized
        #                                 | pg.QtCore.Qt.WindowActive)
        # self.main_window.activateWindow()
        self.main_window.show()
        # self.main_window.showMaximized()


    #
    # def setup_app(self):
    #
    #     if sys.platform == "darwin":
    #         pg.QtGui.qt_mac_set_native_menubar(False)
    #
    #     # self.main_window = pg.QtGui.QMainWindow()
    #     self.main_window.setWindowTitle('gwiz!')
    #     self.main_window.resize(1200, 1200)


    # def setup_statusbar(self):
    #
    #     self.statusbar = self.main_window.statusBar()


    # def setup_menubar(self):
    #
    #     self.menubar = self.main_window.menuBar()
    #     self.menubar = pg.QtGui.QMenuBar()
    #     self.menubar.setNativeMenuBar(False)
    #     self.main_window.setMenuBar(self.menubar)
    #
    #     self.filemenu = self.menubar.addMenu(' &Test1')
    #
    #     exitAction = pg.QtGui.QAction(pg.QtGui.QIcon('exit.png'), ' &Test2', self.main_window)
    #     # exitAction.setShortcut('Ctrl+Q')
    #     # exitAction.setStatusTip('Exit gwiz!')
    #     exitAction.triggered.connect(self.app.quit)
    #     self.filemenu.addAction(exitAction)

    def setup_layout(self):

        pass
        # self.central_widget = pg.QtGui.QWidget()
        # self.main_window.setCentralWidget(self.central_widget)
        # self.layout = pg.QtGui.QGridLayout()
        # self.central_widget.setLayout(self.layout)
        # self.layout.setSpacing(0)
        # self.layout.setMargin(0)
        # self.layout.setMenuBar(self.menubar)


    def setup_graphics_view(self):

        # self.graphics_view = pg.GraphicsView()
        self.viewbox = pg.ViewBox()
        self.viewbox.setAspectLocked()
        self.graphics_view.setCentralItem(self.viewbox)
        # self.layout.addWidget(self.graphics_view, 0, 0)


    def setup_histogram_tool(self):

        self.lut = bpg.MultiHistogramLUTWidget()
        self.layout.addWidget(self.lut, 0, 1)
        self.lut.setImageItems(self.images)


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
            im = pg.ImageItem(d) #, levels=self.overall_levels)
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


# app = pg.mkQApp()
# main_window = uic.loadUi('gwiz.ui')
# main_window.show()
# app.exec_()

padgui = PADGui(data=data_list, pad_geometry=pads)
# # padgui.add_rings([100])
padgui.start()
