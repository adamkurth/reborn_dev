import numpy as np
from PyQt5 import uic
import pkg_resources

import pyqtgraph as pg
import bornagain.external.pyqtgraph as bpg

padviewui = pkg_resources.resource_filename('bornagain.viewers.qtviews', 'padview.ui')

class PADView(object):

    pad_geometry = []
    data = []
    rois = []
    images = []
    rings = []
    grid = None

    def __init__(self, pad_geometry=[], data=[]):

        self.pad_geometry = pad_geometry
        self.data = data

        self.app = pg.mkQApp()
        self.main_window = uic.loadUi(padviewui)
        self.viewbox = pg.ViewBox()
        self.viewbox.setAspectLocked()
        self.main_window.graphics_view.setCentralItem(self.viewbox)
        self.setup_pads()
        self.setup_histogram_tool()
        self.main_window.show()
        # self.main_window.setWindowState(self.main_window.windowState()
        #                                 & ~pg.QtCore.Qt.WindowMinimized
        #                                 | pg.QtCore.Qt.WindowActive)
        # self.main_window.activateWindow()
        # self.main_window.showMaximized()

    def setup_histogram_tool(self):

        self.main_window.histogram.setImageItems(self.images)
        # self.lut = bpg.MultiHistogramLUTWidget()
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

    def toggle_geometry_adjustment(self):

        if roi.translatable:
            self.disable_geometry_adjustment()
        else:
            self.enable_geometry_adjustment()

    def enable_geometry_adjustment(self):

        for roi in self.rois:

            roi.translatable = True
            roi.rotateAllowed = True
            roi.setPen(pg.mkPen('g'), width=4)
            roi.addRotateHandle([0, 0], [0.5, 0.5])
            roi.addRotateHandle([0, .5], [0.5, 0.5])

        self.add_grid()

    def disable_geometry_adjustment(self):

        for roi in self.rois:

            roi.translatable = False
            roi.rotateAllowed = False
            roi.setPen(None)
            for handle in roi.handles:
                roi.removeHandle(handle)
            self.remove_grid()

    def add_rings(self, radii=[], pens=None, radius_handle=False):

        if not isinstance(radii, (list,)):
            radii = [radii]

        n = len(radii)

        if pens is None:
            pens = [pg.glColor([255, 255, 255])]*n

        for i in range(0, n):
            circ = pg.CircleROI(pos=[-radii[i]*0.5]*2, size=radii[i], pen=pens[i])
            circ.translatable = False
            circ.removable = True
            self.rings.append(circ)
            self.viewbox.addItem(circ)

        if not radius_handle:

            self.hide_ring_radius_handles()

    def hide_ring_radius_handles(self):

        for circ in self.rings:
            for handle in circ.handles:
                print(circ)
                print(handle)
                circ.removeHandle(handle['item'])

    def remove_rings(self):

        pass

    def add_grid(self):

        if self.grid is None:
            self.grid = pg.GridItem()
        self.viewbox.addItem(self.grid)

    def remove_grid(self):

        if self.grid is not None:
            self.viewbox.removeItem(self.grid)

    def init_ui(self):

        mw = self.main_window
        mw.statusBar()
        mw.center()

    def start(self):

        self.app.exec_()
