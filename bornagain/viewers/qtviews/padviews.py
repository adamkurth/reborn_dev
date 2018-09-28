# -*- coding: utf-8 -*-

import numpy as np
# try:
#     from PyQt5 import uic
#     from PyQt5.QtWidgets import QInputDialog, QLineEdit
#     from PyQt5.QtGui import QShortcut, QKeySequence, QTransform
# except ImportError:
#     from PyQt4 import uic
#     # from PyQt4.QtWidgets import
#     from PyQt4.QtGui import QShortcut, QKeySequence, QTransform, QInputDialog, QLineEdit

import pkg_resources

import pyqtgraph as pg
from pyqtgraph.Qt import uic, QtGui, QtCore
QShortcut = QtGui.QShortcut
QKeySequence = QtGui.QKeySequence
QTransform = QtGui.QTransform
QInputDialog = QtGui.QInputDialog
QLineEdit = QtGui.QLineEdit

pg.setConfigOptions(imageAxisOrder='row-major')
import bornagain.external.pyqtgraph as bpg
from bornagain.utils import vec_norm, vec_mag
import bornagain as ba

padviewui = pkg_resources.resource_filename('bornagain.viewers.qtviews', 'padview.ui')


class PADView(object):

    r"""
    This is supposed to be an easy way to view PAD data, particularly if you have multiple
    detector panels.  You can set it up by providing a list of :class:`PADGeometry` instances
    along with a list of data arrays.

    It is a work in progress...
    """

    # Note that most of the interface was created using the QT Designer tool.  Here are some important ones:
    # self.histogram is a bornagain.external.pyqtgraph.MultiHistogramLUTWidget

    pad_geometry = []
    pad_data = []
    pad_labels = None
    rois = []
    images = []
    scatter_plots = []
    rings = []
    grid = None
    coord_axes = None
    scan_arrows = None

    def __init__(self, pad_geometry=[], pad_data=[], logscale=False):

        self.pad_geometry = pad_geometry
        self.pad_data = pad_data
        self.logscale = logscale

        self.app = pg.mkQApp()
        self.main_window = uic.loadUi(padviewui)
        self.viewbox = pg.ViewBox()
        self.viewbox.invertX()
        self.viewbox.setAspectLocked()
        self.main_window.graphics_view.setCentralItem(self.viewbox)
        self.setup_pads()
        self.setup_histogram_tool()
        self.main_window.show()
        self.proxy = pg.SignalProxy(self.viewbox.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)
        # self.label = pg.LabelItem(justify='right')
        # self.viewbox.addItem(self.label)
        self.main_window.actionGrid.triggered.connect(self.toggle_grid)
        self.main_window.actionRings.triggered.connect(self.edit_ring_radii)

        self.grid_shortcut = QShortcut(QKeySequence("Ctrl+g"), self.main_window)
        self.grid_shortcut.activated.connect(self.toggle_grid)

        self.rings_shortcut = QShortcut(QKeySequence("Ctrl+r"), self.main_window)
        self.rings_shortcut.activated.connect(self.edit_ring_radii)

        self.coords_shortcut = QShortcut(QKeySequence("Ctrl+a"), self.main_window)
        self.coords_shortcut.activated.connect(self.toggle_coordinate_axes)

        self.label_shortcut = QShortcut(QKeySequence("Ctrl+l"), self.main_window)
        self.label_shortcut.activated.connect(self.toggle_pad_labels)


        # self.main_window.setWindowState(self.main_window.windowState()
        #                                 & ~pg.QtCore.Qt.WindowMinimized
        #                                 | pg.QtCore.Qt.WindowActive)
        # self.main_window.activateWindow()
        # self.main_window.showMaximized()

    @property
    def n_pads(self):

        return len(self.pad_geometry)

    def setup_histogram_tool(self):

        self.main_window.histogram.gradient.loadPreset('flame')
        self.main_window.histogram.setImageItems(self.images)

    def show_coordinate_axes(self):

        if self.coord_axes is None:
            x = pg.ArrowItem(pos=(30, 0), brush=pg.mkBrush('r'), pxMode=False, angle=180, pen=None)
            y = pg.ArrowItem(pos=(0, 30), brush=pg.mkBrush('g'), pxMode=False, angle=-90, pen=None)
            z = pg.ScatterPlotItem([0], [0], pen=None, brush=pg.mkBrush('b'), pxMode=False, size=15)
            self.coord_axes = [x, y, z]
            self.viewbox.addItem(z)
            self.viewbox.addItem(x)
            self.viewbox.addItem(y)

    def hide_coordinate_axes(self):

        if self.coord_axes is not None:
            for c in self.coord_axes:
                self.viewbox.removeItem(c)
            self.coord_axes = None

    def toggle_coordinate_axes(self):

        if self.coord_axes is None:
            self.show_coordinate_axes()
        else:
            self.hide_coordinate_axes()

    def show_fast_scan_directions(self):

        if self.scan_arrows is None:

            self.scan_arrows = []

            for p in self.pad_geometry:

                f = p.fs_vec.ravel()
                t = p.t_vec.ravel()
                ang = np.arctan2(f[1], f[0])*180/np.pi
                a = pg.ArrowItem(pos=(t[0], t[1]), angle=ang, brush=pg.mkBrush('r'), pen=None)

                self.scan_arrows.append(a)
                self.viewbox.addItem(a)

    def hide_fast_scan_directions(self):

        if self.scan_arrows is not None:

            for a in self.scan_arrows:

                self.viewbox.removeItem(a)

            self.scan_arrows = None

    def toggle_fast_scan_directions(self):

        if self.scan_arros is None:
            self.show_fast_scan_directions()
        else:
            self.hide_fast_scan_directions()

    def show_all_geom_info(self):

        self.show_pad_frames()
        self.show_grid()
        self.show_pad_labels()
        self.show_fast_scan_directions()
        self.show_coordinate_axes()

    def hide_all_geom_info(self):

        self.hide_pad_frames()
        self.hide_grid()
        self.hide_pad_labels()
        self.hide_fast_scan_directions()
        self.show_coordinate_axes()

    def show_pad_labels(self):

        if self.pad_labels is None:

            self.pad_labels = []

            for i in range(0, self.n_pads):

                lab = pg.TextItem(text="%d" % i, fill=pg.mkBrush('b'), color='y', anchor=(0.5, 0.5))
                g = self.pad_geometry[i]
                fs = g.fs_vec.ravel()*g.n_fs/2
                ss = g.ss_vec.ravel()*g.n_ss/2
                t = g.t_vec.ravel()
                x = (fs[0] + ss[0] + t[0])/g.pixel_size()
                y = (fs[1] + ss[1] + t[1])/g.pixel_size()
                lab.setPos(x, y)
                self.pad_labels.append(lab)
                self.viewbox.addItem(lab)

    def hide_pad_labels(self):

        if self.pad_labels is not None:
            for lab in self.pad_labels:
                self.viewbox.removeItem(lab)
            self.pad_labels = None

    def toggle_pad_labels(self):

        if self.pad_labels is None:
            self.show_pad_labels()
        else:
            self.hide_pad_labels()

    def _apply_pad_transform(self, im, p):

        # This is really aweful.  I don't know if this is the best way to do the transforms, but it's the best
        # I could do given the fact that I'm not able to track down all the needed info on how all of these
        # transforms are applied.  I can only say that a physicist did not invent this system.

        # 3D basis vectors of panel (length encodes pixel size):
        f = p.fs_vec.ravel()
        s = p.ss_vec.ravel()

        # 3D translation to *center* of corner pixel (first pixel in memory):
        t = p.t_vec.ravel()

        # Normalize all vectors to pixel size.  This is a hack that needs to be fixed later.  Obviously, we cannot
        # show multiple detectors at different distances using this stupid pixel-based convention.
        ps = p.pixel_size()
        f /= ps
        s /= ps
        t /= ps

        # Strip off the z component.  Another dumb move.
        f = f[0:2]
        s = s[0:2]
        t = t[0:2]

        # Offset translation since pixel should be centered.
        t -= np.array([0.5, 0.5])

        # These two operations set the "home" position of the panel such that the fast-scan direction
        # is along the viewbox "x" axis.  I don't know if this is the corret thing to do -- needs further
        # attention.  I would imagine that pyqtgraph has a better way to do this, but I haven't found the right
        # feature yet.
        im.scale(1, -1)
        im.rotate(-90)

        # Scale the axes.  This is a phony way to deal with detector tilts.  Better than nothing I guess.
        scf = np.sqrt(np.sum(f * f))
        scs = np.sqrt(np.sum(s * s))

        # Note that the *sign* of this scale factor takes care of the fact that 2D rotations alone
        # cannot deal with a transpose operation.  What follows is a confusing hack... must be a better way.
        sign = np.sign(np.cross(np.array([f[0], f[1], 0]), np.array([s[0], s[1], 0]))[2])

        # Here goes the re-scaling
        im.scale(sign * scs, scf)

        # Rotate the axes
        fnorm = f / np.sqrt(np.sum(f * f))
        ang = np.arctan2(fnorm[1], fnorm[0])
        # No, wait, don't rotate the axes... experimentation says to translate first despite the fact that it
        # doesn't make much sense to do things that way.  Maybe I just gave up too soon, but I found that I needed
        # to translate first...

        # Translate the scaled/rotated image.  Turns out we need to flip the sign of the translation vector
        # coordinate corresponding to the axis we flipped.  I don't know why, but I've completely given up on the
        # idea of understanding the many layers of coordinate systems and transformations...
        im.translate(sign * t[1], t[0])
        im.rotate(-sign * ang * 180 / np.pi)

        # Now, one would *think* that we could define a simple matrix transform, and then translate the resulting
        # image (which has been rotated and possibly skewed to imitate a 3D rotation).  I tried and failed.  But
        # It's worthwile to try again, since it would be brilliant if we could just learn how to define these
        # transforms and apply them to all relevant graphics items.
        # trans = QTransform()
        # trans.scale(M[0, 0], M[1, 1])
        # trans.translate(t[0], t[1])
        # im.setTransform(trans)

    def setup_pads(self, show_scans=False):

        mx = np.ravel(self.pad_data).max()

        for i in range(0, self.n_pads):

            # p = self.pad_geometry[i]
            d = self.pad_data[i]

            if self.logscale:
                d[d < 0] = 0
                d = np.log10(d)

            if show_scans:  # For testing - show fast scan axis
                d[0, 0:int(np.floor(self.pad_geometry[i].n_fs/2))] = mx

            im = bpg.ImageItem(d)

            self._apply_pad_transform(im, self.pad_geometry[i])

            self.images.append(im)
            self.viewbox.addItem(im)

    def update_pad_data(self, pad_data):

        self.pad_data = pad_data
        for i in range(0, self.n_pads):

            d = self.pad_data[i]

            if self.logscale:
                d[d < 0] = 0
                d = np.log10(d)

            im = self.images[i]
            im.setImage(d)

        self.main_window.histogram.regionChanged()



    def mouse_moved(self, evt):

        pos = evt[0]
        pid = -1
        ppos = (-1, -1)
        intensity = -1
        if self.viewbox.sceneBoundingRect().contains(pos):
            for i in range(0, len(self.images)):
                if self.images[i].sceneBoundingRect().contains(pos):
                    pid = i
                    ppos = self.images[i].mapFromScene(pos)
                    ppos = (np.floor(ppos.x()), np.floor(ppos.y()))
                    continue

            # pnt = self.viewbox.mapSceneToView(pos)
            fs = np.int(ppos[1])
            ss = np.int(ppos[0])
            if pid >= 0:
                d = self.pad_data[pid]
                sh = d.shape
                if ss < sh[0] and fs < sh[1]:
                    intensity = self.pad_data[pid][ss, fs]

            if pid >= 0:
                status = 'Panel %2d; ss=%4d; fs=%4d; I=%8g' % (pid, ss, fs, intensity)
            else:
                status = ''
            self.main_window.statusbar.showMessage(status)
            #self.label.setText(
            #    "<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>" % (
            #    pnt.x(), pnt.y()))

    def edit_ring_radii(self):

        text, ok = QInputDialog.getText(self.main_window, "Enter ring radii (comma separated)", "Ring radii", QLineEdit.Normal, "100,200")
        if ok:
            if text == '':
                self.remove_rings()
                return
            r = text.split(',')
            rad = []
            for i in range(0, len(r)):
                try:
                    rad.append(float(r[i].strip()))
                except:
                    pass
            self.remove_rings()
            self.add_rings(rad)

    def add_rings(self, radii=[], pens=None, radius_handle=False):

        if not isinstance(radii, (list,)):
            radii = [radii]

        n = len(radii)

        if pens is None:
            pens = [pg.mkPen([255, 255, 255], width=2)]*n

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
                circ.removeHandle(handle['item'])

    def remove_rings(self):

        if self.rings is None:
            return

        for i in range(0, len(self.rings)):
            self.viewbox.removeItem(self.rings[i])

    def show_grid(self):

        if self.grid is None:
            self.grid = pg.GridItem()
        self.viewbox.addItem(self.grid)

    def hide_grid(self):

        if self.grid is not None:
            self.viewbox.removeItem(self.grid)
            self.grid = None

    def toggle_grid(self):

        if self.grid is None:
            self.show_grid()
        else:
            self.hide_grid()

    def show_pad_frame(self, n, pen=None):

        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=1)
        self.images[n].setBorder(pen)

    def show_pad_frames(self, pen=None):

        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=1)
        for image in self.images:
            image.setBorder(pen)

    def add_scatter_plot(self, *args, **kargs):

        scat = pg.ScatterPlotItem(*args, **kargs)
        self.scatter_plots.append(scat)
        self.viewbox.addItem(scat)

    def start(self):

        self.app.exec_()
