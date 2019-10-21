# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function, unicode_literals)

from time import time
import pickle
import numpy as np
import pkg_resources
import bornagain
from bornagain.detector import PADGeometry
from bornagain.utils import warn_pyqtgraph
from bornagain.fileio.getters import FrameGetter
from bornagain.analysis.peaks import boxsnr, PeakFinder
# We are using pyqtgraph's wrapper for pyqt because it helps deal with the different APIs in pyqt5 and pyqt4...
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import uic, QtGui, QtCore #, QtWidgets
    from pyqtgraph import ImageItem
except ImportError:
    warn_pyqtgraph()

# from bornagain.external.pyqtgraph import ImageItem

padviewui = pkg_resources.resource_filename('bornagain.viewers.qtviews', 'padview.ui')
snrconfigui = pkg_resources.resource_filename('bornagain.viewers.qtviews', 'configs.ui')


def write(msg):
    """
    Write a message to the terminal.

    Arguments:
        msg: The text message to write

    Returns: None

    """

    print(msg)


class PADView(object):

    r"""
    This is supposed to be an easy way to view PAD data, particularly if you have multiple
    detector panels.  You can set it up by providing a list of :class:`PADGeometry` instances
    along with a list of data arrays.

    It is a work in progress...
    """

    # Note that most of the interface was created using the QT Designer tool.  There are many attributes that are
    # not visible here.
    debug_level = 0  # Levels are 0: no messages, 1: basic messages, 2: more verbose, 3: extremely verbose
    logscale = False
    raw_data = None   # Dictionary with 'pad_data' and 'peaks' keys
    processed_data = None  # Dictionary with 'pad_data' and 'peaks' keys
    pad_geometry = []
    crystfel_geom_file_name = None
    pad_labels = None
    mask_data = None
    mask_images = None
    mask_color = None
    _mask_rois = None
    images = None
    scatter_plots = None
    rings = []
    grid = None
    coord_axes = None
    scan_arrows = None
    frame_getter = FrameGetter()
    _px_mode = False
    _shortcuts = None
    _status_string_mouse = ""
    _status_string_getter = " Frame 1 of 1 | "
    evt = None
    show_true_fast_scans = False
    peak_finders = None
    do_peak_finding = False
    data_filters = None
    show_peaks = True
    peaks = None
    apply_filters = True
    data_processor = None
    widgets = {}

    peak_style = {'pen': pg.mkPen('g'), 'brush': None, 'width': 5, 'size': 10, 'pxMode': False}

    def __init__(self, pad_geometry=None, mask_data=None, logscale=False, frame_getter=None, raw_data=None,
                 debug_level=0):

        """

        Arguments:
            pad_geometry: a list of PADGeometry instances
            mask_data: a list of numpy arrays
            logscale: apply log to data before viewing
            frame_getter: a subclass of the FrameGetter class
        """

        self.debug_level = debug_level
        self.debug('__init__()')

        self.logscale = logscale
        self.mask_data = mask_data
        self.pad_geometry = pad_geometry

        if raw_data is not None:
            if isinstance(raw_data, dict):
                pass
            elif isinstance(raw_data, list):
                raw_data = {'pad_data': raw_data}
            else:
                raw_data = {'pad_data': [raw_data]} # Assuming it's a numpy array...
            self.raw_data = raw_data

        if frame_getter is not None:
            self.frame_getter = frame_getter
            try:
                self.raw_data = self.frame_getter.get_frame(0)
            except:
                self.debug('Failed to get raw data from frame_getter')

        # Possibly, the frame getter has pad_geometry info -- let's have a look:
        if self.pad_geometry is None:
            self.debug('PAD geometry was not supplied at initialization.')
            try:
                self.pad_geometry = self.frame_getter.pad_geometry
            except AttributeError:
                self.debug('Failed to get geometry from frame_getter')
        if self.pad_geometry is None:
            self.debug('Making up some garbage PAD geometry instances')
            pad_geometry = []
            shft = 0
            for dat in self.raw_data['pad_data']:
                pad = PADGeometry(distance=1.0, pixel_size=1.0, shape=dat.shape)
                pad.t_vec[0] += shft
                shft += pad.shape()[0]
                pad_geometry.append(pad)
            self.pad_geometry = pad_geometry

        self.app = pg.mkQApp()
        self.main_window = uic.loadUi(padviewui)
        self.viewbox = pg.ViewBox()
        self.viewbox.invertX()
        self.viewbox.setAspectLocked()
        self.main_window.graphics_view.setCentralItem(self.viewbox)
        self.setup_mouse_interactions()
        self.setup_shortcuts()
        self.setup_menu()
        self.setup_widgets()
        self.main_window.statusbar.setStyleSheet("background-color:rgb(30, 30, 30);color:rgb(255,0,255);"
                                                 "font-weight:bold;font-family:monospace;")
        if self.raw_data is not None:
            self.setup_pads()
            self.show_frame()
        self.main_window.show()

        self.debug('__init__ complete')

    def debug(self, msg, val=1):
        r"""
        Print debug messages according to the self.debug variable.

        Arguments:
            msg: The text message to print.
            val: How verbose to be.
                0: don't print anything
                1: basic messages
                2: more verbose messages
                3: extremely verbose
        Returns: None
        """
        if self.debug_level >= val:
            print(msg)

    def _do_nothing(self):

        return None

    def _print_something(self):

        write("something happened")

    def close_main_window(self):

        self.debug('close_main_window()')

        for key in self.widgets.keys():
            self.widgets[key].close()
        self.main_window.destroy()

    def setup_widgets(self):

        self.debug('setup_widgets()')

        snr_config = SNRConfigWidget()
        snr_config.values_changed.connect(self.update_snr_filter_params)
        self.widgets['SNR Config'] = snr_config

    def setup_mouse_interactions(self):

        self.debug('setup_mouse_interactions()')

        self.proxy = pg.SignalProxy(self.viewbox.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

    def setup_menu(self):

        self.debug('setup_menu()')

        mw = self.main_window
        mw.actionGrid.triggered.connect(self.toggle_grid)
        mw.actionRings.triggered.connect(self.edit_ring_radii)
        mw.actionMaskVisible.triggered.connect(self.toggle_masks)
        mw.actionRectangleROIVisible.triggered.connect(self.toggle_rois)
        mw.actionMask_circle.triggered.connect(self.add_circle_roi)
        mw.actionPeaksVisible.triggered.connect(self.toggle_peaks)
        mw.actionCustomFilter.triggered.connect(self.toggle_filter)
        mw.actionLocal_SNR.triggered.connect(self.show_snr_filter_widget)
        mw.actionSave_Masks.triggered.connect(self.save_masks)
        mw.actionLoad_Masks.triggered.connect(self.load_masks)
        mw.actionPanel_IDs.triggered.connect(self.toggle_pad_labels)
        mw.actionBeam_position.triggered.connect(self.toggle_coordinate_axes)
        mw.actionOpen_data_file.triggered.connect(self.open_data_file)
        mw.actionShow_scan_directions.triggered.connect(self.toggle_fast_scan_directions)
        mw.actionMask_panel_edges.triggered.connect(self.mask_panel_edges)
        mw.actionFind_peaks.triggered.connect(self.toggle_peak_finding)
        mw.actionMask_upper_limit.triggered.connect(self.mask_upper_level)
        mw.actionMask_lower_limit.triggered.connect(self.mask_lower_level)
        mw.actionMask_limits.triggered.connect(self.mask_levels)

    def setup_shortcuts(self):

        self.debug('setup_shortcuts()')

        if self._shortcuts is None:
            self._shortcuts = []

        shortcut = lambda key, func: self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(key),
                                                                            self.main_window).activated.connect(func))

        shortcut(QtCore.Qt.Key_Right, self.show_next_frame)
        shortcut(QtCore.Qt.Key_Left, self.show_previous_frame)
        shortcut("f", self.show_next_frame)
        shortcut("b", self.show_previous_frame)
        shortcut("r", self.show_random_frame)
        shortcut("n", self.show_history_next)
        shortcut("p", self.show_history_previous)
        shortcut("Ctrl+g", self.toggle_all_geom_info)
        shortcut("Ctrl+r", self.edit_ring_radii)
        shortcut("Ctrl+a", self.toggle_coordinate_axes)
        shortcut("Ctrl+l", self.toggle_pad_labels)
        shortcut("Ctrl+s", self.increase_skip)
        shortcut("Shift+s", self.decrease_skip)
        shortcut("m", self.toggle_masks)
        shortcut("t", self.mask_hovering_roi)

    def update_status_string(self, frame_number=None, n_frames=None):

        if frame_number is not None and n_frames is not None:
            n = np.int(np.ceil(np.log10(n_frames)))
            strn = ' Frame %%%dd of %%%dd | ' % (n, n)
            self._status_string_getter = strn % (frame_number, n_frames)

        self.main_window.statusbar.showMessage(self._status_string_getter + self._status_string_mouse)

    @property
    def n_pads(self):

        if self.pad_geometry is not None:
            return len(self.pad_geometry)
        if self.get_pad_display_data() is not None:
            return len(self.get_pad_display_data())

    def setup_histogram_tool(self):

        self.debug('setup_histogram_tool()')
        self.main_window.histogram.gradient.loadPreset('flame')
        self.main_window.histogram.setImageItems(self.images)

    def set_levels(self, min_value, max_value):

        self.main_window.histogram.item.setLevels(min_value, max_value)

    def add_rectangle_roi(self, pos=(0, 0), size=(100, 100)):

        roi = pg.RectROI(pos=pos, size=size, centered=True, sideScalers=True)
        roi.name = 'rectangle'
        roi.addRotateHandle(pos=(0, 1), center=(0.5, 0.5))
        if self._mask_rois is None:
            self._mask_rois = []
        self._mask_rois.append(roi)
        self.viewbox.addItem(roi)

    def add_ellipse_roi(self, pos=(0, 0), size=(100, 100)):

        roi = pg.EllipseROI(pos=pos, size=size, centered=True, sideScalers=True)
        roi.name = 'ellipse'
        roi.addRotateHandle(pos=(0, 1), center=(0.5, 0.5))
        if self._mask_rois is None:
            self._mask_rois = []
        self._mask_rois.append(roi)
        self.viewbox.addItem(roi)

    def add_circle_roi(self, pos=(0, 0), size=100):

        roi = pg.CircleROI(pos=pos, size=size)
        roi.name = 'circle'
        if self._mask_rois is None:
            self._mask_rois = []
        self._mask_rois.append(roi)
        self.viewbox.addItem(roi)

    def hide_rois(self):

        if self._mask_rois is not None:
            for roi in self._mask_rois:
                self.viewbox.removeItem(roi)
            self._mask_rois = None

    def toggle_rois(self):

        if self._mask_rois is None:
            self.add_rectangle_roi()
        else:
            self.hide_rois()

    def increase_skip(self):

        self.frame_getter.skip = 10**np.floor(np.log10(self.frame_getter.skip)+1)

    def decrease_skip(self):

        self.frame_getter.skip = np.max([10**(np.floor(np.log10(self.frame_getter.skip))-1), 1])

    def show_coordinate_axes(self):

        if self.coord_axes is None:
            x = pg.ArrowItem(pos=(30, 0), brush=pg.mkBrush('r'), pxMode=self._px_mode, angle=180, pen=None)
            y = pg.ArrowItem(pos=(0, 30), brush=pg.mkBrush('g'), pxMode=self._px_mode, angle=-90, pen=None)
            z = pg.ScatterPlotItem([0], [0], pen=None, brush=pg.mkBrush('b'), pxMode=self._px_mode, size=15)
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

                f = p.fs_vec.ravel()/p.pixel_size()
                t = p.t_vec.ravel()/p.pixel_size()
                ang = np.arctan2(f[1], f[0])*180/np.pi + 180
                a = pg.ArrowItem(pos=(t[0], t[1]), angle=ang, brush=pg.mkBrush('r'), pen=None, pxMode=False)

                self.scan_arrows.append(a)
                self.viewbox.addItem(a)

    def hide_fast_scan_directions(self):

        if self.scan_arrows is not None:

            for a in self.scan_arrows:

                self.viewbox.removeItem(a)

            self.scan_arrows = None

    def toggle_fast_scan_directions(self):

        if self.scan_arrows is None:
            self.show_fast_scan_directions()
        else:
            self.hide_fast_scan_directions()

    def show_all_geom_info(self):

        self.show_pad_borders()
        self.show_grid()
        self.show_pad_labels()
        self.show_fast_scan_directions()
        self.show_coordinate_axes()

    def hide_all_geom_info(self):

        self.hide_pad_borders()
        self.hide_grid()
        self.hide_pad_labels()
        self.hide_fast_scan_directions()
        self.hide_coordinate_axes()

    def toggle_all_geom_info(self):

        if self.scan_arrows is None:
            self.show_all_geom_info()
        else:
            self.hide_all_geom_info()

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

    def scale_factor(self):

        return 1/self.pad_geometry[0].pixel_size()

    def _apply_pad_transform(self, im, p):

        # This is really aweful.  I don't know if this is the best way to do the transforms, but it's the best
        # I could do given the fact that I'm not able to track down all the needed info on how all of these
        # transforms are applied.  I can only say that a physicist did not invent this system.

        # 3D basis vectors of panel (length encodes pixel size):
        f = p.fs_vec.ravel().copy()
        s = p.ss_vec.ravel().copy()

        # 3D translation to *center* of corner pixel (first pixel in memory):
        t = p.t_vec.ravel().copy()

        # Normalize all vectors to pixel size.  This is a hack that needs to be fixed later.  Obviously, we cannot
        # show multiple detectors at different distances using this stupid pixel-based convention.
        # ps = p.pixel_size()
        scl = self.scale_factor()
        f *= scl
        s *= scl
        t *= scl

        # Strip off the z component.  Another dumb move.
        f = f[0:2]
        s = s[0:2]
        t = t[0:2]

        # Offset translation since pixel should be centered.
        # t += np.array([0.5, -0.5])

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
        # im.translate(sign * (t[1] + sign*0.5*scf), t[0]-0.5*scs)
        im.translate(sign * t[1], t[0])
        im.rotate(-sign * ang * 180 / np.pi)
        im.translate(-sign*0.5*scf, -0.5*scs)

        # Now, one would *think* that we could define a simple matrix transform, and then translate the resulting
        # image (which has been rotated and possibly skewed to imitate a 3D rotation).  I tried and failed.  But
        # It's worthwile to try again, since it would be brilliant if we could just learn how to define these
        # transforms and apply them to all relevant graphics items.
        # trans = QtGui.QTransform()
        # trans.scale(M[0, 0], M[1, 1])
        # trans.translate(t[0], t[1])
        # im.setTransform(trans)

    def _make_mask_rgba(self, mask):

        d = mask
        mask_rgba = np.zeros((d.shape[0], d.shape[1], 4))
        r = np.zeros_like(d)
        r[d == 0] = self.mask_color[0]
        g = np.zeros_like(d)
        g[d == 0] = self.mask_color[1]
        b = np.zeros_like(d)
        b[d == 0] = self.mask_color[2]
        t = np.zeros_like(d)
        t[d == 0] = 255
        mask_rgba[:, :, 0] = r
        mask_rgba[:, :, 1] = g
        mask_rgba[:, :, 2] = b
        mask_rgba[:, :, 3] = t

        return mask_rgba

    def setup_masks(self, mask_data=None):

        self.debug('setup_masks()')

        if self.pad_geometry is None:
            return

        if mask_data is not None:
            self.mask_data = mask_data

        pad_data = self.get_pad_display_data()
        if self.mask_data is None:
            self.mask_data = [np.ones_like(d) for d in pad_data]

        if self.mask_color is None:
            self.mask_color = np.array([128, 0, 0])

        for i in range(0, self.n_pads):

            d = self.mask_data[i]

#            if True: # Mask fast-scan pixels
#                d[0, 0: int(np.floor(self.pad_geometry[i].n_fs / 2))] = 1

            mask_rgba = self._make_mask_rgba(d)

            im = ImageItem(mask_rgba, autoDownsample='max')

            self._apply_pad_transform(im, self.pad_geometry[i])

            if self.mask_images is None:
                self.mask_images = []

            self.mask_images.append(im)
            self.viewbox.addItem(im)

            self.main_window.histogram.regionChanged()

    def update_masks(self, mask_data=None):

        self.debug('update_masks()')

        if mask_data is not None:
            self.mask_data = mask_data

        if self.mask_images is None:
            self.setup_masks()

        for i in range(0, self.n_pads):

            d = self.mask_data[i]

            mask_rgba = self._make_mask_rgba(d)

            self.mask_images[i].setImage(mask_rgba)

    def mask_panel_edges(self, n_pixels=None):

        if n_pixels is None or n_pixels is False:
            text, ok = QtGui.QInputDialog.getText(self.main_window, "Edge mask", "Specify number of edge pixels to mask",
                                                  QtGui.QLineEdit.Normal, "1")
            if ok:
                if text == '':
                    return
                n_pixels = int(str(text).strip())

        for i in range(len(self.mask_data)):
            self.mask_data[i] *= bornagain.detector.edge_mask(self.mask_data[i], n_pixels)

        self.update_masks()

    def mask_upper_level(self):

        val = self.main_window.histogram.item.getLevels()[1]
        dat = self.get_pad_display_data()
        for i in range(len(dat)):
            self.mask_data[i][dat[i] > val] = 0
        self.update_masks()

    def mask_lower_level(self):

        val = self.main_window.histogram.item.getLevels()[0]
        dat = self.get_pad_display_data()
        for i in range(len(dat)):
            self.mask_data[i][dat[i] < val] = 0
        self.update_masks()

    def mask_levels(self):

        val = self.main_window.histogram.item.getLevels()
        dat = self.get_pad_display_data()
        for i in range(len(dat)):
            self.mask_data[i][dat[i] < val[0]] = 0
            self.mask_data[i][dat[i] > val[1]] = 0
        self.update_masks()

    def hide_masks(self):

        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(False)

    def show_masks(self):

        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(True)

    def toggle_masks(self):

        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(not im.isVisible())

    def save_masks(self):

        options = QtGui.QFileDialog.Options()
        file_name, file_type = QtGui.QFileDialog.getSaveFileName(self.main_window, "Save Masks", "mask",
                                                          "Python Pickle (*.pkl)", options=options)
        if file_name == "":
            return

        if file_type == 'Python Pickle (*.pkl)':
            write('Saving masks: ' + file_name)
            pickle.dump(self.mask_data, open(file_name, "wb"))

    def load_masks(self):

        options = QtGui.QFileDialog.Options()
        file_name, file_type = QtGui.QFileDialog.getOpenFileName(self.main_window, "Load Masks", "mask",
                                                          "Python Pickle (*.pkl)", options=options)

        if file_name == "":
            return

        if file_type == 'Python Pickle (*.pkl)':
            write('Saving masks: ' + file_name)
            self.mask_data = pickle.load(open(file_name, "rb"))
            self.update_masks(self.mask_data)

    def mask_hovering_roi(self):

        if self._mask_rois is None:
            return

        noslice = slice(0, 1, None)

        for roi in self._mask_rois:

            if not roi.mouseHovering:
                continue

            pad_data = self.get_pad_display_data()

            for (ind, im, dat, geom) in zip(range(self.n_pads), self.images, pad_data, self.pad_geometry):

                # Using builtin function of pyqtgraph ROI to identify panels associated with the ROI...
                pslice, _ = roi.getArraySlice(dat, im, axes=(0, 1), returnSlice=True)
                if pslice[0] == noslice and pslice[1] == noslice:
                    continue
                if roi.name == 'rectangle':
                    # To find pixels in the rectangular ROI, project pixel coordinates onto the two basis vectors of
                    # the ROI.  I couldn't figure out how to do this directly with the ROI class methods.
                    sides = [roi.size()[1], roi.size()[0]]
                    corner = np.array([roi.pos()[0], roi.pos()[1]])
                    angle = roi.angle() * np.pi / 180
                    pix_pos = (geom.position_vecs() * self.scale_factor()).reshape(geom.n_ss, geom.n_fs, 3)
                    pix_pos = pix_pos[:, :, 0:2] - corner
                    v1 = np.array([-np.sin(angle), np.cos(angle)])
                    v2 = np.array([np.cos(angle), np.sin(angle)])
                    ind1 = np.dot(pix_pos, v1)
                    ind2 = np.dot(pix_pos, v2)
                    ind1 = (ind1 >= 0) * (ind1 <= sides[0])
                    ind2 = (ind2 >= 0) * (ind2 <= sides[1])
                    inds = ind1*ind2
                    self.mask_data[ind][inds] = 0
                    self.mask_images[ind].setImage(self._make_mask_rgba(self.mask_data[ind]))
                if roi.name == 'circle':
                    # To find pixels in the rectangular ROI, project pixel coordinates onto the two basis vectors of
                    # the ROI.  I couldn't figure out how to do this directly with the ROI class methods.
                    radius = roi.size()[0]/2.
                    center = np.array([roi.pos()[0], roi.pos()[1]]) + radius
                    pix_pos = (geom.position_vecs() * self.scale_factor()).reshape(geom.n_ss, geom.n_fs, 3)
                    pix_pos = pix_pos[:, :, 0:2] - center
                    r = np.sqrt(pix_pos[:, :, 0]**2 + pix_pos[:, :, 1]**2)
                    self.mask_data[ind][r < radius] = 0
                    self.mask_images[ind].setImage(self._make_mask_rgba(self.mask_data[ind]))

    def stupid_pyqtgraph_fix(self, dat):

        # Deal with this stupid problem: https://github.com/pyqtgraph/pyqtgraph/issues/769
        return [np.double(d) for d in dat]

    def get_pad_display_data(self, debug=False):

        # The logic of what actually gets displayed should go here.  For now, we display processed data if it is
        # available, else we display raw data, else we display zeros based on the pad geometry.  If none of these
        # are available, this function returns none.

        # self.debug('get_pad_display_data()')

        if self.processed_data is not None:
            if 'pad_data' in self.processed_data.keys():
                dat = self.processed_data['pad_data']
                if dat:
                    # self.debug("Got self.processed_data['pad_data']")
                    return self.stupid_pyqtgraph_fix(dat)

        if self.raw_data is not None:
            if 'pad_data' in self.raw_data.keys():
                dat = self.raw_data['pad_data']
                if dat:
                    # self.debug("Got self.raw_data['pad_data']")
                    return self.stupid_pyqtgraph_fix(dat)

        if self.pad_geometry is not None:
            self.debug('No raw data found - setting display data arrays to zeros')
            return [pad.zeros() for pad in self.pad_geometry]

        return None

    def get_peak_data(self):

        if self.processed_data is not None:
            self.debug('Getting processed peak data')
            if 'peaks' in self.processed_data.keys():
                return self.processed_data['peaks']

        if self.raw_data is not None:
            self.debug('Getting raw peak data')
            if 'peaks' in self.raw_data.keys():
                return self.raw_data['peaks']

        return None

    def setup_pads(self):

        self.debug('setup_pads()')

        pad_data = self.get_pad_display_data()

        mx = np.ravel(pad_data).max()

        self.images = []

        if self.n_pads == 0:
            self.debug("Cannot setup pad display data - there are no pads to display.")

        for i in range(0, self.n_pads):

            d = pad_data[i]

            if self.logscale:
                d[d < 0] = 0
                d = np.log10(d)

            if self.show_true_fast_scans:  # For testing - show fast scan axis
                d[0, 0:int(np.floor(self.pad_geometry[i].n_fs/2))] = mx

            im = ImageItem(d) #, autoDownsample='mean')

            self._apply_pad_transform(im, self.pad_geometry[i])

            self.images.append(im)
            self.viewbox.addItem(im)

            self.main_window.histogram.regionChanged()

        self.setup_histogram_tool()
        self.setup_masks()
        self.set_levels(np.percentile(np.ravel(pad_data), 10), np.percentile(np.ravel(pad_data), 90))

    def update_pads(self):

        self.debug('update_pads()')

        if self.images is None:
            self.setup_pads()

        processed_data = self.get_pad_display_data()
        mx = np.ravel(processed_data).max()
        for i in range(0, self.n_pads):

            d = processed_data[i]

            if self.show_true_fast_scans:  # For testing - show fast scan axis
                d[0, 0:int(np.floor(self.pad_geometry[i].n_fs/2))] = mx

            if self.logscale:
                d[d < 0] = 0
                d = np.log10(d)

            self.images[i].setImage(d)

        self.main_window.histogram.regionChanged()

    def _mouse_moved(self, evt):

        if evt is None:
            return

        if self.pad_geometry is None:
            return

        pad_data = self.get_pad_display_data()

        self.evt = evt
        pos = evt[0]
        pid = -1
        ppos = (-1, -1)
        intensity = -1

        if self.images is None:
            return

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
                d = pad_data[pid]
                sh = d.shape
                if ss < sh[0] and fs < sh[1]:
                    intensity = pad_data[pid][ss, fs]

            if pid >= 0:
                self._status_string_mouse = ' Panel %2d  |  Pixel %4d,%4d  |  Value=%8g  | ' % (pid, ss, fs, intensity)
            else:
                self._status_string_mouse = ''

            self.update_status_string()

    def edit_ring_radii(self):

        text, ok = QtGui.QInputDialog.getText(self.main_window, "Enter ring radii (comma separated)", "Ring radii",
                                              QtGui.QLineEdit.Normal, "100,200")
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

    def show_pad_border(self, n, pen=None):

        if self.images is None:
            return

        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=2)
        self.images[n].setBorder(pen)

    def hide_pad_border(self, n):

        if self.images is None:
            return

        self.images[n].setBorder(None)

    def show_pad_borders(self, pen=None):

        if self.images is None:
            return

        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=1)
        for image in self.images:
            image.setBorder(pen)

    def hide_pad_borders(self):

        if self.images is None:
            return

        for image in self.images:
            image.setBorder(None)

    def show_history_next(self):

        if self.frame_getter is None:
            self.debug('no getter')
            return

        dat = self.frame_getter.get_history_next()
        self.raw_data = dat

        self.update_display_data()

    def show_history_previous(self):

        if self.frame_getter is None:
            self.debug('no getter')
            return

        dat = self.frame_getter.get_history_previous()
        self.raw_data = dat

        self.update_display_data()

    def show_next_frame(self):

        self.debug('show_next_frame()')

        if self.frame_getter is None:
            self.debug('no getter')
            return

        dat = self.frame_getter.get_next_frame()
        self.raw_data = dat
        self.update_display_data()

    def show_previous_frame(self):

        self.debug('show_previous_frame()')

        if self.frame_getter is None:
            self.debug('no getter')
            return

        dat = self.frame_getter.get_previous_frame()
        self.raw_data = dat

        self.update_display_data()

    def show_random_frame(self):

        self.debug('show_random_frame()')

        dat = self.frame_getter.get_random_frame()
        self.raw_data = dat

        self.update_display_data()

    def show_frame(self, frame_number=0):

        self.debug('show_frame()')

        if self.frame_getter is None:
            self.debug("Note: there is no frame getter configured.")
        else:
            raw_data = self.frame_getter.get_frame(frame_number=frame_number)
            if raw_data is None:
                self.debug("Note: frame getter returned None.")
            else:
                self.raw_data = raw_data
        self.update_display_data()

    def process_data(self):

        self.debug('process_data()')

        if self.data_processor is not None:
            self.data_processor()
        else:
            self.processed_data = None

    def update_display_data(self, ):

        r"""

        Update display with new data, e.g. when moving to next frame.

        Arguments:
            dat: input dictionary with keys 'pad_data', 'peaks'

        Returns:

        """

        self.debug('update_display_data()')

        if self.data_processor is not None:
            self.process_data()
        self.update_pads()
        self.remove_scatter_plots()
        if self.do_peak_finding is True:
            self.find_peaks()
        self.display_peaks()
        self.update_status_string(frame_number=self.frame_getter.current_frame, n_frames=self.frame_getter.n_frames)
        self._mouse_moved(self.evt)

    def add_scatter_plot(self, *args, **kargs):

        if self.scatter_plots is None:
            self.scatter_plots = []

        scat = pg.ScatterPlotItem(*args, **kargs)
        self.scatter_plots.append(scat)
        self.viewbox.addItem(scat)

    def remove_scatter_plots(self):

        if self.scatter_plots is not None:
            for scat in self.scatter_plots:
                self.viewbox.removeItem(scat)

        self.scatter_plots = None

    def display_peaks(self):

        self.debug('display_peaks()')

        peaks = self.get_peak_data()

        if peaks is None:
            return

        n_peaks = peaks['n_peaks']
        centroids = peaks['centroids']
        gl_fs_pos = np.empty(n_peaks)
        gl_ss_pos = np.empty(n_peaks)
        n = 0
        for i in range(self.n_pads):
            c = centroids[i]
            if c is None:
                continue
            nc = c.shape[0]
            vec = self.pad_geometry[i].indices_to_vectors(c[:, 1], c[:, 0])#.ravel()
            gl_fs_pos[n:(n + nc)] = vec[:, 0].ravel()
            gl_ss_pos[n:(n + nc)] = vec[:, 1].ravel()
            n += nc
        gl_fs_pos *= self.scale_factor()
        gl_ss_pos *= self.scale_factor()
        self.add_scatter_plot(gl_fs_pos, gl_ss_pos, **self.peak_style)

    def toggle_peaks(self):

        if self.scatter_plots is None:
            self.display_peaks()
            self.show_peaks = True
        else:
            self.remove_scatter_plots()
            self.show_peaks = False

        self.update_pads()

    def toggle_filter(self):

        if self.apply_filters is True:
            self.apply_filters = False
        else:
            self.apply_filters = True

    def show_snr_filter_widget(self):

        self.widgets['SNR Config'].show()

    def apply_snr_filter(self):

        self.debug('apply_snr_filter()')
        t = time()
        if self.snr_filter_params is None:
            return

        if self.snr_filter_params['activate'] is not True:
            return

        if self.raw_data is None:
            return

        if self.mask_data is None:
            return

        a = self.snr_filter_params['inner']
        b = self.snr_filter_params['center']
        c = self.snr_filter_params['outer']
        raw = self.raw_data['pad_data']
        mask = self.mask_data
        processed_pads = [None]*self.n_pads
        self.debug('boxsnr()')
        for i in range(self.n_pads):
            snr, signal = boxsnr(raw[i], mask[i], mask[i], a, b, c)
            m = mask[i]*(snr < 6)
            snr, signal = boxsnr(raw[i], mask[i], m, a, b, c)
            processed_pads[i] = snr
        if self.processed_data is None:
            self.processed_data = {}
        self.processed_data['pad_data'] = processed_pads
        self.debug('%g seconds' % (time()-t,))

    def update_snr_filter_params(self):

        # Get info from the widget

        self.debug("Updating SNR Filter parameters")
        vals = self.widgets['SNR Config'].get_values()
        if vals['activate']:
            self.snr_filter_params = vals
            self.data_processor = self.apply_snr_filter
            self.update_display_data()
        else:
            self.data_processor = None
            self.processed_data = None
            self.update_display_data()

    def load_geometry_file(self):

        options = QtGui.QFileDialog.Options()
        file_name, file_type = QtGui.QFileDialog.getOpenFileName(self.main_window, "Load geometry file", "",
                                                          "CrystFEL geom (*.geom)", options=options)
        if file_name == "":
            return

        if file_type == "CrystFEL geom (*.geom)":
            print('CrystFEL geom not implemented.')
            pass
            # self.pad_geometry = geometry_file_to_pad_geometry_list(file_name)
            # self.crystfel_geom_file_name = file_name

    def open_data_file(self):

        options = QtGui.QFileDialog.Options()
        file_name, file_type = QtGui.QFileDialog.getOpenFileName(self.main_window, "Load data file", "",
                                                          "Cheetah CXI (*.cxi)", options=options)

        if file_name == "":
            return

        if file_type == 'Cheetah CXI (*.cxi)':
            print('Cheetah CXI not implemented.')
            pass
            # if self.crystfel_geom_file_name is None:
            #     msg = QtGui.QMessageBox()
            #     msg.setText("You must load a CrystFEL Geometry file before loading a Cheetah CXI file.")
            #     msg.exec_()
            #     self.load_geometry_file()
            #     if self.crystfel_geom_file_name is None:
            #         return
            #     self.main_window.setWindowTitle(file_name)
            #
            # self.frame_getter = CheetahFrameGetter(file_name, self.crystfel_geom_file_name)

        self.show_frame(frame_number=0)

    def toggle_peak_finding(self):

        if self.do_peak_finding is False:
            self.do_peak_finding = True
        else:
            self.do_peak_finding = False

        self.update_display_data()

    def setup_peak_finders(self, params=None):

        self.debug('setup_peak_finders()')

        self.peak_finders = []
        for i in range(self.n_pads):
            self.peak_finders.append(PeakFinder(mask=self.mask_data[i], radii=(3, 6, 9)))

    def find_peaks(self):

        self.debug('find_peaks()')

        if self.peak_finders is None:
            self.setup_peak_finders()

        centroids = [None]*self.n_pads
        n_peaks = 0
        for i in range(self.n_pads):
            pfind = self.peak_finders[i]
            pfind.find_peaks(data=self.raw_data['pad_data'][i], mask=self.mask_data[i])
            n_peaks += pfind.n_labels
            centroids[i] = pfind.centroids

        self.debug('Found %d peaks' % (n_peaks))

        self.raw_data['peaks'] = {'centroids': centroids, 'n_peaks': n_peaks}

    def start(self):

        self.debug('start()')
        self.app.aboutToQuit.connect(self.stop)
        self.app.exec_()

    def stop(self):

        self.debug('stop()')
        self.app.quit()
        del self.app

    def show(self):

        self.debug('show()')
        self.main_window.show()
        # self.main_window.callback_pb_load()


class SNRConfigWidget(QtGui.QWidget):

    values_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):

        QtGui.QWidget.__init__(self, parent=parent)
        uic.loadUi(snrconfigui, self)

        self.updateButton.clicked.connect(self.send_values)
        self.activateBox.stateChanged.connect(self.send_values)

    def send_values(self):

        self.debug('send_values()')
        self.values_changed.emit()

    def get_values(self):
        self.debug('get_values()')
        dat = {}
        dat['activate'] = self.activateBox.isChecked()
        dat['inner'] = self.spinBoxInnerRadius.value()
        dat['center'] = self.spinBoxCenterRadius.value()
        dat['outer'] = self.spinBoxOuterRadius.value()
        return dat

    def debug(self, msg):

        print(msg)


