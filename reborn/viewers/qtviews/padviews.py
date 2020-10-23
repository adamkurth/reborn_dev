# -*- coding: utf-8 -*-
import os
import glob
import inspect
import importlib
import pickle
from time import time
import numpy as np
import pkg_resources
import reborn
from reborn import detector, source
from reborn.fileio.getters import FrameGetter
from reborn.analysis.peaks import boxsnr, PeakFinder
from reborn.external.pyqtgraph import MultiHistogramLUTWidget
# We are using pyqtgraph's wrapper for pyqt because it helps deal with the different APIs in pyqt5 and pyqt4...
import pyqtgraph as pg
from pyqtgraph.Qt import uic, QtGui, QtCore #, QtWidgets
from pyqtgraph import ImageItem
from reborn.viewers.qtviews import misc
# from reborn.external.pyqtgraph import ImageItem

padviewui = pkg_resources.resource_filename('reborn.viewers.qtviews', 'ui/padview.ui')
snrconfigui = pkg_resources.resource_filename('reborn.viewers.qtviews', 'ui/configs.ui')
plugin_path = pkg_resources.resource_filename('reborn.viewers.qtviews', 'plugins')

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
    beam = None
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
    _shortcuts = []
    _status_string_mouse = ""
    _status_string_getter = " Frame 1 of 1 | "
    evt = None
    show_true_fast_scans = False
    peak_finders = None
    do_peak_finding = False
    data_filters = None
    show_peaks = True
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
        self.debug(get_caller(), 1)
        self.logscale = logscale
        self.mask_data = mask_data
        self.pad_geometry = pad_geometry

        if raw_data is not None:
            if isinstance(raw_data, dict):
                pass
            else:
                raw_data = {'pad_data': reborn.utils.ensure_list(raw_data)}
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
                pad = detector.PADGeometry(distance=1.0, pixel_size=1.0, shape=dat.shape)
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
        self.set_title('padview')
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

    def set_title(self, title):
        r""" Set the title of the main window. """
        self.debug(get_caller(), 1)
        self.main_window.setWindowTitle(title)
        # self.process_events()  # Why?

    def close_main_window(self):
        r""" Not sure if this works... it is supposed to close out all children windows. """
        # FIXME: Check if this works.  I think it doesn't
        self.debug(get_caller(), 1)
        for key in self.widgets.keys():
            self.widgets[key].close()
        self.main_window.destroy()

    def setup_widgets(self):
        r""" Setup widgets that are supposed to talk to the main window. """
        self.debug(get_caller(), 1)
        snr_config = SNRConfigWidget()
        snr_config.values_changed.connect(self.update_snr_filter_params)
        self.widgets['SNR Config'] = snr_config

    def setup_mouse_interactions(self):
        r""" I don't know what this does... obviously something about mouse interactions... """
        # FIXME: What does this do?
        self.debug(get_caller(), 1)
        self.proxy = pg.SignalProxy(self.viewbox.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

    def setup_menu(self):
        r""" Connect menu items (e.g. "File") so that they actually do something when clicked. """
        self.debug(get_caller(), 1)
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

    def set_shortcut(self, shortcut, func):
        r""" Setup one keyboard shortcut so it connects to some function, assuming no arguments are needed. """
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(shortcut), self.main_window).activated.connect(func))

    def setup_shortcuts(self):
        r""" Connect all standard keyboard shortcuts to functions. """
        self.debug(get_caller(), 1)
        self.set_shortcut(QtCore.Qt.Key_Right, self.show_next_frame)
        self.set_shortcut(QtCore.Qt.Key_Left, self.show_previous_frame)
        self.set_shortcut("a", self.call_method_by_name)
        self.set_shortcut("f", self.show_next_frame)
        self.set_shortcut("b", self.show_previous_frame)
        self.set_shortcut("r", self.show_random_frame)
        self.set_shortcut("n", self.show_history_next)
        self.set_shortcut("p", self.show_history_previous)
        self.set_shortcut("c", self.choose_plugins)
        self.set_shortcut("Ctrl+g", self.toggle_all_geom_info)
        self.set_shortcut("Ctrl+r", self.edit_ring_radii)
        self.set_shortcut("Ctrl+a", self.toggle_coordinate_axes)
        self.set_shortcut("Ctrl+l", self.toggle_pad_labels)
        self.set_shortcut("Ctrl+s", self.increase_skip)
        self.set_shortcut("Shift+s", self.decrease_skip)
        self.set_shortcut("m", self.toggle_masks)
        self.set_shortcut("t", self.mask_hovering_roi)

    def update_status_string(self, frame_number=None, n_frames=None):
        r""" Update status string at the bottom of the main window. """
        self.debug(get_caller(), 3)
        if frame_number is not None and n_frames is not None:
            n = np.int(np.ceil(np.log10(n_frames)))
            strn = ' Frame %%%dd of %%%dd | ' % (n, n)
            self._status_string_getter = strn % (frame_number, n_frames)
        self.main_window.statusbar.showMessage(self._status_string_getter + self._status_string_mouse)

    @property
    def n_pads(self):
        r""" Number of PADs in the display. """
        if self.pad_geometry is not None:
            if not isinstance(self.pad_geometry, list):
                self.pad_geometry = [self.pad_geometry]
            return len(self.pad_geometry)
        if self.get_pad_display_data() is not None:
            return len(self.get_pad_display_data())

    def process_events(self):
        r""" Sometimes we need to force events to be processed... I don't understand... """
        # FIXME: Try to make sure that this function is never needed.
        self.debug(get_caller(), 1)
        pg.QtGui.QApplication.processEvents()

    def setup_histogram_tool(self):
        r""" Set up the histogram/colorbar/colormap tool that is located to the right of the PAD display. """
        self.debug(get_caller(), 1)
        self.set_preset_colormap('flame')
        self.main_window.histogram.setImageItems(self.images)

    def set_preset_colormap(self, preset='flame'):
        r""" Change the colormap to one of the presets configured in pyqtgraph.  Right-click on the colorbar to find
        out what values are allowed.
        """
        self.debug(get_caller(), 1)
        self.main_window.histogram.gradient.loadPreset(preset)
        self.main_window.histogram.setImageItems(self.images)
        pg.QtGui.QApplication.processEvents()

    def set_levels_by_percentiles(self, percents=(1, 99)):
        r""" Set upper and lower levels according to percentiles.  This is based on :func:`numpy.percentile`. """
        self.debug(get_caller(), 1)
        d = reborn.detector.concat_pad_data(self.get_pad_display_data())
        lower = np.percentile(d, percents[0])
        upper = np.percentile(d, percents[1])
        self.set_levels(lower, upper)

    def set_levels(self, min_value=None, max_value=None, percentiles=None, colormap=None):
        r""" Set the minimum and maximum levels, same as sliding the yellow sliders on the histogram tool. """
        self.debug(get_caller(), 1)
        if (min_value is None) or (max_value is None):
            self.set_levels_by_percentiles(percents=percentiles)
        else:
            self.main_window.histogram.item.setLevels(min_value, max_value)
        if colormap is not None:
            self.set_preset_colormap(colormap)

    def add_rectangle_roi(self, pos=(0, 0), size=(100, 100)):
        r""" Adds a |pyqtgraph| RectROI """
        self.debug(get_caller(), 1)
        roi = pg.RectROI(pos=pos, size=size, centered=True, sideScalers=True)
        roi.name = 'rectangle'
        roi.addRotateHandle(pos=(0, 1), center=(0.5, 0.5))
        if self._mask_rois is None:
            self._mask_rois = []
        self._mask_rois.append(roi)
        self.viewbox.addItem(roi)

    def add_ellipse_roi(self, pos=(0, 0), size=(100, 100)):
        self.debug(get_caller(), 1)
        roi = pg.EllipseROI(pos=pos, size=size, centered=True, sideScalers=True)
        roi.name = 'ellipse'
        roi.addRotateHandle(pos=(0, 1), center=(0.5, 0.5))
        if self._mask_rois is None:
            self._mask_rois = []
        self._mask_rois.append(roi)
        self.viewbox.addItem(roi)

    def add_circle_roi(self, pos=(0, 0), size=100):
        self.debug(get_caller(), 1)
        roi = pg.CircleROI(pos=pos, size=size)
        roi.name = 'circle'
        if self._mask_rois is None:
            self._mask_rois = []
        self._mask_rois.append(roi)
        self.viewbox.addItem(roi)

    def hide_rois(self):
        self.debug(get_caller(), 1)
        if self._mask_rois is not None:
            for roi in self._mask_rois:
                self.viewbox.removeItem(roi)
            self._mask_rois = None

    def toggle_rois(self):
        self.debug(get_caller(), 1)
        if self._mask_rois is None:
            self.add_rectangle_roi()
        else:
            self.hide_rois()

    def increase_skip(self):
        self.debug(get_caller(), 1)
        self.frame_getter.skip = 10**np.floor(np.log10(self.frame_getter.skip)+1)

    def decrease_skip(self):
        self.debug(get_caller(), 1)
        self.frame_getter.skip = np.max([10**(np.floor(np.log10(self.frame_getter.skip))-1), 1])

    def show_coordinate_axes(self):
        self.debug(get_caller(), 1)
        if self.coord_axes is None:
            x = pg.ArrowItem(pos=(15, 0), brush=pg.mkBrush('r'), angle=0, pen=None) #, pxMode=self._px_mode)
            y = pg.ArrowItem(pos=(0, 15), brush=pg.mkBrush('g'), angle=90, pen=None) #, pxMode=self._px_mode)
            z = pg.ScatterPlotItem([0], [0], pen=None, brush=pg.mkBrush('b'), size=15) #, pxMode=self._px_mode)
            self.coord_axes = [x, y, z]
            self.viewbox.addItem(z)
            self.viewbox.addItem(x)
            self.viewbox.addItem(y)

    def hide_coordinate_axes(self):
        self.debug(get_caller(), 1)
        if self.coord_axes is not None:
            for c in self.coord_axes:
                self.viewbox.removeItem(c)
            self.coord_axes = None

    def toggle_coordinate_axes(self):
        self.debug(get_caller(), 1)
        if self.coord_axes is None:
            self.show_coordinate_axes()
        else:
            self.hide_coordinate_axes()

    def show_fast_scan_directions(self):
        self.debug(get_caller(), 1)
        if self.scan_arrows is None:
            self.scan_arrows = []
            for p in self.pad_geometry:
                f = p.fs_vec/p.pixel_size()
                t = (p.t_vec + p.fs_vec + p.ss_vec)/p.pixel_size()
                ang = np.arctan2(f[1], f[0])*180/np.pi
                a = pg.ArrowItem(pos=(t[0], t[1]), angle=ang, brush=pg.mkBrush('r'), pen=None) #, pxMode=False)
                self.scan_arrows.append(a)
                self.viewbox.addItem(a)

    def hide_fast_scan_directions(self):
        self.debug(get_caller(), 1)
        if self.scan_arrows is not None:
            for a in self.scan_arrows:
                self.viewbox.removeItem(a)
            self.scan_arrows = None

    def toggle_fast_scan_directions(self):
        self.debug(get_caller(), 1)
        if self.scan_arrows is None:
            self.show_fast_scan_directions()
        else:
            self.hide_fast_scan_directions()

    def show_all_geom_info(self):
        self.debug(get_caller(), 1)
        self.show_pad_borders()
        self.show_grid()
        self.show_pad_labels()
        self.show_fast_scan_directions()
        self.show_coordinate_axes()

    def hide_all_geom_info(self):
        self.debug(get_caller(), 1)
        self.hide_pad_borders()
        self.hide_grid()
        self.hide_pad_labels()
        self.hide_fast_scan_directions()
        self.hide_coordinate_axes()

    def toggle_all_geom_info(self):
        self.debug(get_caller(), 1)
        if self.scan_arrows is None:
            self.show_all_geom_info()
        else:
            self.hide_all_geom_info()

    def show_pad_labels(self):
        self.debug(get_caller(), 1)
        if self.pad_labels is None:
            self.pad_labels = []
            for i in range(0, self.n_pads):
                lab = pg.TextItem(text="%d" % i, fill=pg.mkBrush(20, 20, 20, 128), color='w', anchor=(0.5, 0.5),
                                  border=pg.mkPen('w'))
                g = self.pad_geometry[i]
                fs = g.fs_vec*g.n_fs/2
                ss = g.ss_vec*g.n_ss/2
                t = g.t_vec + (g.fs_vec + g.ss_vec)/2
                x = (fs[0] + ss[0] + t[0])/g.pixel_size()
                y = (fs[1] + ss[1] + t[1])/g.pixel_size()
                lab.setPos(x, y)
                self.pad_labels.append(lab)
                self.viewbox.addItem(lab)

    def hide_pad_labels(self):
        self.debug(get_caller(), 1)
        if self.pad_labels is not None:
            for lab in self.pad_labels:
                self.viewbox.removeItem(lab)
            self.pad_labels = None

    def toggle_pad_labels(self):
        self.debug(get_caller(), 1)
        if self.pad_labels is None:
            self.show_pad_labels()
        else:
            self.hide_pad_labels()

    def scale_factor(self):
        return 1/self.pad_geometry[0].pixel_size()

    def _apply_pad_transform(self, im, p):
        self.debug(get_caller(), 2)
        f = p.fs_vec.copy()
        s = p.ss_vec.copy()
        t = p.t_vec.copy()
        f = f * self.scale_factor()
        s = s * self.scale_factor()
        t = t * self.scale_factor() + (f + s)/2.0
        trans = QtGui.QTransform()
        trans.setMatrix(s[0], s[1], 0, f[0], f[1], 0, t[0], t[1], 1)
        im.setTransform(trans)

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
        self.debug(get_caller(), 1)
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
            mask_rgba = self._make_mask_rgba(d)
            im = ImageItem(mask_rgba, autoDownsample='max')
            self._apply_pad_transform(im, self.pad_geometry[i])
            if self.mask_images is None:
                self.mask_images = []
            self.mask_images.append(im)
            self.viewbox.addItem(im)
            self.main_window.histogram.regionChanged()

    def update_masks(self, mask_data=None):
        self.debug(get_caller(), 1)
        if mask_data is not None:
            self.mask_data = mask_data
        if self.mask_images is None:
            self.setup_masks()
        for i in range(0, self.n_pads):
            self.mask_images[i].setImage(self._make_mask_rgba(self.mask_data[i]))

    def mask_panel_edges(self, n_pixels=None):
        self.debug(get_caller(), 1)
        if n_pixels is None or n_pixels is False:
            text, ok = QtGui.QInputDialog.getText(self.main_window, "Edge mask", "Specify number of edge pixels to mask",
                                                  QtGui.QLineEdit.Normal, "1")
            if ok:
                if text == '':
                    return
                n_pixels = int(str(text).strip())
        for i in range(len(self.mask_data)):
            self.mask_data[i] *= reborn.detector.edge_mask(self.mask_data[i], n_pixels)
        self.update_masks()

    def mask_upper_level(self):
        r""" Mask pixels above upper threshold in the current colormap. """
        self.debug(get_caller(), 1)
        val = self.main_window.histogram.item.getLevels()[1]
        dat = self.get_pad_display_data()
        for i in range(len(dat)):
            self.mask_data[i][dat[i] > val] = 0
        self.update_masks()

    def mask_lower_level(self):
        r""" Mask pixels above upper threshold in the current colormap. """
        self.debug(get_caller(), 1)
        val = self.main_window.histogram.item.getLevels()[0]
        dat = self.get_pad_display_data()
        for i in range(len(dat)):
            self.mask_data[i][dat[i] < val] = 0
        self.update_masks()

    def mask_levels(self):
        self.debug(get_caller(), 1)
        val = self.main_window.histogram.item.getLevels()
        dat = self.get_pad_display_data()
        for i in range(len(dat)):
            self.mask_data[i][dat[i] < val[0]] = 0
            self.mask_data[i][dat[i] > val[1]] = 0
        self.update_masks()

    def hide_masks(self):
        self.debug(get_caller(), 1)
        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(False)

    def show_masks(self):
        self.debug(get_caller(), 1)
        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(True)

    def toggle_masks(self):
        self.debug(get_caller(), 1)
        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(not im.isVisible())

    def save_masks(self):
        r""" Save list of masks in pickle or reborn mask format. """
        self.debug(get_caller(), 1)
        options = QtGui.QFileDialog.Options()
        file_name, file_type = QtGui.QFileDialog.getSaveFileName(self.main_window, "Save Masks", "mask",
                                                          "reborn Mask File (*.mask);;Python Pickle (*.pkl)",
                                                                 options=options)
        if file_name == "":
            return
        if file_type == 'Python Pickle (*.pkl)':
            write('Saving masks: ' + file_name)
            with open(file_name, "wb") as f:
                pickle.dump(self.mask_data, f)
        if file_type == 'reborn Mask File (*.mask)':
            reborn.detector.save_pad_masks(file_name, self.mask_data)

    def load_masks(self):
        r""" Load list of masks that have been saved in pickle or reborn mask format. """
        self.debug(get_caller(), 1)
        options = QtGui.QFileDialog.Options()
        file_name, file_type = QtGui.QFileDialog.getOpenFileName(self.main_window, "Load Masks", "mask",
                                                          "reborn Mask File (*.mask);;Python Pickle (*.pkl)",
                                                                 options=options)

        if file_name == "":
            return
        if file_type == 'Python Pickle (*.pkl)':
            with open(file_name, "rb") as f:
                self.mask_data = pickle.load(f)
        if file_type == 'reborn Mask File (*.mask)':
            self.mask_data = reborn.detector.load_pad_masks(file_name)
        self.update_masks(self.mask_data)
        write('Loaded mask: ' + file_name)

    def mask_hovering_roi(self):
        r""" Mask the ROI region that the mouse cursor is hovering over. """
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
                    p = geom.position_vecs() + geom.fs_vec + geom.ss_vec  # Why?
                    pix_pos = (p * self.scale_factor()).reshape(geom.n_ss, geom.n_fs, 3)
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
                    p = geom.position_vecs() + geom.fs_vec + geom.ss_vec  # Why?
                    pix_pos = (p * self.scale_factor()).reshape(geom.n_ss, geom.n_fs, 3)
                    pix_pos = pix_pos[:, :, 0:2] - center
                    r = np.sqrt(pix_pos[:, :, 0]**2 + pix_pos[:, :, 1]**2)
                    self.mask_data[ind][r < radius] = 0
                    self.mask_images[ind].setImage(self._make_mask_rgba(self.mask_data[ind]))

    def get_pad_display_data(self):
        # The logic of what actually gets displayed should go here.  For now, we display processed data if it is
        # available, else we display raw data, else we display zeros based on the pad geometry.  If none of these
        # are available, this function returns none.
        self.debug(get_caller(), 3)
        if self.processed_data is not None:
            if 'pad_data' in self.processed_data.keys():
                dat = self.processed_data['pad_data']
                if dat:
                    return [np.double(d) for d in dat]
        if self.raw_data is not None:
            if 'pad_data' in self.raw_data.keys():
                dat = self.raw_data['pad_data']
                if dat:
                    return [np.double(d) for d in dat]
        if self.pad_geometry is not None:
            self.debug('No raw data found - setting display data arrays to zeros')
            return [pad.zeros() for pad in self.pad_geometry]
        return None

    def get_peak_data(self):
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 3)
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
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
        for circ in self.rings:
            for handle in circ.handles:
                circ.removeHandle(handle['item'])

    def remove_rings(self):
        self.debug(get_caller(), 1)
        if self.rings is None:
            return
        for i in range(0, len(self.rings)):
            self.viewbox.removeItem(self.rings[i])

    def show_grid(self):
        self.debug(get_caller(), 1)
        if self.grid is None:
            self.grid = pg.GridItem()
        self.viewbox.addItem(self.grid)

    def hide_grid(self):
        self.debug(get_caller(), 1)
        if self.grid is not None:
            self.viewbox.removeItem(self.grid)
            self.grid = None

    def toggle_grid(self):
        self.debug(get_caller(), 1)
        if self.grid is None:
            self.show_grid()
        else:
            self.hide_grid()

    def show_pad_border(self, n, pen=None):
        self.debug(get_caller(), 1)
        if self.images is None:
            return
        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=2)
        self.images[n].setBorder(pen)

    def hide_pad_border(self, n):
        self.debug(get_caller(), 1)
        if self.images is None:
            return
        self.images[n].setBorder(None)

    def show_pad_borders(self, pen=None):
        self.debug(get_caller(), 1)
        if self.images is None:
            return
        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=1)
        for image in self.images:
            image.setBorder(pen)

    def hide_pad_borders(self):
        self.debug(get_caller(), 1)
        if self.images is None:
            return
        for image in self.images:
            image.setBorder(None)

    def show_history_next(self):
        self.debug(get_caller(), 1)
        if self.frame_getter is None:
            self.debug('no getter')
            return
        dat = self.frame_getter.get_history_next()
        self.raw_data = dat
        self.update_display_data()

    def show_history_previous(self):
        self.debug(get_caller(), 1)
        if self.frame_getter is None:
            self.debug('no getter', 0)
            return
        dat = self.frame_getter.get_history_previous()
        self.raw_data = dat
        self.update_display_data()

    def show_next_frame(self):
        self.debug(get_caller(), 1)
        if self.frame_getter is None:
            self.debug('no getter', 0)
            return
        dat = self.frame_getter.get_next_frame()
        self.raw_data = dat
        self.update_display_data()

    def show_previous_frame(self):
        self.debug(get_caller(), 1)
        if self.frame_getter is None:
            self.debug('no getter', 0)
            return
        dat = self.frame_getter.get_previous_frame()
        self.raw_data = dat
        self.update_display_data()

    def show_random_frame(self):
        self.debug(get_caller(), 1)
        dat = self.frame_getter.get_random_frame()
        self.raw_data = dat
        self.update_display_data()

    def show_frame(self, frame_number=0):
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
        if self.scatter_plots is None:
            self.scatter_plots = []
        scat = pg.ScatterPlotItem(*args, **kargs)
        self.scatter_plots.append(scat)
        self.viewbox.addItem(scat)

    def remove_scatter_plots(self):
        self.debug(get_caller(), 1)
        if self.scatter_plots is not None:
            for scat in self.scatter_plots:
                self.viewbox.removeItem(scat)
        self.scatter_plots = None

    def display_peaks(self):
        self.debug(get_caller(), 1)
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
            vec = self.pad_geometry[i].indices_to_vectors(c[:, 1], c[:, 0])
            gl_fs_pos[n:(n + nc)] = vec[:, 0].ravel()
            gl_ss_pos[n:(n + nc)] = vec[:, 1].ravel()
            n += nc
        gl_fs_pos *= self.scale_factor()
        gl_ss_pos *= self.scale_factor()
        self.add_scatter_plot(gl_fs_pos, gl_ss_pos, **self.peak_style)

    def toggle_peaks(self):
        self.debug(get_caller(), 1)
        if self.scatter_plots is None:
            self.display_peaks()
            self.show_peaks = True
        else:
            self.remove_scatter_plots()
            self.show_peaks = False
        self.update_pads()

    def toggle_filter(self):
        self.debug(get_caller(), 1)
        if self.apply_filters is True:
            self.apply_filters = False
        else:
            self.apply_filters = True

    def show_snr_filter_widget(self):
        self.debug(get_caller(), 1)
        self.widgets['SNR Config'].show()

    def apply_snr_filter(self):
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
        if self.do_peak_finding is False:
            self.do_peak_finding = True
        else:
            self.do_peak_finding = False
        self.update_display_data()

    def setup_peak_finders(self):
        self.debug(get_caller(), 1)
        self.peak_finders = []
        for i in range(self.n_pads):
            self.peak_finders.append(PeakFinder(mask=self.mask_data[i], radii=(3, 6, 9)))

    def choose_plugins(self):
        self.debug(get_caller(), 1)
        init = ''
        init += "subtract_median_fs\n"
        init += "#subtract_median_ss\n"
        init += "#subtract_median_radial\n"
        text = self.get_text(text=init)
        a = text.strip().split("\n")
        plugins = []
        for b in a:
            if len(b) == 0:
                continue
            if b[0] != '#':  # Ignore commented lines
                c = b.split('#')[0]
                if c != '':
                    plugins.append(c)
        if len(plugins) > 0:
            self.run_plugins(plugins)

    def run_plugins(self, module_names=['subtract_median_ss']):
        self.debug(get_caller(), 1)
        if len(module_names) <= 0:
            return
        mod = module_names[0]
        try:
            self.debug('plugin: %s' % mod)
            module = importlib.import_module(__package__+'.plugins.'+mod)
            module.plugin(self)  # This is the syntax for running a plugin
        except ImportError:
            print('Failed to find plugin %s' % module)

    def get_text(self, title="Title", label="Label", text="Text"):
        text, ok = QtGui.QInputDialog.getText(self.main_window, title, label, QtGui.QLineEdit.Normal, text)
        return text

    # def get_multiline_text(self, title="Title", label="Label", text="Text"):
    #     # FIXME: this is only Qt 5.2+
    #     text, ok = QtGui.QInputDialog.getMultilineText(self.main_window, title=title, label=label, text=text)
    #     return text

    def get_float(self, title="Title", label="Label", text="Text"):
        return float(self.get_text(title=title, label=label, text=text))

    def find_peaks(self):
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
        self.app.aboutToQuit.connect(self.stop)
        self.app.exec_()

    def stop(self):
        self.debug(get_caller(), 1)
        self.app.quit()
        del self.app

    def show(self):
        self.debug(get_caller(), 1)
        self.main_window.show()
        # self.main_window.callback_pb_load()

    def call_method_by_name(self, method_name=None, *args, **kwargs):
        r""" Call a method via it's name in string format. Try not to do this... """
        self.debug(get_caller(), 1)
        if method_name is None:
            method_name = self.get_text('Call method', 'Method name', '')
        self.debug('method_name: ' + method_name)
        method = getattr(self, method_name, None)
        if method is not None:
            method(*args, **kwargs)


class PADView2(object):

    r"""
    This is supposed to be an easy way to view PAD data, particularly if you have multiple
    detector panels.  You can set it up by providing a list of :class:`PADGeometry` instances
    along with a list of data arrays.

    It is a work in progress...
    """

    # Note that most of the interface was created using the QT Designer tool.  There are many attributes that are
    # not visible here.
    dataframe = None
    debug_level = None  # Levels are 0: no messages, 1: basic messages, 2: more verbose, 3: extremely verbose
    logscale = False
    raw_data = None   # Dictionary with 'pad_data' and 'peaks' keys
    processed_data = None  # Dictionary with 'pad_data' and 'peaks' keys
    pad_geometry = []
    beam = None
    crystfel_geom_file_name = None
    pad_labels = None
    mask_data = None
    mask_images = None
    mask_color = None
    _mask_rois = None
    images = None
    scatter_plots = None
    plot_items = None
    rings = []
    grid = None
    coord_axes = None
    scan_arrows = None
    frame_getter = FrameGetter()
    _px_mode = False
    _shortcuts = []
    _status_string_mouse = ""
    _status_string_getter = " Frame 1 of 1 | "
    evt = None
    show_true_fast_scans = False
    peak_finders = None
    do_peak_finding = False
    data_filters = None
    peaks_visible = True
    apply_filters = True
    data_processor = None
    widgets = {}
    plugins = None

    peak_style = {'pen': pg.mkPen('g'), 'brush': None, 'width': 5, 'size': 10, 'pxMode': True}

    def __init__(self, pad_geometry=None, mask_data=None, logscale=False, frame_getter=None, raw_data=None,
                 beam=None, debug_level=0):
        """
        Arguments:
            pad_geometry (|PADGeometry| list): PAD geometry information.
            mask_data (|ndarray| list): Data masks.
            raw_data (|ndarray| list or dict): The data arrays, or a dictionary with at least a 'pad_data' key.
            beam (|Beam|): X-ray beam parameters.
            logscale (bool): Log the data before viewing (because the viewer is 8bit!).
            frame_getter (|FrameGetter| subclass): Optionally, a frame getter.
        """
        self.debug_level = debug_level
        self.debug(get_caller(), 1)
        self.logscale = logscale
        self.mask_data = mask_data
        self.pad_geometry = pad_geometry
        self.beam = beam
        self.dataframe = {}

        if raw_data is not None:
            if isinstance(raw_data, dict):
                pass
            else:
                raw_data = {'pad_data': reborn.utils.ensure_list(raw_data)}
            self.raw_data = raw_data

        if frame_getter is not None:
            self.frame_getter = frame_getter
            # TODO: check that this is an appropriate dictionary
            self.raw_data = self.frame_getter.get_frame(0)

        # Possibly, the frame getter has pad_geometry info -- let's have a look:
        if self.pad_geometry is None:
            if self.frame_getter is not None:
                self.debug('Checking if the frame getter contains PAD geometry...')
                try:
                    self.pad_geometry = self.frame_getter.pad_geometry
                    self.debug('Found PAD geometry.')
                except AttributeError:
                    self.debug('Failed to get geometry from frame getter.')
        if self.pad_geometry is None:
            self.debug('WARNING: Making up some *GARBAGE* PAD geometry because you provided no specification.')
            pad_geometry = []
            shft = 0
            for dat in self.raw_data['pad_data']:
                pad = detector.PADGeometry(distance=1.0, pixel_size=1.0, shape=dat.shape)
                pad.t_vec[0] += shft
                shft += pad.shape()[0]
                pad_geometry.append(pad)
            self.pad_geometry = pad_geometry

        # Possibly, the frame getter has pad_geometry info -- let's have a look:
        if self.beam is None:
            if self.frame_getter is not None:
                self.debug('Checking if the frame getter contains x-ray beam information...')
                try:
                    self.beam = self.frame_getter.beam
                    self.debug('Found x-ray beam information.')
                except AttributeError:
                    self.debug('Failed to get x-ray beam information from frame_getter.')
        if self.beam is None:
            self.debug('WARNING: Making up some *GARBAGE* beam information because you provided no specification.')
            self.beam = source.Beam(photon_energy=9000*1.602e-19)

        if self.mask_data is None:
            if self.frame_getter is not None:
                self.debug('Checking if the frame getter contains pixel masks...')
                try:
                    self.mask_data = self.frame_getter.mask_data
                    self.debug('Found masks.')
                except AttributeError:
                    self.debug('Failed to get pixel masks from frame_getter.')
            self.mask_data = [np.ones_like(d) for d in self.raw_data['pad_data']]

        self.app = pg.mkQApp()
        self.setup_ui()
        self.viewbox = pg.ViewBox()
        self.viewbox.invertX()
        self.viewbox.setAspectLocked()
        self.graphics_view.setCentralItem(self.viewbox)
        self.setup_mouse_interactions()
        self.setup_shortcuts()
        self.setup_menubar()
        # self.setup_widgets()
        self.statusbar.setStyleSheet("background-color:rgb(30, 30, 30);color:rgb(255,0,255);"
                                                 "font-weight:bold;font-family:monospace;")
        if self.raw_data is not None:
            self.setup_pads()
            self.show_frame()
        self.main_window.show()
        self.debug('__init__ complete', 1)

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

    def setup_ui(self):

        self.main_window = QtGui.QMainWindow()
        self.menubar = self.main_window.menuBar()
        self.statusbar = self.main_window.statusBar()
        self.hbox = QtGui.QHBoxLayout()
        self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        # self.setup_widgets()
        # self.side_panel = QtGui.QWidget()
        # self.side_panel_layout = QtGui.QVBoxLayout()
        # self.side_panel_layout.setAlignment(QtCore.Qt.AlignTop)
        # box = misc.CollapsibleBox('Display') ###########################
        # lay = QtGui.QGridLayout()
        # lay.addWidget(QtGui.QLabel('CMap min:'), 1, 1)
        # maxspin = QtGui.QSpinBox()
        # lay.addWidget(maxspin, 1, 2)
        # box.setContentLayout(lay)
        # self.side_panel_layout.addWidget(box)
        # box = misc.CollapsibleBox('Peaks') ##############################
        # lay = QtGui.QGridLayout()
        # lay.addWidget(self.widget_peakfinder_config, 1, 1)
        # box.setContentLayout(lay)
        # self.side_panel_layout.addWidget(box)
        # box = misc.CollapsibleBox('Analysis') ###########################
        # lay = QtGui.QGridLayout()
        # row = 0
        # row += 1
        # lay.addWidget(QtGui.QLabel('Polarization Correction'), row, 1)
        # polarization_button = QtGui.QCheckBox()
        # # polarization_button.toggled.connect()
        # lay.addWidget(polarization_button, row, 2, alignment=QtCore.Qt.AlignCenter)
        # row += 1
        # lay.addWidget(self.widget_plugin, row, 1)
        # box.setContentLayout(lay)
        # self.side_panel_layout.addWidget(box)
        # self.side_panel.setLayout(self.side_panel_layout)
        # self.splitter.addWidget(self.side_panel)
        self.graphics_view = pg.GraphicsView()
        self.viewbox = pg.ViewBox()
        self.viewbox.invertX()
        self.viewbox.setAspectLocked()
        self.graphics_view.setCentralItem(self.viewbox)
        self.splitter.addWidget(self.graphics_view)
        self.histogram = MultiHistogramLUTWidget()
        self.splitter.addWidget(self.histogram)
        self.hbox.addWidget(self.splitter)
        self.main_widget = QtGui.QWidget()
        self.main_widget.setLayout(self.hbox)
        self.main_window.setCentralWidget(self.main_widget)

    def set_title(self, title):
        r""" Set the title of the main window. """
        self.debug(get_caller(), 1)
        self.main_window.setWindowTitle(title)
        # self.process_events()  # Why?

    # def setup_widgets(self):
    #     r""" Setup widgets that are supposed to talk to the main window. """
    #     self.debug(get_caller(), 1)
    #     snr_config = SNRConfigWidget()
    #     snr_config.values_changed.connect(self.update_snr_filter_params)
    #     self.widget_snr_config = snr_config
    #     self.widget_peakfinder_config = PeakfinderConfigWidget()
    #     self.widget_peakfinder_config.values_changed.connect(self.update_peakfinder_params)
    #     self.widget_plugin = PluginWidget(self)

    def setup_mouse_interactions(self):
        r""" I don't know what this does... obviously something about mouse interactions... """
        # FIXME: What does this do?
        self.debug(get_caller(), 1)
        self.proxy = pg.SignalProxy(self.viewbox.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

    def setup_menubar(self):
        r""" Connect menu items (e.g. "File") so that they actually do something when clicked. """
        self.debug(get_caller(), 1)
        def add_menu(append_to, name, short=None, tip=None, connect=None):
            action = QtGui.QAction(name, self.main_window)
            if short: action.setShortcut(short)
            if tip: action.setStatusTip(tip)
            if connect: action.triggered.connect(connect)
            append_to.addAction(action)
            return action
        file_menu = self.menubar.addMenu('&File')
        add_menu(file_menu, 'Open file...', connect=self.open_data_file)
        add_menu(file_menu, 'Exit', short='Ctrl+Q', connect=self.app.quit)
        geom_menu = self.menubar.addMenu('Geometry')
        add_menu(geom_menu, 'Show coordinates', connect=self.toggle_coordinate_axes)
        add_menu(geom_menu, 'Show grid', connect=self.toggle_grid)
        add_menu(geom_menu, 'Show PAD labels', connect=self.toggle_pad_labels)
        add_menu(geom_menu, 'Show scan directions', connect=self.toggle_fast_scan_directions)
        add_menu(geom_menu, 'Edit ring radii...', connect=self.edit_ring_radii)
        add_menu(geom_menu, 'Save PAD geometry...', connect=self.save_pad_geometry)
        mask_menu = self.menubar.addMenu('Mask')
        add_menu(mask_menu, 'Clear masks', connect=self.clear_masks)
        add_menu(mask_menu, 'Toggle masks visible', connect=self.toggle_masks)
        add_menu(mask_menu, 'Mask panel edges...', connect=self.mask_panel_edges)
        add_menu(mask_menu, 'Mask above upper limit', connect=self.mask_upper_level)
        add_menu(mask_menu, 'Mask below lower limit', connect=self.mask_lower_level)
        add_menu(mask_menu, 'Mask outside limits', connect=self.mask_levels)
        add_menu(mask_menu, 'Add rectangle ROI', connect=self.toggle_rois)
        add_menu(mask_menu, 'Add circle ROI', connect=self.add_circle_roi)
        add_menu(mask_menu, 'Save mask...', connect=self.save_masks)
        add_menu(mask_menu, 'Load mask...', connect=self.load_masks)
        analysis_menu = self.menubar.addMenu('Analysis')
        add_menu(analysis_menu, 'Toggle peak finding', connect=self.toggle_peak_finding)
        add_menu(analysis_menu, 'Toggle peaks visible', connect=self.toggle_peaks_visible)
        plugin_menu = self.menubar.addMenu('Plugins')
        import functools
        for plg in glob.glob(os.path.join(plugin_path, '*.py')):
            self.debug('\tSetup plugin ' + os.path.basename(plg).replace('.py', ''), 2)
            add_menu(plugin_menu, os.path.basename(plg).replace('.py', '').replace('_', ' '),
                     connect=functools.partial(self.run_plugin, os.path.basename(plg).replace('.py', '')))
        # add_menu(analysis_menu, 'SNR transform', connect=self.show_snr_filter_widget)

    def set_shortcut(self, shortcut, func):
        r""" Setup one keyboard shortcut so it connects to some function, assuming no arguments are needed. """
        self._shortcuts.append(QtGui.QShortcut(QtGui.QKeySequence(shortcut), self.main_window).activated.connect(func))

    def setup_shortcuts(self):
        r""" Connect all standard keyboard shortcuts to functions. """
        self.debug(get_caller(), 1)
        self.set_shortcut(QtCore.Qt.Key_Right, self.show_next_frame)
        self.set_shortcut(QtCore.Qt.Key_Left, self.show_previous_frame)
        self.set_shortcut("a", self.call_method_by_name)
        self.set_shortcut("f", self.show_next_frame)
        self.set_shortcut("b", self.show_previous_frame)
        self.set_shortcut("r", self.show_random_frame)
        self.set_shortcut("n", self.show_history_next)
        self.set_shortcut("p", self.show_history_previous)
        self.set_shortcut("Ctrl+g", self.toggle_all_geom_info)
        self.set_shortcut("Ctrl+r", self.edit_ring_radii)
        self.set_shortcut("Ctrl+a", self.toggle_coordinate_axes)
        self.set_shortcut("Ctrl+l", self.toggle_pad_labels)
        self.set_shortcut("Ctrl+s", self.increase_skip)
        self.set_shortcut("Shift+s", self.decrease_skip)
        self.set_shortcut("m", self.toggle_masks)
        self.set_shortcut("t", self.mask_hovering_roi)
        self.set_shortcut("h", self.mask_hovering_roi_inverse)
        self.set_shortcut("d", self.mask_hovering_roi_toggle)

    def update_status_string(self, frame_number=None, n_frames=None):
        r""" Update status string at the bottom of the main window. """
        self.debug(get_caller(), 3)
        if frame_number is not None and n_frames is not None:
            n = np.int(np.ceil(np.log10(n_frames)))
            strn = ' Frame %%%dd of %%%dd | ' % (n, n)
            self._status_string_getter = strn % (frame_number, n_frames)
        self.statusbar.showMessage(self._status_string_getter + self._status_string_mouse)

    @property
    def n_pads(self):
        r""" Number of PADs in the display. """
        if self.pad_geometry is not None:
            if not isinstance(self.pad_geometry, list):
                self.pad_geometry = [self.pad_geometry]
            return len(self.pad_geometry)
        if self.get_pad_display_data() is not None:
            return len(self.get_pad_display_data())

    # def process_events(self):
    #     r""" Sometimes we need to force events to be processed... I don't understand... """
    #     # FIXME: Try to make sure that this function is never needed.
    #     self.debug(get_caller(), 1)
    #     pg.QtGui.QApplication.processEvents()

    def setup_histogram_tool(self):
        r""" Set up the histogram/colorbar/colormap tool that is located to the right of the PAD display. """
        self.debug(get_caller(), 1)
        self.set_preset_colormap('flame')
        self.histogram.setImageItems(self.images)

    def set_preset_colormap(self, preset='flame'):
        r""" Change the colormap to one of the presets configured in pyqtgraph.  Right-click on the colorbar to find
        out what values are allowed.
        """
        self.debug(get_caller(), 1)
        self.histogram.gradient.loadPreset(preset)
        self.histogram.setImageItems(self.images)
        pg.QtGui.QApplication.processEvents()

    def set_levels_by_percentiles(self, percents=(1, 99), colormap=None):
        r""" Set upper and lower levels according to percentiles.  This is based on :func:`numpy.percentile`. """
        self.debug(get_caller(), 1)
        d = reborn.detector.concat_pad_data(self.get_pad_display_data())
        lower = np.percentile(d, percents[0])
        upper = np.percentile(d, percents[1])
        self.set_levels(lower, upper, colormap=colormap)

    def get_levels(self):
        r""" Get the minimum and maximum levels of the current image display. """
        return self.histogram.item.getLevels()

    def set_levels(self, min_value=None, max_value=None, levels=None, percentiles=None, colormap=None):
        r""" Set the minimum and maximum levels, same as sliding the yellow sliders on the histogram tool. """
        self.debug(get_caller(), 1)
        if levels is not None:
            min_value = levels[0]
            max_value = levels[1]
        if (min_value is None) or (max_value is None):
            self.set_levels_by_percentiles(percents=percentiles)
        else:
            self.histogram.item.setLevels(min_value, max_value)
        if colormap is not None:
            self.set_preset_colormap(colormap)

    def get_view_bounding_rect(self):
        r""" Bounding rectangle of everything presently visible, in view (i.e real-space, 1-meter plane) coordinates."""
        vb = self.viewbox
        return vb.mapSceneToView(vb.mapToScene(vb.rect()).boundingRect()).boundingRect().getRect()

    def add_rectangle_roi(self, pos=(0, 0), size=None):
        r""" Adds a |pyqtgraph| RectROI """
        self.debug(get_caller(), 1)
        if size is None:
            br = self.get_view_bounding_rect()
            s = min(br[2], br[3])
            size = (s/4, s/4)
        pos = (pos[0] - size[0]/2, pos[1] - size[0]/2)
        roi = pg.RectROI(pos=pos, size=size, centered=True, sideScalers=True)
        roi.name = 'rectangle'
        roi.addRotateHandle(pos=(0, 1), center=(0.5, 0.5))
        if self._mask_rois is None:
            self._mask_rois = []
        self._mask_rois.append(roi)
        self.viewbox.addItem(roi)

    # def add_ellipse_roi(self, pos=(0, 0), size=(100, 100)):
    #     self.debug(get_caller(), 1)
    #     roi = pg.EllipseROI(pos=pos, size=size, centered=True, sideScalers=True)
    #     roi.name = 'ellipse'
    #     roi.addRotateHandle(pos=(0, 1), center=(0.5, 0.5))
    #     if self._mask_rois is None:
    #         self._mask_rois = []
    #     self._mask_rois.append(roi)
    #     self.viewbox.addItem(roi)

    def add_circle_roi(self, pos=(0, 0), radius=None):
        self.debug(get_caller(), 1)
        if radius is None:
            br = self.get_view_bounding_rect()
            radius = min(br[2], br[3]) / 2
        pos = np.array(pos) - radius / 2
        roi = pg.CircleROI(pos=pos, size=radius)
        roi.name = 'circle'
        if self._mask_rois is None:
            self._mask_rois = []
        self._mask_rois.append(roi)
        self.viewbox.addItem(roi)

    def hide_rois(self):
        self.debug(get_caller(), 1)
        if self._mask_rois is not None:
            for roi in self._mask_rois:
                self.viewbox.removeItem(roi)
            self._mask_rois = None

    def toggle_rois(self):
        self.debug(get_caller(), 1)
        if self._mask_rois is None:
            self.add_rectangle_roi()
        else:
            self.hide_rois()

    def increase_skip(self):
        self.debug(get_caller(), 1)
        self.frame_getter.skip = 10**np.floor(np.log10(self.frame_getter.skip)+1)

    def decrease_skip(self):
        self.debug(get_caller(), 1)
        self.frame_getter.skip = np.max([10**(np.floor(np.log10(self.frame_getter.skip))-1), 1])

    def show_coordinate_axes(self):
        self.debug(get_caller(), 1)
        if self.coord_axes is None:
            corners = self.vector_to_view_coords(np.vstack([p.corner_position_vectors() for p in self.pad_geometry]))
            length = np.max(np.abs(corners[:, 0:2]))/10
            xl = pg.PlotDataItem([0, length], [0, 0], pen='r')
            self.viewbox.addItem(xl)
            yl = pg.PlotDataItem([0, 0], [0, length], pen='g')
            self.viewbox.addItem(yl)
            x = pg.ArrowItem(pos=(length, 0), brush=pg.mkBrush('r'), angle=0, pen=None) #, pxMode=self._px_mode)
            self.viewbox.addItem(x)
            y = pg.ArrowItem(pos=(0, length), brush=pg.mkBrush('g'), angle=90, pen=None) #, pxMode=self._px_mode)
            self.viewbox.addItem(y)
            z1 = pg.ScatterPlotItem([0], [0], pen=None, brush=pg.mkBrush('b'), size=15, symbol='x') #, pxMode=self._px_mode)
            self.viewbox.addItem(z1)
            z2 = pg.ScatterPlotItem([0], [0], pen='b', brush=None, size=15, symbol='o')  # , pxMode=self._px_mode)
            self.viewbox.addItem(z2)
            xt = pg.TextItem(text="x", color='r', anchor=(0.5, 0.5))
            xt.setPos(length*0.6, -length/3)
            self.viewbox.addItem(xt)
            yt = pg.TextItem(text="y", color='g', anchor=(0.5, 0.5))
            yt.setPos(-length/3, length*0.6)
            self.viewbox.addItem(yt)
            zt = pg.TextItem(text="z", color='b', anchor=(0.5, 0.5))
            zt.setPos(-length/3, -length/3)
            self.viewbox.addItem(zt)
            self.coord_axes = [xl, yl, x, y, z1, z2, xt, yt, zt]

    def hide_coordinate_axes(self):
        self.debug(get_caller(), 1)
        if self.coord_axes is not None:
            for c in self.coord_axes:
                self.viewbox.removeItem(c)
            self.coord_axes = None

    def toggle_coordinate_axes(self):
        self.debug(get_caller(), 1)
        if self.coord_axes is None:
            self.show_coordinate_axes()
        else:
            self.hide_coordinate_axes()

    def show_fast_scan_directions(self):
        self.debug(get_caller(), 1)
        if self.scan_arrows is None:
            self.scan_arrows = []
            for p in self.pad_geometry:
                t = p.t_vec
                f = p.fs_vec
                n = p.n_fs
                x = self.vector_to_view_coords(np.array([t, t + f*n/3]))
                plot = pg.PlotDataItem(x[:, 0], x[:, 1], pen='r')
                self.scan_arrows.append(plot)
                self.viewbox.addItem(plot)
                ang = np.arctan2(f[1], f[0])*180/np.pi
                a = pg.ArrowItem(pos=(x[1, 0], x[1, 1]), angle=ang, brush=pg.mkBrush('r'), pen=None)#, pxMode=False)
                self.scan_arrows.append(a)
                self.viewbox.addItem(a)

    def hide_fast_scan_directions(self):
        self.debug(get_caller(), 1)
        if self.scan_arrows is not None:
            for a in self.scan_arrows:
                self.viewbox.removeItem(a)
            self.scan_arrows = None

    def toggle_fast_scan_directions(self):
        self.debug(get_caller(), 1)
        if self.scan_arrows is None:
            self.show_fast_scan_directions()
        else:
            self.hide_fast_scan_directions()

    def show_all_geom_info(self):
        self.debug(get_caller(), 1)
        self.show_pad_borders()
        self.show_grid()
        self.show_pad_labels()
        self.show_fast_scan_directions()
        self.show_coordinate_axes()

    def hide_all_geom_info(self):
        self.debug(get_caller(), 1)
        self.hide_pad_borders()
        self.hide_grid()
        self.hide_pad_labels()
        self.hide_fast_scan_directions()
        self.hide_coordinate_axes()

    def toggle_all_geom_info(self):
        self.debug(get_caller(), 1)
        if self.scan_arrows is None:
            self.show_all_geom_info()
        else:
            self.hide_all_geom_info()

    def show_pad_labels(self):
        self.debug(get_caller(), 1)
        if self.pad_labels is None:
            self.pad_labels = []
            for i in range(0, self.n_pads):
                lab = pg.TextItem(text="%d" % i, fill=pg.mkBrush(20, 20, 20, 128), color='w', anchor=(0.5, 0.5),
                                  border=pg.mkPen('w'))
                g = self.pad_geometry[i]
                fs = g.fs_vec*g.n_fs/2
                ss = g.ss_vec*g.n_ss/2
                t = g.t_vec + (g.fs_vec + g.ss_vec)/2
                x = (fs[0] + ss[0] + t[0])
                y = (fs[1] + ss[1] + t[1])
                vec = self.pad_geometry[i].center_pos_vec()
                vec = self.vector_to_view_coords(vec)
                lab.setPos(vec[0], vec[1])
                self.pad_labels.append(lab)
                self.viewbox.addItem(lab)

    def hide_pad_labels(self):
        self.debug(get_caller(), 1)
        if self.pad_labels is not None:
            for lab in self.pad_labels:
                self.viewbox.removeItem(lab)
            self.pad_labels = None

    def toggle_pad_labels(self):
        self.debug(get_caller(), 1)
        if self.pad_labels is None:
            self.show_pad_labels()
        else:
            self.hide_pad_labels()

    def _apply_pad_transform(self, im, p):
        self.debug(get_caller(), 2)
        f = p.fs_vec.copy()
        s = p.ss_vec.copy()
        t = p.t_vec.copy() - (f + s)/2.0
        trans = QtGui.QTransform()
        trans.setMatrix(s[0], s[1], s[2], f[0], f[1], f[2], t[0], t[1], t[2])
        im.setTransform(trans)

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
        self.debug(get_caller(), 1)
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
            mask_rgba = self._make_mask_rgba(d)
            im = ImageItem(mask_rgba, autoDownsample='max')
            self._apply_pad_transform(im, self.pad_geometry[i])
            if self.mask_images is None:
                self.mask_images = []
            self.mask_images.append(im)
            self.viewbox.addItem(im)
            self.histogram.regionChanged()

    def update_masks(self, mask_data=None):
        self.debug(get_caller(), 1)
        if mask_data is not None:
            self.mask_data = mask_data
        if self.mask_images is None:
            self.setup_masks()
        for i in range(0, self.n_pads):
            self.mask_images[i].setImage(self._make_mask_rgba(self.mask_data[i]))

    def mask_panel_edges(self, n_pixels=None):
        self.debug(get_caller(), 1)
        if n_pixels is None or n_pixels is False:
            text, ok = QtGui.QInputDialog.getText(self.main_window, "Edge mask", "Specify number of edge pixels to mask",
                                                  QtGui.QLineEdit.Normal, "1")
            if ok:
                if text == '':
                    return
                n_pixels = int(str(text).strip())
        for i in range(len(self.mask_data)):
            self.mask_data[i] *= reborn.detector.edge_mask(self.mask_data[i], n_pixels)
        self.update_masks()

    def mask_upper_level(self):
        r""" Mask pixels above upper threshold in the current colormap. """
        self.debug(get_caller(), 1)
        val = self.histogram.item.getLevels()[1]
        dat = self.get_pad_display_data()
        for i in range(len(dat)):
            self.mask_data[i][dat[i] > val] = 0
        self.update_masks()

    def mask_lower_level(self):
        r""" Mask pixels above upper threshold in the current colormap. """
        self.debug(get_caller(), 1)
        val = self.histogram.item.getLevels()[0]
        dat = self.get_pad_display_data()
        for i in range(len(dat)):
            self.mask_data[i][dat[i] < val] = 0
        self.update_masks()

    def mask_levels(self):
        self.debug(get_caller(), 1)
        val = self.histogram.item.getLevels()
        dat = self.get_pad_display_data()
        for i in range(len(dat)):
            self.mask_data[i][dat[i] < val[0]] = 0
            self.mask_data[i][dat[i] > val[1]] = 0
        self.update_masks()

    def hide_masks(self):
        self.debug(get_caller(), 1)
        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(False)

    def show_masks(self):
        self.debug(get_caller(), 1)
        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(True)

    def toggle_masks(self):
        self.debug(get_caller(), 1)
        if self.mask_images is not None:
            for im in self.mask_images:
                im.setVisible(not im.isVisible())

    def save_masks(self):
        r""" Save list of masks in pickle or reborn mask format. """
        self.debug(get_caller(), 1)
        options = QtGui.QFileDialog.Options()
        file_name, file_type = QtGui.QFileDialog.getSaveFileName(self.main_window, "Save Masks", "mask",
                                                          "reborn Mask File (*.mask);;Python Pickle (*.pkl)",
                                                                 options=options)
        if file_name == "":
            return
        if file_type == 'Python Pickle (*.pkl)':
            write('Saving masks: ' + file_name)
            with open(file_name, "wb") as f:
                pickle.dump(self.mask_data, f)
        if file_type == 'reborn Mask File (*.mask)':
            reborn.detector.save_pad_masks(file_name, self.mask_data)

    def load_masks(self):
        r""" Load list of masks that have been saved in pickle or reborn mask format. """
        self.debug(get_caller(), 1)
        options = QtGui.QFileDialog.Options()
        file_name, file_type = QtGui.QFileDialog.getOpenFileName(self.main_window, "Load Masks", "mask",
                                                          "reborn Mask File (*.mask);;Python Pickle (*.pkl)",
                                                                 options=options)
        if file_name == "":
            return
        if file_type == 'Python Pickle (*.pkl)':
            with open(file_name, "rb") as f:
                self.mask_data = pickle.load(f)
        if file_type == 'reborn Mask File (*.mask)':
            self.mask_data = reborn.detector.load_pad_masks(file_name)
        self.update_masks(self.mask_data)
        write('Loaded mask: ' + file_name)

    def mask_hovering_roi(self, setval=0, toggle=False, mask_outside=False):
        r""" Mask the ROI region that the mouse cursor is hovering over. """
        self.debug(get_caller(), 1)
        if self._mask_rois is None: return
        roi = [r for r in self._mask_rois if r.mouseHovering]
        if len(roi) == 0: return
        roi = roi[0]
        p_vecs = np.vstack([p.position_vecs() for p in self.pad_geometry])
        v_vecs = self.vector_to_view_coords(p_vecs)[:, 0:2]
        mask = detector.concat_pad_data(self.mask_data)
        if roi.name == 'rectangle':  # Find all pixels within the rectangle
            self.debug('\tmask rectangle roi', 1)
            sides = [roi.size()[1], roi.size()[0]]
            corner = np.array([roi.pos()[0], roi.pos()[1]])
            angle = roi.angle() * np.pi / 180
            v1 = np.array([-np.sin(angle), np.cos(angle)])
            v2 = np.array([np.cos(angle), np.sin(angle)])
            d = v_vecs - corner
            ind1 = np.dot(d, v1)
            ind2 = np.dot(d, v2)
            inds = (ind1 >= 0) * (ind1 <= sides[0]) * (ind2 >= 0) * (ind2 <= sides[1])
        elif roi.name == 'circle':
            self.debug('\tmask circle roi', 1)
            radius = roi.size()[0]/2.
            center = np.array([roi.pos()[0], roi.pos()[1]]) + radius
            inds = np.sqrt(np.sum((v_vecs - center)**2, axis=1)) < radius
        else:
            return
        if mask_outside:
            inds = -(inds - 1)
        if toggle:
            mask[inds] = -(mask[inds] - 1)
        else:
            mask[inds] = setval
        self.update_masks(detector.split_pad_data(self.pad_geometry, mask))

    def mask_hovering_roi_inverse(self):
        self.mask_hovering_roi(setval=1)

    def mask_hovering_roi_toggle(self):
        self.mask_hovering_roi(toggle=True)

    def clear_masks(self):
        if self.mask_data is not None:
            for m in range(self.n_pads):
                self.mask_data[m] = self.mask_data[m]*0 + 1
        self.update_masks(self.mask_data)

    def get_pad_display_data(self):
        # The logic of what actually gets displayed should go here.  For now, we display processed data if it is
        # available, else we display raw data, else we display zeros based on the pad geometry.  If none of these
        # are available, this function returns none.
        self.debug(get_caller(), 3)
        if self.processed_data is not None:
            if 'pad_data' in self.processed_data.keys():
                dat = self.processed_data['pad_data']
                if dat:
                    return [np.double(d) for d in dat]
        if self.raw_data is not None:
            if 'pad_data' in self.raw_data.keys():
                dat = self.raw_data['pad_data']
                if dat:
                    return [np.double(d) for d in dat]
        if self.pad_geometry is not None:
            self.debug('No raw data found - setting display data arrays to zeros')
            return [pad.zeros() for pad in self.pad_geometry]
        return None

    def set_pad_display_data(self, data, auto_levels=False, update_display=True, levels=None, percentiles=None, colormap=None):
        if type(data) == dict:
            if 'pad_data' in dict.keys():
                self.processed_data = data
        elif type(data) == list:
            if self.processed_data is None:
                self.processed_data = {}
            self.processed_data['pad_data'] = data
        elif type(data) == np.ndarray:
            data = detector.split_pad_data(self.pad_geometry, data)
            if self.processed_data is None:
                self.processed_data = {}
            self.processed_data['pad_data'] = data
        else:
            raise TypeError('Allowed types are dict, ndarray, list')
        if update_display:
            self.update_pads()
        if auto_levels:
            self.set_levels_by_percentiles(percents=(2, 98))
        if (levels is not None) or (percentiles is not None):
            self.set_levels(levels=levels, percentiles=percentiles, colormap=colormap)

    def setup_pads(self):
        self.debug(get_caller(), 1)
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
            self.histogram.regionChanged()
        self.setup_histogram_tool()
        self.setup_masks()
        self.set_levels(np.percentile(np.ravel(pad_data), 10), np.percentile(np.ravel(pad_data), 90))

    def update_pad_geometry(self, pad_geometry):
        self.pad_geometry = pad_geometry
        for i in range(self.n_pads):
            if self.images is not None:
                self._apply_pad_transform(self.images[i], self.pad_geometry[i])
            if self.mask_images is not None:
                self._apply_pad_transform(self.mask_images[i], self.pad_geometry[i])
        if self.pad_labels is not None:
            self.toggle_pad_labels()
            self.toggle_pad_labels()

    def update_pads(self):
        self.debug(get_caller(), 1)
        if self.images is None:
            self.setup_pads()
        processed_data = self.get_pad_display_data()
        mx = detector.concat_pad_data(processed_data).max()
        for i in range(0, self.n_pads):
            d = processed_data[i]
            if self.show_true_fast_scans:  # For testing - show fast scan axis
                d[0, 0:int(np.floor(self.pad_geometry[i].n_fs/2))] = mx
            if self.logscale:
                d[d < 0] = 0
                d = np.log10(d)
            self.images[i].setImage(d)
        self.histogram.regionChanged()

    def save_pad_geometry(self):
        r""" Save list of pad geometry specifications in json format. """
        self.debug(get_caller(), 1)
        options = QtGui.QFileDialog.Options()
        file_name, file_type = QtGui.QFileDialog.getSaveFileName(self.main_window, "Save PAD Geometry", "geometry",
                                                          "reborn PAD Geometry File (*.json);;",
                                                                 options=options)
        if file_name == "":
            return
        self.debug('Saving PAD geometry to file: %s' % file_name)
        reborn.detector.save_pad_geometry_list(file_name, self.pad_geometry)

    def vector_to_view_coords(self, vec):
        r""" If you have a vector (or vectors) pointing in some direction in space, this function will tell you the 2D
        point at which it intercepts with the view plane (the plane 1 meter away from the origin)."""
        vec = np.atleast_2d(vec)
        vec = (vec.T / np.squeeze(vec[:, 2])).T.copy()
        vec[:, 2] = 1
        return np.squeeze(vec)

    def get_pad_coords_from_view_coords(self, view_coords):
        self.debug(get_caller(), 3)
        x = view_coords[0]
        y = view_coords[1]
        pad_idx = None
        for n in range(self.n_pads):
            vec = np.array([x, y, 1])  # This vector points from origin to the plane of the scene
            ss_idx, fs_idx = self.pad_geometry[n].vectors_to_indices(vec, insist_in_pad=True)
            if np.isfinite(ss_idx[0]) and np.isfinite(fs_idx[0]):
                pad_idx = n
                break
        return ss_idx, fs_idx, pad_idx

    def get_pad_coords_from_mouse_pos(self):
        self.debug(get_caller(), 3)
        view_coords = self.get_view_coords_from_mouse_pos()
        ss_idx, fs_idx, pad_idx = self.get_pad_coords_from_view_coords(view_coords)
        self.debug('\tpad coords: '+(ss_idx, fs_idx, pad_idx).__str__(), 3)
        return ss_idx[0], fs_idx[0], pad_idx

    def get_view_coords_from_mouse_pos(self):
        r""" These are the real-space coordinates in the plane situated 1 meter from the interaction point. """
        self.debug(get_caller(), 3)
        if self.evt is None:  # Note: self.evt is updated by _mouse_moved
            return 0, 0
        sc = self.viewbox.mapSceneToView(self.evt[0])
        self.debug('\tview coords: '+sc.__str__(), 3)
        return sc.x(), sc.y()

    def _mouse_moved(self, evt):
        self.debug(get_caller(), 3)
        self.debug('\tmouse position: ' + evt.__str__(), 3)
        if evt is None:
            return
        self.evt = evt
        ss, fs, pid = self.get_pad_coords_from_mouse_pos()
        if pid is None:
            self._status_string_mouse = ''
        else:
            fs = int(np.round(fs))
            ss = int(np.round(ss))
            intensity = self.get_pad_display_data()[pid][ss, fs]
            self._status_string_mouse = ' Panel %2d  |  Pixel %4d,%4d  |  Value=%8g  | ' % (pid, ss, fs, intensity)
        self.update_status_string()

    def edit_ring_radii(self):
        self.debug(get_caller(), 1)
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
        r""" Plot rings.  Note that these are in a plane located 1 meter from the sample position; calculate the radius
        Needed for an equivalent detector at that distance.  If you know the scattering angle, the radius is tan(theta)"""
        self.debug(get_caller(), 1)
        if not isinstance(radii, (list,)):
            radii = [radii]
        n = len(radii)
        if pens is None:
            pens = [pg.mkPen([255, 255, 255], width=2)]*n
        for i in range(0, n):
            r = 2*radii[i]
            s = -r/2 
            circ = pg.CircleROI(pos=[s, s], size=r, pen=pens[i])
            circ.translatable = False
            circ.removable = True
            self.rings.append(circ)
            self.viewbox.addItem(circ)
        if not radius_handle:
            self.hide_ring_radius_handles()

    def hide_ring_radius_handles(self):
        self.debug(get_caller(), 1)
        for circ in self.rings:
            for handle in circ.handles:
                circ.removeHandle(handle['item'])

    def remove_rings(self):
        self.debug(get_caller(), 1)
        if self.rings is None:
            return
        for i in range(0, len(self.rings)):
            self.viewbox.removeItem(self.rings[i])

    def show_grid(self):
        self.debug(get_caller(), 1)
        if self.grid is None:
            self.grid = pg.GridItem()
        self.viewbox.addItem(self.grid)

    def hide_grid(self):
        self.debug(get_caller(), 1)
        if self.grid is not None:
            self.viewbox.removeItem(self.grid)
            self.grid = None

    def toggle_grid(self):
        self.debug(get_caller(), 1)
        if self.grid is None:
            self.show_grid()
        else:
            self.hide_grid()

    def show_pad_border(self, n, pen=None):
        self.debug(get_caller(), 1)
        if self.images is None:
            return
        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=2)
        self.images[n].setBorder(pen)

    def hide_pad_border(self, n):
        self.debug(get_caller(), 1)
        if self.images is None:
            return
        self.images[n].setBorder(None)

    def show_pad_borders(self, pen=None):
        self.debug(get_caller(), 1)
        if self.images is None:
            return
        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=1)
        for image in self.images:
            image.setBorder(pen)

    def hide_pad_borders(self):
        self.debug(get_caller(), 1)
        if self.images is None:
            return
        for image in self.images:
            image.setBorder(None)

    def show_history_next(self):
        self.debug(get_caller(), 1)
        if self.frame_getter is None:
            self.debug('no getter')
            return
        dat = self.frame_getter.get_history_next()
        self.raw_data = dat
        self.update_display_data()

    def show_history_previous(self):
        self.debug(get_caller(), 1)
        if self.frame_getter is None:
            self.debug('no getter', 0)
            return
        dat = self.frame_getter.get_history_previous()
        self.raw_data = dat
        self.update_display_data()

    def show_next_frame(self):
        self.debug(get_caller(), 1)
        if self.frame_getter is None:
            self.debug('no getter', 0)
            return
        dat = self.frame_getter.get_next_frame()
        self.raw_data = dat
        self.update_display_data()

    def show_previous_frame(self):
        self.debug(get_caller(), 1)
        if self.frame_getter is None:
            self.debug('no getter', 0)
            return
        dat = self.frame_getter.get_previous_frame()
        self.raw_data = dat
        self.update_display_data()

    def show_random_frame(self):
        self.debug(get_caller(), 1)
        dat = self.frame_getter.get_random_frame()
        self.raw_data = dat
        self.update_display_data()

    def show_frame(self, frame_number=0):
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
        if self.data_processor is not None:
            self.process_data()
        self.update_pads()
        self.remove_scatter_plots()
        if self.do_peak_finding is True:
            self.find_peaks()
        self.display_peaks()
        self.update_status_string(frame_number=self.frame_getter.current_frame, n_frames=self.frame_getter.n_frames)
        self._mouse_moved(self.evt)

    def add_plot_item(self, *args, **kargs):
        r"""
        Example: self.add_plot_item(x, y, pen=pg.mkPen(width=3, color='g'))
        """
        self.debug(get_caller(), 1)
        if self.plot_items is None:
            self.plot_items = []
        plot_item = pg.PlotDataItem(*args, **kargs)
        self.plot_items.append(plot_item)
        self.viewbox.addItem(plot_item)
        return plot_item

    def add_scatter_plot(self, *args, **kargs):
        self.debug(get_caller(), 1)
        if self.scatter_plots is None:
            self.scatter_plots = []
        scat = pg.ScatterPlotItem(*args, **kargs)
        self.scatter_plots.append(scat)
        self.viewbox.addItem(scat)

    def remove_scatter_plots(self):
        self.debug(get_caller(), 1)
        if self.scatter_plots is not None:
            for scat in self.scatter_plots:
                self.viewbox.removeItem(scat)
        self.scatter_plots = None

    def toggle_filter(self):
        self.debug(get_caller(), 1)
        if self.apply_filters is True:
            self.apply_filters = False
        else:
            self.apply_filters = True

    def apply_snr_filter(self):
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
        vals = self.widget_snr_config.get_values()
        if vals['activate']:
            self.snr_filter_params = vals
            self.data_processor = self.apply_snr_filter
            self.update_display_data()
        else:
            self.data_processor = None
            self.processed_data = None
            self.update_display_data()

    def load_geometry_file(self):
        self.debug(get_caller(), 1)
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
        self.debug(get_caller(), 1)
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

    def vector_coords_to_2d_display_coords(self, vecs):
        r""" Convert 3D vector coords to the equivalent coords in the 2D display plane.  This corresponds to ignoring
        the "z" coordinate, and scaling the "x,y" coordinates to that of an equivalent detector located at a distance
        of 1 meter from the origin.  Simply put: remove the z component, divide the x,y components by the z component"""
        return (vecs[:, 0:2].T/vecs[:, 2]).T.copy()

    def panel_scatter_plot(self, panel_number, ss_coords, fs_coords, style=None):
        r""" Scatter plot points given coordinates (i.e. indices) corresponding to a particular panel.  This will
        take care of the re-mapping to the display coordinates."""
        if style is None: style = self.peak_style
        vecs = self.pad_geometry[panel_number].indices_to_vectors(ss_coords, fs_coords)
        vecs = self.vector_coords_to_2d_display_coords(vecs)
        self.add_scatter_plot(vecs[:, 0], vecs[:, 1], **style)

    def display_peaks(self):
        r""" Scatter plot the peaks that are cached in the class instance. """
        self.debug(get_caller(), 1)
        peaks = self.get_peak_data()
        if peaks is None:
            return
        centroids = peaks['centroids']
        for i in range(self.n_pads):
            c = centroids[i]
            if c is not None:
                self.panel_scatter_plot(i, c[:, 1], c[:, 0])

    def show_peaks(self):
        r""" Make peak scatter plots visible. """
        self.debug(get_caller(), 1)
        self.display_peaks()
        self.peaks_visible = True
        self.update_pads()

    def hide_peaks(self):
        r""" Make peak scatter plots invisible. """
        self.debug(get_caller(), 1)
        self.remove_scatter_plots()
        self.peaks_visible = False
        self.update_pads()

    def toggle_peaks_visible(self):
        r""" Toggle peak scatter plots visible/invisible. """
        self.debug(get_caller(), 1)
        if self.peaks_visible == False:
            self.display_peaks()
            self.peaks_visible = True
        else:
            self.hide_peaks()
            self.peaks_visible = False

    def get_peak_data(self):
        r""" Fetch peak data, which might be stored in various places.
        FIXME: Need to simplify the data structure so that it is not a hassle to find peaks."""
        self.debug(get_caller(), 1)
        if self.processed_data is not None:
            self.debug('Getting processed peak data')
            if 'peaks' in self.processed_data.keys():
                return self.processed_data['peaks']
        if self.raw_data is not None:
            self.debug('Getting raw peak data')
            if 'peaks' in self.raw_data.keys():
                return self.raw_data['peaks']
        return None

    def update_peakfinder_params(self):
        r""" Reset the peak finders with new parameters.  This also launges a peakfinding job.
        FIXME: Need to make this more intelligent so that unnecessary jobs are not launched."""
        self.peakfinder_params = self.widget_peakfinder_config.get_values()
        self.setup_peak_finders()
        self.find_peaks()
        self.hide_peaks()
        if self.peakfinder_params['activate']:
            self.show_peaks()
        else:
            self.hide_peaks()

    def find_peaks(self):
        r""" Launch a peak-finding job, and cache the results.  This will not display anything. """
        self.debug(get_caller(), 1)
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

    def toggle_peak_finding(self):
        r""" Toggle peakfinding on/off.  Set this to true if you want to automatically do peakfinding when a new
        image data is displayed. """
        self.debug(get_caller(), 1)
        if self.do_peak_finding is False:
            self.do_peak_finding = True
        else:
            self.do_peak_finding = False
        self.update_display_data()

    def setup_peak_finders(self):
        r""" Create peakfinder class instances.  We use peakfinder classes rather than functions in order to tidy up
        the data structure. """
        self.debug(get_caller(), 1)
        self.peak_finders = []
        a = self.peakfinder_params['inner']
        b = self.peakfinder_params['center']
        c = self.peakfinder_params['outer']
        t = self.peakfinder_params['snr_threshold']
        for i in range(self.n_pads):
            self.peak_finders.append(PeakFinder(mask=self.mask_data[i], radii=(a, b, c), snr_threshold=t))

    def _import_plugin_module(self, module_name):
        self.debug(get_caller(), 1)
        if module_name in self.plugins:
            return self.plugins[module_name]  # Check if module already imported and cached
        module_path = __package__+'.plugins.'+module_name
        if module_path[-3:] == '.py':
            module_path = module_path[:-3]
        self.debug('\tImporting plugin: %s' % module_path)
        module = importlib.import_module(module_path)  # Attempt to import
        if self.plugins is None: self.plugins = {}
        self.plugins[module_name] = module  # Cache the module
        return module

    def run_plugin(self, module_name):
        self.debug(get_caller(), 1)
        if self.plugins is None:
            self.plugins = {}
        if not module_name in self.plugins.keys():
            module = self._import_plugin_module(module_name)  # Get the module (import or retrieve from cache)
        else:
            module = self.plugins[module_name]
        if hasattr(module, 'plugin'):  # If the module has a plugin function, run the function and return
            module.plugin(self)
            return
        if module_name+'.widget' in self.plugins.keys():
            self.plugins[module_name+'.widget'].show()  # Check if a widget is already cached.  If so, show it.
            return
        if hasattr(module, 'Plugin'):
            plugin_instance = module.Plugin(self)  # Check if the plugin defines a class.  If so, create an instance.
            self.plugins[module_name+'.class_instance'] = plugin_instance  # Cache the instance
            self.plugins[module_name + '.widget'] = plugin_instance.widget  # Get the widget and cache it.
            plugin_instance.widget.show()  # Show the widget.
            self.debug('\tCreated plugin class instance.  Showing widget.')
            return
        self.debug('\tPlugin module has no functions or classes defined.')
        return

    def run_plugins(self, module_names=[]):
        self.debug(get_caller(), 1)
        self.debug('\tplugin module names: '+module_names.__str__(), 1)
        if len(module_names) <= 0:
            return
        for module_name in module_names:
            self.run_plugin(module_name)

    def get_text(self, title="Title", label="Label", text="Text"):
        r""" Simple popup widget that allows the capture of a text string."""
        text, ok = QtGui.QInputDialog.getText(self.main_window, title, label, QtGui.QLineEdit.Normal, text)
        return text

    def get_float(self, title="Title", label="Label", text="Text"):
        r""" Simple popup widget that allows the capture of a float number."""
        return float(self.get_text(title=title, label=label, text=text))

    def start(self):
        self.debug(get_caller(), 1)
        self.app.aboutToQuit.connect(self.stop)
        self.app.exec_()

    def stop(self):
        self.debug(get_caller(), 1)
        self.app.quit()
        del self.app

    def show(self):
        self.debug(get_caller(), 1)
        self.main_window.show()
        # self.main_window.callback_pb_load()

    def call_method_by_name(self, method_name=None, *args, **kwargs):
        r""" Call a method via it's name in string format. Try not to do this... """
        self.debug(get_caller(), 1)
        if method_name is None:
            method_name = self.get_text('Call method', 'Method name', '')
        self.debug('method_name: ' + method_name)
        method = getattr(self, method_name, None)
        if method is not None:
            method(*args, **kwargs)


def get_caller():
    r""" Get the name of the function that calls this one. """
    try:
        stack = inspect.stack()
        if len(stack) > 1:
            return inspect.stack()[1][3]
    except:
        pass
    return 'get_caller'


class PluginWidget(QtGui.QWidget):
    def __init__(self, padview=None):
        super().__init__()
        self.padview = padview
        self.plugin_files = glob.glob(os.path.join(plugin_path, '*.py'))
        self.plugin_basenames = [os.path.basename(p).replace('.py', '').replace('_', ' ').capitalize() for p in self.plugin_files]
        self.layout = QtGui.QGridLayout()
        row = 0
        row += 1
        self.layout.addWidget(QtGui.QLabel('Plugin:'), row, 1)
        self.combo_box = QtGui.QComboBox()
        self.combo_box.addItems(self.plugin_basenames)
        self.layout.addWidget(self.combo_box, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.run_plugin_button = QtGui.QPushButton("Run Plugin")
        self.run_plugin_button.clicked.connect(self.run_plugin)
        self.layout.addWidget(self.run_plugin_button, row, 1, 1, 2)
        self.setLayout(self.layout)
    def run_plugin(self):
        print()
        self.padview.run_plugins([self.combo_box.currentText().split('.')[0].replace(' ', '_').lower()+'.py'])


class PeakfinderConfigWidget(QtGui.QWidget):

    values_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle('Peakfinder Settings')
        self.layout = QtGui.QGridLayout()
        row = 0
        row += 1
        self.layout.addWidget(QtGui.QLabel('Activate Peakfinder'), row, 1)
        self.activate_peakfinder_button = QtGui.QCheckBox()
        self.activate_peakfinder_button.toggled.connect(self.send_values)
        self.layout.addWidget(self.activate_peakfinder_button, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Show SNR Transform'), row, 1)
        self.activate_snrview_button = QtGui.QCheckBox()
        self.activate_snrview_button.toggled.connect(self.send_values)
        self.layout.addWidget(self.activate_snrview_button, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('SNR Threshold'), row, 1)
        self.snr_spinbox = QtGui.QDoubleSpinBox()
        self.snr_spinbox.setMinimum(0)
        self.snr_spinbox.setValue(6)
        self.layout.addWidget(self.snr_spinbox, row, 2)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Inner Size'), row, 1)
        self.inner_spinbox = QtGui.QSpinBox()
        self.inner_spinbox.setMinimum(1)
        self.inner_spinbox.setValue(1)
        self.layout.addWidget(self.inner_spinbox, row, 2)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Center Size'), row, 1)
        self.center_spinbox = QtGui.QSpinBox()
        self.center_spinbox.setMinimum(1)
        self.center_spinbox.setValue(5)
        self.layout.addWidget(self.center_spinbox, row, 2)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Outer Size'), row, 1)
        self.outer_spinbox = QtGui.QSpinBox()
        self.outer_spinbox.setMinimum(2)
        self.outer_spinbox.setValue(10)
        self.layout.addWidget(self.outer_spinbox, row, 2)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Max Filter Iterations'), row, 1)
        self.iter_spinbox = QtGui.QSpinBox()
        self.iter_spinbox.setMinimum(3)
        self.iter_spinbox.setValue(3)
        self.layout.addWidget(self.iter_spinbox, row, 2)
        row += 1
        self.update_button = QtGui.QPushButton("Update Peakfinder")
        self.update_button.clicked.connect(self.send_values)
        self.layout.addWidget(self.update_button, row, 1, 1, 2)

        self.setLayout(self.layout)

    def send_values(self):
        self.debug('PeakfinderConfigWidget.send_values()')
        self.values_changed.emit()

    def get_values(self):
        self.debug('PeakfinderConfigWidget.get_values()')
        dat = {}
        dat['activate'] = self.activate_peakfinder_button.isChecked()
        dat['show_snr'] = self.activate_snrview_button.isChecked()
        dat['inner'] = self.inner_spinbox.value()
        dat['center'] = self.center_spinbox.value()
        dat['outer'] = self.outer_spinbox.value()
        dat['snr_threshold'] = self.snr_spinbox.value()
        dat['max_iterations'] = self.iter_spinbox.value()
        return dat

    def debug(self, msg):

        print(msg)


class SNRConfigWidget(QtGui.QWidget):

    values_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle('SNR Filter Settings')
        self.layout = QtGui.QGridLayout()

        self.layout.addWidget(QtGui.QLabel('Activate Filter'), 1, 1)
        self.activate_button = QtGui.QRadioButton()
        self.activate_button.toggled.connect(self.send_values)
        self.layout.addWidget(self.activate_button, 1, 2, alignment=QtCore.Qt.AlignCenter)

        self.layout.addWidget(QtGui.QLabel('Inner Size'), 2, 1)
        self.inner_spinbox = QtGui.QSpinBox()
        self.inner_spinbox.setMinimum(1)
        self.inner_spinbox.setValue(1)
        self.layout.addWidget(self.inner_spinbox, 2, 2)

        self.layout.addWidget(QtGui.QLabel('Center Size'), 3, 1)
        self.center_spinbox = QtGui.QSpinBox()
        self.center_spinbox.setMinimum(1)
        self.center_spinbox.setValue(5)
        self.layout.addWidget(self.center_spinbox, 3, 2)

        self.layout.addWidget(QtGui.QLabel('Outer Size'), 4, 1)
        self.outer_spinbox = QtGui.QSpinBox()
        self.outer_spinbox.setMinimum(1)
        self.outer_spinbox.setValue(10)
        self.layout.addWidget(self.outer_spinbox, 4, 2)

        self.update_button = QtGui.QPushButton("Update Filter")
        self.update_button.clicked.connect(self.send_values)
        self.layout.addWidget(self.update_button, 5, 1, 1, 2)

        self.setLayout(self.layout)

    def send_values(self):
        self.debug('SNRConfigWidget.send_values()')
        self.values_changed.emit()

    def get_values(self):
        self.debug('SNRConfigWidget.get_values()')
        dat = {}
        dat['activate'] = self.activate_button.isChecked()
        dat['inner'] = self.inner_spinbox.value()
        dat['center'] = self.center_spinbox.value()
        dat['outer'] = self.outer_spinbox.value()
        return dat

    def debug(self, msg):

        print(msg)


class LevelsWidget(QtGui.QWidget):
    min_value = None
    max_value = None
    def __init__(self, padview=None):
        super().__init__()
        self.setWindowTitle("Levels")
        layout = QtGui.QVBoxLayout()
        min_value = layout.addWidget(QtGui.QDoubleSpinBox())
        max_value = layout.addWidget(QtGui.QDoubleSpinBox())
        self.setLayout(layout)
    def get_min_value(self):
        return self.min_value.value()
    def get_max_value(self):
        return self.max_value.value()


if __name__ == '__main__':
    from reborn.simulate import solutions
    from reborn import detector, source
    from reborn.viewers.qtviews.padviews import PADView2
    np.random.seed(10)
    beam_vec = [0, 0, 1] #reborn.utils.vec_norm(np.array([0, 0.1, 0.9]))
    theta1 = 0.0
    theta2 = 0.0
    Tscl = 0.0
    dist = 0.1
    pix = 1e-3
    shape = (200, 200)
    tiles = (2, 2)
    gap = pix*10
    #pads = [reborn.detector.PADGeometry(pixel_size=1e-3, distance=1, shape=(100, 100))]
    #pads[0].t_vec[0:2] = 0
    pads = detector.tiled_pad_geometry_list(pad_shape=shape, pixel_size=pix, distance=dist, tiling_shape=tiles, pad_gap=gap)
    ct = np.cos(theta1)
    st = np.sin(theta1)
    R2 = np.array([[1, 0, 0], [0, ct, st], [0, -st, ct]])
    ct = np.cos(theta2)
    st = np.sin(theta2)
    R1 = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])
    R = np.dot(R2, R1.T)
    T = np.array([0.05, -0.07, 0])*Tscl
    for p in pads:
        p.ss_vec = np.dot(p.ss_vec, R.T)
        p.fs_vec = np.dot(p.fs_vec, R.T)
        p.t_vec = np.dot(p.t_vec, R.T)
        p.t_vec += T
    beam = source.Beam(photon_energy=8000 * 1.602e-19, diameter_fwhm=1e-6, pulse_energy=0.1e-3, beam_vec=beam_vec)
    def make_images():
        dat = solutions.get_pad_solution_intensity(pads, beam, thickness=10e-6, liquid='water')
        for d in dat:
            nx, ny = d.shape
            x, y = np.indices(d.shape)
            xo = np.random.rand() * nx
            yo = np.random.rand() * ny
            d += 100 * np.exp((-(x - xo) ** 2 - (y - yo) ** 2)/3)
            # d.flat[0:10] = 0
        return dat
    dat = make_images()
    # for p in pads:
    #     p.t_vec[0] += 0.2
    mask = [np.ones(p.shape()) for p in pads]
    # for i in range(len(mask)):
    #     mask[i][dat[i] < 40000] = 0
    [print(p) for p in pads]
    pv = PADView2(raw_data=dat, pad_geometry=pads, mask_data=mask, debug_level=2)
    pv.show_coordinate_axes()
    # pv.run_plugin('snr_filter')
    # pv.add_circle_roi(pos=(0.1, 0.1), radius=0.01)
    # pv.show_fast_scan_directions()
    pv.set_title('testing')
    # pv.toggle_grid()
    pv.start()
    # lw = LevelsWidget(pv)
    # lw.show()
    # app.exec_()
