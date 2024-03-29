# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-
import sys
import os
import time
import glob
import importlib
import pickle
import json
import numpy as np
import pkg_resources
import functools
import logging
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.Qt.QtWidgets as qwgt
from pyqtgraph import ImageItem
from ... import source, detector, utils, fileio
from ...dataframe import DataFrame
from ...fileio.getters import FrameGetter
from ...external.pyqtgraph import MultiHistogramLUTWidget #, ImageItem


logger = logging.getLogger(__name__)

# Add some pre-defined colormaps (called "gradients" in pyqtgraph)
g = pg.graphicsItems.GradientEditorItem.Gradients
g['bipolar2'] = {'ticks': [(0.0, (255, 0, 0, 255)), (0.5, (0, 0, 0, 255)), (1.0, (0, 255, 255, 255))], 'mode': 'rgb'}
pg.graphicsItems.GradientEditorItem.Gradients = g

plugin_path = pkg_resources.resource_filename('reborn.viewers.qtviews', 'plugins')


class PADViewMainWindow(pg.Qt.QtWidgets.QMainWindow):
    r""" A QMainWindow that closes all windows when it is closed.  Be careful... """
    def __init__(self, main=True):
        super().__init__()
        self.main = main
    def closeEvent(self, *args, **kwargs):
        if self.main:
            pg.mkQApp().closeAllWindows()


def ensure_dataframe(data, parent=None):
    r""" Convert old-style dictionaries to proper DataFrame instances. """
    if parent is None:
        parent = DataFrame()
    if not isinstance(parent, DataFrame):
        raise ValueError('parent must be a DataFrame')
    if isinstance(data, DataFrame):
        return data
    if isinstance(data, dict):
        dataframe = parent
        dataframe.set_raw_data(data['pad_data'])
        return dataframe


class PADView(QtCore.QObject):

    r"""
    This is supposed to be an easy way to view PAD data.  It is a work in progress... but here is the basic concept:

    |PADView| relies on a |FrameGetter| subclass to serve up |DataFrame| instances.
    """

    # Note that most of the interface was created using the QT Designer tool.  There are many attributes that are
    # not visible here.
    logger = None
    frame_getter = None
    _dataframe = None
    debug_level = 0  # Levels are 0: no messages, 1: basic messages, 2: more verbose, 3: extremely verbose
    pad_labels = None
    mask_image_items = None
    mask_color = [128, 0, 0, 255]
    _mask_rgba_arrays = None
    _mask_rois = None
    skip_empty_frames = True
    pad_image_items = None
    scatter_plots = None
    plot_items = None
    rings = []
    ring_pen = pg.mkPen([255, 255, 255], width=2)
    grid = None
    coord_axes = None
    scan_arrows = None
    _px_mode = False
    _shortcuts = []
    _status_string_mouse = ""
    _status_string_getter = " Frame 1 of 1 | "
    evt = None
    show_true_fast_scans = False
    apply_filters = True
    plugins = None
    _fixed_title = False
    _dataframe_preprocessor = None
    default_levels = [None, None]
    default_percentiles = [2, 98]
    mirror_levels = False
    levels_ignore_masked = True
    previous_levels = [0, 1]  # Levels just before PADs are updated
    _is_updating_display_data = False
    _tic_times = []

    status_bar_style = "background-color:rgb(30, 30, 30);color:rgb(255,0,255);font-weight:bold;font-family:monospace;"
    scatterplot_style = {'pen': pg.mkPen('g'), 'brush': None, 'width': 5, 'size': 10, 'pxMode': True}

    sig_geometry_changed = QtCore.pyqtSignal()
    sig_beam_changed = QtCore.pyqtSignal()
    sig_dataframe_changed = QtCore.pyqtSignal()

    def __init__(self, frame_getter=None, data=None, pad_geometry=None, mask=None,
                 beam=None, levels=None, percentiles=None, debug_level=0, main=True, dataframe_preprocessor=None,
                 hold_levels=False, **kwargs):
        """
        Arguments:
            frame_getter (|FrameGetter| subclass): The preferred way to flip through many frames.
            data (|DataFrame| or |ndarray|): The data to be displayed.
            pad_geometry (|PADGeometryList|): PAD geometry information.
            mask (|ndarray|): Data masks.
            beam (|Beam|): X-ray beam parameters.
            levels (tuple): The minimum and maximum levels for the colormap.
            percentiles (tuple): The minimum and maximum percentages for the colormap.
            debug_level (int): The debug level. Larger values are more verbose.
            main (bool): Set to False if you don't want this window to close all other windows when it is closed.
            hold_levels (bool): If True, do not attempt to re-scale the colormap for each frame.
            dataframe_preprocessor (function): Experimental.
        """
        self.debug_level = debug_level
        self.setup_logger()
        self.debug('Initializing PADView instance.')
        super().__init__()
        self.main = main
        self.hold_levels = hold_levels
        self._dataframe_preprocessor = dataframe_preprocessor
        self.auto_percentiles = percentiles
        self.fixed_levels = levels
        if frame_getter is None:
            frame_getter = DummyFrameGetter(data=data, pad_geometry=pad_geometry, mask=mask, beam=beam, **kwargs)
        self.frame_getter = frame_getter
        self.app = pg.mkQApp()
        self.setup_ui()
        self.setup_mouse_interactions()
        self.setup_shortcuts()
        self.setup_menubar()
        self.statusbar.setStyleSheet(self.status_bar_style)
        self.show_first_frame()
        self.main_window.show()
        self.sig_beam_changed.connect(self.update_rings)
        # self.sig_geometry_changed.connect(self.update_rings)
        self.debug('Initialization complete.')

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        self.logger = logger
        if len(logger.handlers) > 0:
            return
        logger.propagate = False
        level = logging.DEBUG
        logger.setLevel(level)
        formatter = " - ".join(["%(asctime)s", "%(levelname)s", "%(name)s", "%(message)s"])
        formatter = logging.Formatter(formatter)
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level=level)
        logger.addHandler(console_handler)

    def debug(self, msg="", level=1):
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
        if level <= self.debug_level:
            self.logger.debug(msg)

    @property
    def dataframe(self):
        r""" The current |DataFrame| instance. """
        return self._dataframe

    @dataframe.setter
    def dataframe(self, val):
        self.debug("dataframe setter")
        if self._dataframe is None:
            self._dataframe = val
        val = ensure_dataframe(val, self._dataframe)
        if self._dataframe_preprocessor is not None:
            val = self._dataframe_preprocessor(val)
        if isinstance(val, DataFrame):
            self._dataframe = val
            self.sig_dataframe_changed.emit()
            return
        # if self.dataframe is not None:
        #     d = self.dataframe.get_raw_data_flat()
        #     self.dataframe.set_raw_data(d*0)
        #     self.dataframe.clear_processed_data()
        self.debug(f'Attempted to set dataframe to wrong type!!!! {val}')

    def setup_ui(self):
        r""" Creates the main interface: QMainWindow, menubar, statusbar, viewbox, etc."""
        self.debug("setup_ui")
        self.main_window = PADViewMainWindow(main=self.main)
        self.menubar = self.main_window.menuBar()
        self.statusbar = self.main_window.statusBar()
        self.hbox = qwgt.QHBoxLayout()
        self.splitter = qwgt.QSplitter(QtCore.Qt.Horizontal)
        # self.side_panel = qwgt.QWidget()
        # self.side_panel_layout = qwgt.QVBoxLayout()
        # self.side_panel_layout.setAlignment(QtCore.Qt.AlignTop)
        # box = misc.CollapsibleBox('Display') ###########################
        # lay = qwgt.QGridLayout()
        # lay.addWidget(qwgt.QLabel('CMap min:'), 1, 1)
        # maxspin = qwgt.QSpinBox()
        # lay.addWidget(maxspin, 1, 2)
        # box.setContentLayout(lay)
        # self.side_panel_layout.addWidget(box)
        # box = misc.CollapsibleBox('Analysis') ###########################
        # lay = qwgt.QGridLayout()
        # row = 0
        # row += 1
        # lay.addWidget(qwgt.QLabel('Polarization Correction'), row, 1)
        # polarization_button = qwgt.QCheckBox()
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
        self.main_widget = qwgt.QWidget()
        self.main_widget.setLayout(self.hbox)
        self.main_window.setCentralWidget(self.main_widget)

    def set_title(self, title=None):
        r""" Set the title of the main window. """
        self.debug("set_title")
        if title is not None:
            self.main_window.setWindowTitle(title)
            self._fixed_title = True
        if not self._fixed_title:
            title = ''
            try:
                t = self.dataframe.get_dataset_id().__str__()
                if t:
                    title += t+' '
                f = self.dataframe.get_frame_id().__str__()
                if f:
                    title += f
                self.main_window.setWindowTitle(title)
            except:
                pass

    def setup_mouse_interactions(self):
        r""" I don't know what this does... obviously something about mouse interactions... """
        # FIXME: What does this do?
        self.debug("setup_mouse_interactions")
        self.proxy = pg.SignalProxy(self.viewbox.scene().sigMouseMoved, rateLimit=30, slot=self.mouse_moved)

    def setup_menubar(self):
        r""" Connect menu items (e.g. "File") so that they actually do something when clicked. """
        self.debug("setup_menubar")
        def add_menu(append_to, name, short=None, tip=None, connect=None):
            action = qwgt.QAction(name, self.main_window)
            if short: action.setShortcut(short)
            if tip: action.setStatusTip(tip)
            if connect: action.triggered.connect(connect)
            append_to.addAction(action)
            return action
        file_menu = self.menubar.addMenu('File')
        add_menu(file_menu, 'Open file...', connect=self.open_data_file_dialog)
        add_menu(file_menu, 'Save file...', connect=self.save_data_file_dialog)
        add_menu(file_menu, 'Save screenshot...', connect=self.save_screenshot_dialog)
        add_menu(file_menu, 'Exit', short='Ctrl+Q', connect=self.app.quit)
        data_menu = self.menubar.addMenu('Data')
        add_menu(data_menu, 'Frame navigator...', connect=lambda: self.run_plugin('frame_navigator'))
        add_menu(data_menu, 'Clear processed data', connect=self.clear_processed_data)
        geom_menu = self.menubar.addMenu('Geometry')
        add_menu(geom_menu, 'Shift detector...', connect=lambda: self.run_plugin('shift_detector'))
        add_menu(geom_menu, 'Show coordinates', connect=self.toggle_coordinate_axes)
        add_menu(geom_menu, 'Show grid', connect=self.toggle_grid)
        add_menu(geom_menu, 'Show PAD labels', connect=self.toggle_pad_labels)
        add_menu(geom_menu, 'Show scan directions', connect=self.toggle_fast_scan_directions)
        add_menu(geom_menu, 'Edit ring radii...', connect=self.edit_ring_radii)
        add_menu(geom_menu, 'Save PAD geometry...', connect=self.save_pad_geometry)
        add_menu(geom_menu, 'Load PAD geometry...', connect=self.load_pad_geometry)
        mask_menu = self.menubar.addMenu('Mask')
        add_menu(mask_menu, 'Mask editor...', connect=lambda: self.run_plugin('mask_editor'))
        add_menu(mask_menu, 'Clear masks', connect=self.clear_masks)
        add_menu(mask_menu, 'Toggle masks visible', connect=self.toggle_masks)
        # add_menu(mask_menu, 'Choose mask color', connect=self.choose_mask_color)
        # add_menu(mask_menu, 'Mask PADs by name', connect=self.mask_pads_by_names)
        # add_menu(mask_menu, 'Mask panel edges...', connect=self.mask_panel_edges)
        # add_menu(mask_menu, 'Mask above upper limit', connect=self.mask_upper_level)
        # add_menu(mask_menu, 'Mask below lower limit', connect=self.mask_lower_level)
        # add_menu(mask_menu, 'Mask outside limits', connect=self.mask_levels)
        # add_menu(mask_menu, 'Add rectangle ROI', connect=self.add_rectangle_roi)
        # add_menu(mask_menu, 'Add circle ROI', connect=self.add_circle_roi)
        # add_menu(mask_menu, 'Toggle ROIs visible', connect=self.toggle_rois)
        add_menu(mask_menu, 'Save mask...', connect=self.save_masks)
        add_menu(mask_menu, 'Load mask...', connect=self.load_masks)
        beam_menu = self.menubar.addMenu('Beam')
        add_menu(beam_menu, 'Save beam...', connect=self.save_beam)
        add_menu(beam_menu, 'Load beam...', connect=self.load_beam)
        analysis_menu = self.menubar.addMenu('Analysis')
        add_menu(analysis_menu, 'Scattering profile...', connect=lambda: self.run_plugin('scattering_profile'))
        plugin_menu = self.menubar.addMenu('Plugins')
        self.plugin_names = []
        for plg in sorted(glob.glob(plugin_path+'/**/*.py', recursive=True)):
            plugin_name = plg.replace(plugin_path, '')[1:]  #os.path.basename(plg).replace('.py', '')
            if '__init__' in plugin_name:
                continue
            self.plugin_names.append(plugin_name)
            self.debug('\tSetup plugin ' + plugin_name)
            add_menu(plugin_menu, plugin_name, #.replace('_', ' '),
                     connect=functools.partial(self.run_plugin, plugin_name))

    def set_shortcut(self, shortcut, func):
        r""" Setup one keyboard shortcut so it connects to some function, assuming no arguments are needed. """
        self._shortcuts.append(qwgt.QShortcut(QtGui.QKeySequence(shortcut), self.main_window).activated.connect(
            func))

    def setup_shortcuts(self):
        r""" Connect all standard keyboard shortcuts to functions. """
        # FIXME: Many of these shortcuts were randomly assigned in a hurry.  Need to re-think the keystrokes.
        self.debug("setup_shortcuts")
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
        self.set_shortcut("m", self.toggle_masks)

    def update_status_string(self, frame_number=None, n_frames=None):
        r""" Update status string at the bottom of the main window. """
        self.debug("update_status_string", level=3)
        strn = ''
        if frame_number is None:
            frame_number = self.frame_getter.current_frame
        if n_frames is None:
            n_frames = self.frame_getter.n_frames
        if frame_number is not None:
            strn += ' Frame %d' % (frame_number + 1)
            if n_frames is not None:
                if n_frames == np.inf:
                    strn += ' of inf'
                else:
                    strn += ' of %d' % n_frames
            strn += ' | '
            self._status_string_getter = strn
        self.statusbar.showMessage(self._status_string_getter + self._status_string_mouse)

    # FIXME: For some reason, the histogram is only updated on the first frame.  Need to track down this issue.
    # FIXME: Also, we need to allow for the histogram to be disabled since it takes time to compute.
    def setup_histogram_tool(self):
        r""" Set up the histogram/colorbar/colormap tool that is located to the right of the PAD display. """
        self.debug("setup_histogram_tool")
        self.set_colormap('flame')
        self.histogram.setImageItems(self.pad_image_items)

    def set_colormap(self, preset='flame'):
        r""" Change the colormap to one of the presets configured in pyqtgraph.  Right-click on the colorbar to find
        out what values are allowed.
        """
        self.debug("set_colormap")
        self.histogram.gradient.loadPreset(preset)
        self.histogram.setImageItems(self.pad_image_items)
        pg.mkQApp().processEvents()

    def get_levels(self):
        r""" Get the minimum and maximum levels of the current image display. """
        return self.histogram.item.getLevels()

    def set_levels(self, levels=(None, None), percentiles=(None, None), colormap=None):
        r"""
        This method handles setting levels (i.e. mapping the upper and lower bounds of detector pixel values to the
        span of the color table).

        If no arguments are provided, the default settings are used.

        Level setting is a somewhat complicated matter.  For both the upper and lower bounds, there are 5 ways that
        we presently set the levels of the display data.  In order of priority, highest first:

          1) Set the level to the particular value specified in the levels keyword argument of this method.
          2) Set the level to the value corresponding to the Nth percentile of the current display data,
             using the percentiles keyword argument.
          3) Set the level to the particular value specified in the "default_levels" attribute of this class instance.
          4) Set the level to the percentile specified in the "default_percentiles" attribute of this class instance.
          5) Do nothing; leave the level unchanged.

        An additional option is to mirror the levels: i.e. set the levels to range from [-x, +x].  This is controlled
        via the "mirror_levels" attribute.

        When determining levels from percentiles, it is possible to ignore masked pixels.  This is controlled by the
        attribute "levels_ignore_masked".

        Arguments:
            levels (tuple of floats): Fixed-value levels.  Default is (None, None).
            percentiles (tuple of floats): Percentiles for determining levels.  Values should be 0-100.  Default is
                                           (None, None)
            colormap:  For convenience, you may also specify a new colormap.  This is passed to "set_colormap".
        """
        self.debug("set_levels")
        data = self.get_pad_display_data(as_list=False)
        if data is None:
            return
        if self.levels_ignore_masked:
            mask = self.dataframe.get_mask_flat()
            data = data[np.where(mask)]
        new_levels = [None, None]
        for i in range(2):
            # Top priority: requested fixed level
            v = levels[i]
            self.debug(f"fixed level = {v}")
            # Second priority: requested percentile
            if v is None and percentiles[i] is not None:
                v = np.percentile(data, percentiles[i])
                self.debug(f"percentile = {percentiles[i]}")
            # Third priority: default fixed level
            if v is None:
                v = self.default_levels[i]
                self.debug(f"default fixed level = {v}")
            # Fourth priority: default percentile
            if v is None and self.default_percentiles[i] is not None:
                    v = np.percentile(data, self.default_percentiles[i])
                    self.debug(f"default percentile level = {self.default_percentiles[i]}")
            # Fifth priority: do not change the level.  Keep the current levels.
            if v is None:
                v = self.previous_levels[i]
                self.debug(f"previous level = {v}")
            new_levels[i] = v
        # Special override: Mirror levels if requested.  I.e. set the range to span [-abs(max), +abs(max)]
        if self.mirror_levels:
            self.debug(f"mirror levels")
            a = np.max(np.abs(np.array(new_levels)))
            new_levels[0] = -a
            new_levels[1] = a
        self.histogram.item.setLevels(new_levels[0], new_levels[1])
        if colormap is not None:
            self.set_colormap(colormap)

    def add_rectangle_roi(self, pos=(0, 0), size=None):
        r""" Adds a |pyqtgraph| RectROI """
        self.debug("add_rectangle_roi")
        if type(pos) != tuple:
            pos = (0, 0)
        if size is None:
            br = self.get_view_bounding_rect()
            s = min(br[2], br[3])
            size = (s/4, s/4)
        self.debug((size, pos).__str__())
        pos = (pos[0] - size[0]/2, pos[1] - size[0]/2)
        roi = pg.RectROI(pos=pos, size=size, centered=True, sideScalers=True)
        roi.name = 'rectangle'
        roi.addRotateHandle(pos=(0, 1), center=(0.5, 0.5))
        if self._mask_rois is None:
            self._mask_rois = []
        # FIXME: We need better handling of these ROIs: better manipulations of sizes/shapes, more features (e.g. the
        # FIXME: ability to show a histogram of pixels within the ROI)
        self._mask_rois.append(roi)
        self.viewbox.addItem(roi)

    # FIXME: Circle ROI should have an option to fix the center on the beam center.
    def add_circle_roi(self, pos=(0, 0), radius=None):
        self.debug("add_circle_roi")
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

    def show_rois(self):
        self.debug("show_rois")
        if self._mask_rois is not None:
            for roi in self._mask_rois:
                roi.setVisible(True)

    def hide_rois(self):
        self.debug("hide_rois")
        if self._mask_rois is not None:
            for roi in self._mask_rois:
                roi.setVisible(False)

    def toggle_rois(self):
        self.debug("toggle_rois")
        if self._mask_rois is None:
            return
        for roi in self._mask_rois:
            if roi.isVisible():
                self.hide_rois()
                return
        self.show_rois()

    def show_coordinate_axes(self):
        self.debug("show_coordinate_axes")
        if self.coord_axes is None:
            geom = self.dataframe.get_pad_geometry()
            corners = self.vector_to_view_coords(np.vstack([p.corner_position_vectors() for p in geom]))
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
        self.debug("hide_coordinate_axes")
        if self.coord_axes is not None:
            for c in self.coord_axes:
                self.viewbox.removeItem(c)
            self.coord_axes = None

    def toggle_coordinate_axes(self):
        self.debug("toggle_coordinate_axes")
        if self.coord_axes is None:
            self.show_coordinate_axes()
        else:
            self.hide_coordinate_axes()

    def show_fast_scan_directions(self):
        self.debug("show_fast_scan_directions")
        if self.scan_arrows is None:
            self.scan_arrows = []
            for p in self.dataframe.get_pad_geometry():
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
        self.debug("hide_fast_scan_directions")
        if self.scan_arrows is not None:
            for a in self.scan_arrows:
                self.viewbox.removeItem(a)
            self.scan_arrows = None

    def toggle_fast_scan_directions(self):
        self.debug("toggle_fast_scan_directions")
        if self.scan_arrows is None:
            self.show_fast_scan_directions()
        else:
            self.hide_fast_scan_directions()

    def show_all_geom_info(self):
        self.debug("show_all_geom_info")
        self.show_pad_borders()
        self.show_grid()
        self.show_pad_labels()
        self.show_fast_scan_directions()
        self.show_coordinate_axes()

    def hide_all_geom_info(self):
        self.debug("hide_all_geom_info")
        self.hide_pad_borders()
        self.hide_grid()
        self.hide_pad_labels()
        self.hide_fast_scan_directions()
        self.hide_coordinate_axes()

    def toggle_all_geom_info(self):
        self.debug("toggle_all_geom_info")
        if self.scan_arrows is None:
            self.show_all_geom_info()
        else:
            self.hide_all_geom_info()

    def show_pad_labels(self):
        self.debug("show_pad_labels")
        if self.pad_labels is None:
            self.pad_labels = []
            pad_geometry = self.dataframe.get_pad_geometry()
            for i in range(0, self.dataframe.n_pads):
                g = pad_geometry[i]
                if (not hasattr(g, 'name')) or (g.name is None) or (g.name == ''):
                    g.name = "%d" % i
                lab = pg.TextItem(text=g.name, fill=pg.mkBrush(20, 20, 20, 128), color='w', anchor=(0.5, 0.5),
                                  border=pg.mkPen('w'))
                vec = pad_geometry[i].center_pos_vec()
                vec = self.vector_to_view_coords(vec)
                lab.setPos(vec[0], vec[1])
                self.pad_labels.append(lab)
                self.viewbox.addItem(lab)

    def hide_pad_labels(self):
        self.debug("hide_pad_labels")
        if self.pad_labels is not None:
            for lab in self.pad_labels:
                self.viewbox.removeItem(lab)
            self.pad_labels = None

    def toggle_pad_labels(self):
        self.debug("toggle_pad_labels")
        if self.pad_labels is None:
            self.show_pad_labels()
        else:
            self.hide_pad_labels()

    def _apply_pad_transform(self, im, p):
        self.debug("_apply_pad_transform", level=2)
        f = p.fs_vec.copy()
        s = p.ss_vec.copy()
        t = p.t_vec.copy() - (f + s)/2.0
        trans = QtGui.QTransform()
        trans.setMatrix(s[0], s[1], s[2], f[0], f[1], f[2], t[0], t[1], t[2])
        im.setTransform(trans)

    def choose_mask_color(self):
        self.debug("choose_mask_color")
        color = qwgt.QColorDialog.getColor()
        if color is None:
            self.debug('Color is None')
            return
        if not color.isValid():
            self.debug('Color is invalid')
            return
        self.mask_color[0] = color.red()
        self.mask_color[1] = color.green()
        self.mask_color[2] = color.blue()
        self._mask_rgba_arrays = None
        self.update_masks()

    def set_mask_color(self, color):
        self.mask_color = color
        self._mask_rgba_arrays = None
        self.update_masks()

    def update_masks(self, masks=None):
        r""" Update the data shown in mask image items. """
        self.debug('update_masks')
        self.tic('update_masks')
        self.tic('Getting mask list from dataframe...')
        if masks is None:
            masks = self.dataframe.get_mask_list()
        else:
            geom = self.dataframe.get_pad_geometry()
            masks = geom.split_data(masks)
        self.toc()
        if len(self.mask_color) < 4:
            self.mask_color.append(0)
        if self._mask_rgba_arrays is None:
            self.tic('Initializing RGBA mask arrays...')
            self._mask_rgba_arrays = []
            for (i, mask) in enumerate(masks):
                m = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                m[:, :, 0] = int(self.mask_color[0])
                m[:, :, 1] = int(self.mask_color[1])
                m[:, :, 2] = int(self.mask_color[2])
                m[:, :, 3] = int(self.mask_color[3])
                self._mask_rgba_arrays.append(m)
            self.toc()
        for (item, mask, rgba) in zip(self.mask_image_items, masks, self._mask_rgba_arrays):
            self.tic('Updating RGBA mask array...')
            m = rgba[:, :, 3]
            m[:, :] = 0
            m[np.where(mask == 0)] = self.mask_color[3]
            self.toc()
            self.tic('Setting mask image...')
            item.setImage(rgba)
            self.toc()
        self.toc()

    def hide_masks(self):
        self.debug("hide_masks")
        if self.mask_image_items is not None:
            for im in self.mask_image_items:
                im.setVisible(False)

    def show_masks(self):
        self.debug("show_masks")
        if self.mask_image_items is not None:
            for im in self.mask_image_items:
                im.setVisible(True)

    def toggle_masks(self):
        self.debug("toggle_masks")
        if self.mask_image_items is not None:
            for im in self.mask_image_items:
                im.setVisible(not im.isVisible())

    def save_masks(self):
        r""" Save list of masks in pickle or reborn mask format. """
        self.debug("save_masks")
        options = qwgt.QFileDialog.Options()
        file_name, file_type = qwgt.QFileDialog.getSaveFileName(self.main_window, "Save Masks", "mask",
                                                          "reborn Mask File (*.mask);;Python Pickle (*.pkl)",
                                                                 options=options)
        if file_name == "":
            return
        if file_type == 'Python Pickle (*.pkl)':
            print('Saving masks: ' + file_name)
            with open(file_name, "wb") as f:
                pickle.dump(self.dataframe.get_mask_list(), f)
        if file_type == 'reborn Mask File (*.mask)':
            if file_name.split('.')[-1] != 'mask':
                file_name += '.mask'
            print('Saving masks: ' + file_name)
            detector.save_pad_masks(file_name, self.dataframe.get_mask_list())

    def load_masks(self):
        r""" Load list of masks that have been saved in pickle or reborn mask format. """
        self.debug("load_masks")
        options = qwgt.QFileDialog.Options()
        file_name, file_type = qwgt.QFileDialog.getOpenFileName(self.main_window, "Load Masks", "mask",
                                                          "reborn Mask File (*.mask);;Python Pickle (*.pkl)",
                                                                 options=options)
        if file_name == "":
            return
        if file_type == 'Python Pickle (*.pkl)':
            with open(file_name, "rb") as f:
                mask = pickle.load(f)
        if file_type == 'reborn Mask File (*.mask)':
            mask = detector.load_pad_masks(file_name)
        self.dataframe.set_mask(mask)
        self.update_masks()
        print('Loaded mask: ' + file_name)

    def get_hovering_roi_indices(self, flat=True):
        r"""Get the indices within the ROI that the mouse is presently hovering over.  flat=True indicates that you wish
        to have the flattened indices, where are pads are concatenated.  If no ROI is selected, return None, None.
        Otherwise, return indices, roi.name (str).
        """
        if flat is False:
            raise ValueError('flat=False is not implemented in get_hovering_roi_indices (yet)')
        if self._mask_rois is None:
            return None, None
        roi = [r for r in self._mask_rois if r.mouseHovering]
        if len(roi) == 0:
            return None, None
        roi = roi[0]
        geom = self.dataframe.get_pad_geometry()
        p_vecs = geom.position_vecs()
        v_vecs = self.vector_to_view_coords(p_vecs)[:, 0:2]
        if roi.name == 'rectangle':  # Find all pixels within the rectangle
            self.debug('\tGetting rectangle ROI indices')
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
            self.debug('\tGetting circle ROI indices')
            radius = roi.size()[0]/2.
            center = np.array([roi.pos()[0], roi.pos()[1]]) + radius
            inds = np.sqrt(np.sum((v_vecs - center)**2, axis=1)) < radius
        return inds, roi.name

    def clear_masks(self):
        self.debug("clear_masks")
        self.dataframe.set_mask(None)
        self.update_masks()

    def get_pad_display_data(self, as_list=True):
        self.debug("get_pad_display_data", level=3)
        if self.dataframe is not None:
            if as_list:
                return self.dataframe.get_processed_data_list()
            return self.dataframe.get_processed_data_flat()

    def set_pad_display_data(self, data, update_display=True, levels=(None, None), percentiles=(None, None),
                             colormap=None):
        self.debug("set_pad_display_data")
        self.dataframe.set_processed_data(data)
        if update_display:
            self.update_pads()
        if (levels is not None) or (percentiles is not None):
            self.set_levels(levels=levels, percentiles=percentiles, colormap=colormap)

    def clear_processed_data(self):
        r""" Clear processed data and (show raw data). """
        self.dataframe._processed_data = None
        self.update_display()

    def setup_image_items(self):
        r""" Creates the PAD and mask ImageItems. Applies geometry transforms.  Sets data and colormap levels. """
        self.debug("setup_image_items")
        if self.pad_image_items:
            for item in self.pad_image_items:
                self.viewbox.removeItem(item)
        if self.mask_image_items:
            for item in self.mask_image_items:
                self.viewbox.removeItem(item)
        self.pad_image_items = []
        self.mask_image_items = []
        for i in range(self.dataframe.n_pads):
            im = ImageItem() #, autoDownsample='mean')
            self.pad_image_items.append(im)
            self.viewbox.addItem(im)
            im = ImageItem() #, autoDownsample='mean')
            self.mask_image_items.append(im)
            self.viewbox.addItem(im)
        self.update_pad_geometry()
        self.update_pads()
        # self.update_masks()
        self.setup_histogram_tool()
        self.set_levels(percentiles=(2, 98))

    def update_pads(self):
        r""" Update only the PAD data that is displayed, including colormap.  Usually you should run update_display
        instead of this. """
        self.debug("update_pads")
        self.tic('Updating PAD displays.')
        levels = self.get_levels()
        data = self.get_pad_display_data()
        if data is None:
            self.debug('get_pad_display_data returned None!')
            return
        if self.pad_image_items is None:
            self.setup_image_items()
        for (i, d) in enumerate(data):
            self.pad_image_items[i].setImage(np.nan_to_num(d))
        self.set_levels()
        self.tic('Updating histogram')
        self.histogram.regionChanged()
        self.toc()
        self.toc()

    def update_pad_geometry(self, pad_geometry=None):
        r""" Update the PAD geometry.  Apply Qt transforms.  Also emits the sig_geometry_changed signal."""
        self.debug("update_pad_geometry")
        if pad_geometry is None:
            pad_geometry = self.dataframe.get_pad_geometry()
        else:
            self.dataframe.set_pad_geometry(pad_geometry)
        for i in range(self.dataframe.n_pads):
            if self.pad_image_items is not None:
                self._apply_pad_transform(self.pad_image_items[i], pad_geometry[i])
            if self.mask_image_items is not None:
                self._apply_pad_transform(self.mask_image_items[i], pad_geometry[i])
        if self.pad_labels is not None:
            self.toggle_pad_labels()
            self.toggle_pad_labels()
        self.sig_geometry_changed.emit()

    def update_beam(self, beam=None):
        r""" Update the |Beam|.  Emits the sig_beam_changed signal. """
        self.debug("update_beam")
        if beam is None:
            return
        self.dataframe.set_beam(beam)
        self.sig_beam_changed.emit()

    def set_auto_level_percentiles(self, percents=(1, 99)):
        r""" Set to None if auto-scaling is not desired """
        self.debug("set_auto_level_percentiles")
        self.auto_percentiles = percents
        self.update_display()

    def save_pad_geometry(self):
        r""" Save list of pad geometry specifications in json format. """
        self.debug("save_pad_geometry")
        options = qwgt.QFileDialog.Options()
        file_name, file_type = qwgt.QFileDialog.getSaveFileName(self.main_window, "Save PAD Geometry", "geometry",
                                                          "reborn PAD Geometry File (*.json);;",
                                                                 options=options)
        if file_name == "":
            return
        self.debug(f'Saving PAD geometry to file: {file_name}')
        detector.save_pad_geometry_list(file_name, self.dataframe.get_pad_geometry())

    def load_pad_geometry(self):
        r""" Load list of pad geometry specifications in json format. """
        self.debug("load_pad_geometry")
        options = qwgt.QFileDialog.Options()
        file_name, file_type = qwgt.QFileDialog.getOpenFileName(self.main_window, "Load PAD Geometry", "geometry",
                                                          "reborn PAD Geometry File (*.json);;",
                                                                 options=options)
        if file_name == "":
            return
        self.debug(f'Loading PAD geometry from file: {file_name}')
        pads = detector.load_pad_geometry_list(file_name)
        self.update_pad_geometry(pads)

    def save_beam(self):
        r""" Save beam specifications in json format. """
        self.debug("save_beam")
        options = qwgt.QFileDialog.Options()
        file_name, file_type = qwgt.QFileDialog.getSaveFileName(self.main_window, "Save Beam", "beam",
                                                          "reborn Beam File (*.json);;",
                                                                 options=options)
        if file_name == "":
            return
        self.debug(f'Saving Beam to file: {file_name}')
        source.save_beam(self.dataframe.get_beam(), file_name)

    def load_beam(self):
        r""" Load beam specifications in json format. """
        self.debug("load_beam")
        options = qwgt.QFileDialog.Options()
        file_name, file_type = qwgt.QFileDialog.getOpenFileName(self.main_window, "Load Beam", "beam",
                                                          "reborn Beam File (*.json);;",
                                                                 options=options)
        if file_name == "":
            return
        self.debug('Loading Beam from file: {file_name}')
        beam = source.load_beam(file_name)
        self.update_beam(beam)

    @staticmethod
    def vector_to_view_coords(vec):
        r""" If you have a vector (or vectors) pointing in some direction in space, this function provides the
        point at which it intercepts with the view plane (the plane that is 1 meter away from the origin)."""
        vec = np.atleast_2d(vec)
        vec = (vec.T / np.squeeze(vec[:, 2])).T.copy()
        vec[:, 2] = 1
        return np.squeeze(vec)

    def get_pad_coords_from_view_coords(self, view_coords):
        r""" Get PAD coordinates (slow-scan index, fast-scan index, PAD index) from view coordinates.  The view
        coordinates correspond to the plane that is 1 meter away from the origin."""
        self.debug("get_pad_coords_from_view_coords", level=3)
        x = view_coords[0]
        y = view_coords[1]
        pad_idx = None
        geom = self.dataframe.get_pad_geometry()
        for n in range(self.dataframe.n_pads):
            vec = np.array([x, y, 1])  # This vector points from origin to the plane of the scene
            ss_idx, fs_idx = geom[n].vectors_to_indices(vec, insist_in_pad=True)
            if np.isfinite(ss_idx[0]) and np.isfinite(fs_idx[0]):
                pad_idx = n
                break
        return ss_idx, fs_idx, pad_idx

    def get_pad_coords_from_mouse_pos(self):
        r""" Maps mouse cursor position to view coordinates, then maps the view coords to PAD coordinates."""
        self.debug("get_pad_coords_from_mouse_pos", level=3)
        view_coords = self.get_view_coords_from_mouse_pos()
        ss_idx, fs_idx, pad_idx = self.get_pad_coords_from_view_coords(view_coords)
        self.debug('PAD coords: '+(ss_idx, fs_idx, pad_idx).__str__(), level=3)
        return ss_idx[0], fs_idx[0], pad_idx

    def get_view_coords_from_mouse_pos(self):
        r""" Map the current mouse cursor position to the display plane situated 1 meter from the interaction point. """
        self.debug("get_view_coords_from_mouse_pos", level=3)
        if self.evt is None:  # Note: self.evt is updated by _mouse_moved
            return 0, 0
        sc = self.viewbox.mapSceneToView(self.evt[0])
        self.debug('\tview coords: '+sc.__str__(), level=3)
        return sc.x(), sc.y()

    def get_view_bounding_rect(self):
        r""" Bounding rectangle of everything presently visible, in view (i.e real-space, 1-meter plane) coordinates."""
        vb = self.viewbox
        return vb.mapSceneToView(vb.mapToScene(vb.rect()).boundingRect()).boundingRect().getRect()

    def mouse_moved(self, evt):
        r""" Updates the status string with info about mouse position (e.g. data value under cursor)."""
        self.debug("mouse_moved", level=3)
        self.debug('mouse position: ' + evt.__str__(), level=3)
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
            self._status_string_mouse = '| PAD %2d  |  Pix %4d,%4d  |  Val=%8g  |' % (pid, ss, fs, intensity)
            q = self.dataframe.get_q_mags_list()[pid][ss, fs]/1e10
            self._status_string_mouse += ' q=%8g/A |'  % (q,)
            self._status_string_mouse += ' d=%8g A |'  % (2*np.pi/q,)
        self.update_status_string()

    def edit_ring_radii(self):
        self.debug("edit_ring_radii")
        text, ok = qwgt.QInputDialog.getText(self.main_window, "Enter ring radii (dict format)", "Ring radii",
                                              qwgt.QLineEdit.Normal,
                                              '{"q_mags":[], "d_spacings":[1e-10, 5e-10], "radii":[], "angles":[], '
                                              '"pens":[]}')
        if ok:
            d = json.loads(text)
            # # if text == '':
            # #     self.remove_rings()
            # #     return
            # # r = text.split(',')
            # # rad = []
            # # for i in range(0, len(r)):
            # #     try:
            # #         rad.append(float(r[i].strip()))
            # #     except:
            # #         pass
            # self.remove_rings()
            self.add_rings(**d)

    def add_rings(self, radii=None, angles=None, q_mags=None, d_spacings=None, pens=None, repeat=False):
        r""" Plot rings.  Note that these are in a plane located 1 meter from the sample position; calculate the radius
        needed for an equivalent detector at that distance.  If you know the scattering angle, the radius is
        tan(theta).  The repeat keyword will include all rings for a given d-spacing."""
        self.debug('add_rings')
        pens = utils.ensure_list(pens)
        if len(pens) < 1:
            pens = [None]
        # We allow various input types... so we must now ensure they are either list or None.
        input = []
        for d in [radii, angles, q_mags, d_spacings]:
            if isinstance(d, np.ndarray):
                d = [i for i in d]
            if not d:
                input.append(None)
                continue
            d = utils.ensure_list(d)
            input.append(d)
        radii, angles, q_mags, d_spacings = input
        if radii is not None:
            pens *= int(len(radii) / len(pens))
            # self.debug('add_rings:radii', radii, ', pens', pens)
            for (r, p) in zip(radii, pens):
                self.add_ring(radius=r, pen=p)
            return True
        if angles is not None:
            pens *= int(len(angles) / len(pens))
            # self.debug('add_rings:angles', angles, ', pens', pens)
            for (r, p) in zip(angles, pens):
                self.add_ring(angle=r, pen=p)
            return True
        if q_mags is not None:
            pens *= int(len(q_mags) / len(pens))
            # self.debug('add_rings:q_mags', q_mags, ', pens', pens)
            for (r, p) in zip(q_mags, pens):
                self.add_ring(q_mag=r, pen=p)
            return True
        if d_spacings is not None:
            pens *= int(len(d_spacings) / len(pens))
            # self.debug('add_rings:d_spacings', d_spacings, ', pens', pens)
            if repeat is True:
                d_spacings = [d_spacings[0]/i for i in range(1, 21)]
            for (r, p) in zip(d_spacings, pens):
                self.add_ring(d_spacing=r, pen=p)
            return True
        return False

    def add_ring(self, radius=None, angle=None, q_mag=None, d_spacing=None, pen=None):
        self.debug('add_ring') #, radius, angle, q_mag, d_spacing, pen)
        if angle is not None:
            if angle >= np.pi:
                return False
            radius = np.tan(angle)
        if q_mag is not None:
            a = q_mag*self.dataframe.get_beam().wavelength/(4*np.pi)
            angle = 2*np.arcsin(a)
            if angle >= np.pi:
                return False
            radius = np.tan(angle)
        if d_spacing is not None:
            a = self.dataframe.get_beam().wavelength / (2*d_spacing)
            angle = 2*np.arcsin(a)
            if angle >= np.pi:
                return False
            q_mag = 4*np.pi/d_spacing
            radius = np.tan(angle)
        if pen is None:
            pen = self.ring_pen
        ring = pg.CircleROI(pos=[-radius, -radius], size=2*radius, pen=pen, movable=False)
        ring.q_mag = q_mag
        self.rings.append(ring)
        self.viewbox.addItem(ring)
        for handle in ring.handles:
            ring.removeHandle(handle['item'])
        return True

    def update_rings(self):
        r""" Update rings (needed if the |Beam| changes). """
        self.debug('update_rings')
        if self.rings:
            for ring in self.rings:
                if ring.q_mag:
                    r = np.tan(2*np.arcsin(ring.q_mag*self.dataframe.get_beam().wavelength/(4*np.pi)))
                    ring.setState({"pos": [-r, -r], "size": 2*r, "angle": 0})

    def hide_ring_radius_handles(self):
        self.debug('hide_ring_radius_handles')
        for circ in self.rings:
            for handle in circ.handles:
                circ.removeHandle(handle['item'])

    def remove_rings(self):
        self.debug('remove_rings')
        if self.rings is None:
            return
        for i in range(0, len(self.rings)):
            self.viewbox.removeItem(self.rings[i])

    def show_grid(self):
        self.debug('show_grid')
        if self.grid is None:
            self.grid = pg.GridItem()
        self.viewbox.addItem(self.grid)

    def hide_grid(self):
        self.debug('hide_grid')
        if self.grid is not None:
            self.viewbox.removeItem(self.grid)
            self.grid = None

    def toggle_grid(self):
        self.debug('toggle_grid')
        if self.grid is None:
            self.show_grid()
        else:
            self.hide_grid()

    def show_pad_border(self, n, pen=None):
        self.debug("show_pad_border")
        if self.pad_image_items is None:
            return
        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=2)
        self.pad_image_items[n].setBorder(pen)

    def hide_pad_border(self, n):
        self.debug("hide_pad_border")
        if self.pad_image_items is None:
            return
        self.pad_image_items[n].setBorder(None)

    def show_pad_borders(self, pen=None):
        self.debug("show_pad_borders")
        if self.pad_image_items is None:
            return
        if pen is None:
            pen = pg.mkPen([0, 255, 0], width=1)
        for image in self.pad_image_items:
            image.setBorder(pen)

    def hide_pad_borders(self):
        self.debug("hide_pad_borders")
        if self.pad_image_items is None:
            return
        for image in self.pad_image_items:
            image.setBorder(None)

    def show_history_next(self):
        self.debug("show_history_next")
        self.dataframe = self.frame_getter.get_history_next()
        self.update_display()

    def show_history_previous(self):
        self.debug("show_history_previous")
        self.dataframe = self.frame_getter.get_history_previous()
        self.update_display()

    def show_next_frame(self, skip=1):
        self.debug("show_next_frame")
        self.dataframe = self.frame_getter.get_next_frame(skip=skip)
        self.update_display()

    def show_previous_frame(self, skip=1):
        self.debug("show_previous_frame")
        self.dataframe = self.frame_getter.get_previous_frame(skip=skip)
        self.update_display()

    def show_random_frame(self):
        self.debug("show_random_frame")
        self.dataframe = self.frame_getter.get_random_frame()
        self.update_display()

    def show_frame(self, frame_number=0):
        self.debug("show_frame")
        self.dataframe = self.frame_getter.get_frame(frame_number=frame_number)
        # self.debug(self.dataframe, level=2)
        self.update_display()

    def show_first_frame(self):
        self.debug("show_first_frame")
        self.dataframe = self.frame_getter.get_first_frame()
        # self.debug(self.dataframe, level=2)
        self.update_display()

    def update_display(self, dataframe=None):
        r"""
        Update display, e.g. when moving to next frame.  This includes PADs, masks, title, rings, etc.

        Arguments:
            dataframe (|DataFrame|): Optional input dataframe.  Will use self.dataframe otherwise.
        """
        self.debug("update_display")
        if dataframe is not None:
            self.dataframe = dataframe
        self.update_pads()
        self.update_masks()
        self.set_title()
        self.update_status_string()
        self.mouse_moved(self.evt)

    def add_plot_item(self, *args, **kargs):
        r"""
        Example: self.add_plot_item(x, y, pen=pg.mkPen(width=3, color='g'))
        """
        self.debug("add_plot_item")
        if self.plot_items is None:
            self.plot_items = []
        plot_item = pg.PlotDataItem(*args, **kargs)
        self.plot_items.append(plot_item)
        self.viewbox.addItem(plot_item)
        return plot_item

    def add_scatter_plot(self, *args, **kargs):
        self.debug("add_scatter_plot")
        if self.scatter_plots is None:
            self.scatter_plots = []
        scat = pg.ScatterPlotItem(*args, **kargs)
        self.scatter_plots.append(scat)
        self.viewbox.addItem(scat)

    def remove_scatter_plots(self):
        self.debug("remove_scatter_plots")
        if self.scatter_plots is not None:
            for scat in self.scatter_plots:
                self.viewbox.removeItem(scat)
        self.scatter_plots = None

    def load_geometry_file(self):
        self.debug("load_geometry_file")
        options = qwgt.QFileDialog.Options()
        file_name, file_type = qwgt.QFileDialog.getOpenFileName(self.main_window, "Load geometry file", "",
                                                          "CrystFEL geom (*.geom)", options=options)
        if file_name == "":
            return
        if file_type == "CrystFEL geom (*.geom)":
            print('CrystFEL geom not implemented.')
            pass

    # FIXME: Use DataFrame methods for saving/loading
    def load_pickled_dataframe(self, file_name):
        r""" Load data in pickle format.  Should be a dictionary with keys:

        mask_data, pad_display_data, beam, pad_geometry
        """
        self.debug("load_pickled_dataframe")
        dataframe = fileio.misc.load_pickle(file_name)
        print('Loaded pickled dictionary:' + list(dataframe.keys()).__str__())
        self.update_masks(dataframe['mask'])
        self.set_pad_display_data(dataframe['pad_display_data'])
        self.beam = dataframe['beam']
        self.update_pad_geometry(dataframe['pad_geometry'])

    def save_pickled_dataframe(self, file_name):
        r""" Save dataframe in pickle format.  It is a dictionary with the keys:

        mask_data, pad_display_data, beam, pad_geometry
        """
        dataframe = {}
        dataframe['pad_display_data'] = [p.astype(np.float32) for p in self.get_pad_display_data()]
        dataframe['mask_data'] = [m.astype(np.uint8) for m in self.mask_data]
        dataframe['pad_geometry'] = self.pad_geometry
        dataframe['beam'] = self.beam
        print('Saving pickled dictionary:' + list(dataframe.keys()).__str__())
        if file_name.split('.')[-1] != 'pkl':
            file_name = file_name + '.pkl'
        with open(file_name, "wb") as f:
            pickle.dump(dataframe, f)

    def open_data_file_dialog(self):
        self.debug("open_data_file_dialog")
        options = qwgt.QFileDialog.Options()
        file_name, file_type = qwgt.QFileDialog.getOpenFileName(self.main_window, "Load data file", "",
                                                          "Python Pickle (*.pkl)", options=options)
        if file_name == "":
            return
        if file_type == 'Python Pickle (*.pkl)':
            self.load_pickled_dataframe(file_name)

    def save_data_file_dialog(self):
        r""" Save list of masks in pickle or reborn mask format. """
        self.debug("save_data_file_dialog")
        options = qwgt.QFileDialog.Options()
        file_name, file_type = qwgt.QFileDialog.getSaveFileName(self.main_window, "Save Data Frame", "data",
                                                                 "Python Pickle (*.pkl)", options=options)
        if file_name == "":
            return
        if file_type == 'Python Pickle (*.pkl)':
            self.save_pickled_dataframe(file_name)

    def vector_coords_to_2d_display_coords(self, vecs):
        r""" Convert 3D vector coords to the equivalent coords in the 2D display plane.  This corresponds to ignoring
        the "z" coordinate, and scaling the "x,y" coordinates to that of an equivalent detector located at a distance
        of 1 meter from the origin.  Simply put: remove the z component, divide the x,y components by the z component"""
        return (vecs[:, 0:2].T/vecs[:, 2]).T.copy()

    def panel_scatter_plot(self, panel_number, ss_coords, fs_coords, style=None):
        r""" Scatter plot points given coordinates (i.e. indices) corresponding to a particular panel.  This will
        take care of the re-mapping to the display coordinates."""
        if style is None: style = self.scatterplot_style
        vecs = self.pad_geometry[panel_number].indices_to_vectors(ss_coords, fs_coords)
        vecs = self.vector_coords_to_2d_display_coords(vecs)
        self.add_scatter_plot(vecs[:, 0], vecs[:, 1], **style)

    def import_plugin_module(self, module_name):
        self.debug("import_plugin_module")
        if module_name in self.plugins:
            return self.plugins[module_name]  # Check if module already imported and cached
        module_path = __package__+'.plugins.'+module_name
        if module_path[-3:] == '.py':
            module_path = module_path[:-3]
        module_path = module_path.replace('/', '.')
        self.debug('\tImporting plugin: %s' % module_path)
        module = importlib.import_module(module_path)  # Attempt to import
        if self.plugins is None: self.plugins = {}
        self.plugins[module_name] = module  # Cache the module
        return module

    def run_plugin(self, module_name):
        self.debug('run_plugin '+module_name, 1)
        if self.plugins is None:
            self.plugins = {}
        if not module_name in self.plugins.keys():
            module = self.import_plugin_module(module_name)  # Get the module (import or retrieve from cache)
        else:
            module = self.plugins[module_name]
        if hasattr(module, 'plugin'):  # If the module has a simple plugin function, run the function and return
            module.plugin(self)
            return
        # If the plugin has a widget already, show it:
        if module_name+'.widget' in self.plugins.keys():
            self.plugins[module_name+'.widget'].show()  # Check if a widget is already cached.  If so, show it.
            return
        # If the plugin is a class with an action method, run it:
        if module_name+'.action' in self.plugins.keys():
            self.plugins[module_name+'.action']()
            return
        if hasattr(module, 'Plugin'):
            plugin_instance = module.Plugin(self)  # Check if the plugin defines a class.  If so, create an instance.
            self.plugins[module_name+'.class_instance'] = plugin_instance  # Cache the instance
            self.debug('\tCreated plugin class instance.')
            if hasattr(plugin_instance, 'widget'):
                self.plugins[module_name + '.widget'] = plugin_instance.widget  # Get the widget and cache it.
                plugin_instance.widget.show()  # Show the widget.
                self.debug('\tShowing widget.')
            if hasattr(plugin_instance, 'action'):
                self.plugins[module_name + '.action'] = plugin_instance.action  # Get the widget and cache it.
                self.debug('\tConfiguring action method.')
            return
        self.debug('\tPlugin module has no functions or classes defined.')
        return

    def run_plugins(self, module_names=[]):
        self.debug("run_plugins")
        self.debug('\tplugin module names: '+module_names.__str__(), 1)
        if len(module_names) <= 0:
            return
        for module_name in module_names:
            self.run_plugin(module_name)

    def list_plugins(self):
        return self.plugin_names

    def get_text(self, title="Title", label="Label", text="Text"):
        r""" Simple popup widget that allows the capture of a text string."""
        text, ok = qwgt.QInputDialog.getText(self.main_window, title, label, qwgt.QLineEdit.Normal, text)
        return text

    def get_float(self, title="Title", label="Label", text="Text"):
        r""" Simple popup widget that allows the capture of a float number."""
        return float(self.get_text(title=title, label=label, text=text))

    # FIXME: Figure out how to make start work without blocking the ipython terminal.
    def start(self):
        self.debug("start")
        self.app.aboutToQuit.connect(self.stop)
        if self.main:
            self.app.exec_()

    def stop(self):
        self.debug("stop")

    def show(self):
        self.debug("show")
        self.main_window.show()
        # self.main_window.callback_pb_load()

    # FIXME: This should be eliminated, if we can figure out how to start PADView without blocking the terminal.  It's
    # FIXME: purpose is for debugging.
    def call_method_by_name(self, method_name=None, *args, **kwargs):
        r""" Call a method via it's name in string format. Try not to do this... """
        self.debug("call_method_by_name")
        if method_name is None:
            method_name = self.get_text('Call method', 'Method name', '')
        self.debug('method_name: ' + method_name)
        method = getattr(self, method_name, None)
        if method is not None:
            method(*args, **kwargs)

    def save_screenshot_dialog(self):
        self.debug("save_screenshot_dialog")
        filepath, _ = self.save_file_dialog(title='Screenshot', default='padview_screenshot.jpg')
        self.save_screenshot(filepath)

    def save_screenshot(self, filename):
        self.debug('save_screenshot')
        p = self.main_window.grab()
        p.save(filename)

    def save_file_dialog(self, title='Filepath', default='file', types=None):
        options = qwgt.QFileDialog.Options()
        name, type = qwgt.QFileDialog.getSaveFileName(self.main_window, title, default, types, options=options)
        return name, type

    def tic(self, msg, level=1):
        self.debug('tic' + '  '*len(self._tic_times) + msg, level=level)
        self._tic_times.append(time.time())

    def toc(self, level=1):
        if len(self._tic_times) == 0:
            print('Something is wrong with tic/toc')
        tic = self._tic_times.pop()
        self.debug('toc' + '  '*len(self._tic_times) + f"{time.time()-tic} seconds.", level=level)


def view_pad_data(data=None, pad_geometry=None, show=True, title=None, **kwargs):
    r""" Convenience function that creates a PADView instance and starts it. """
    pv = PADView(data=data, pad_geometry=pad_geometry, **kwargs)
    if title is not None:
        pv.set_title(title)
    if show:
        pv.start()


class DummyFrameGetter(FrameGetter):
    r""" Makes a FrameGetter for a single DataFrame. """
    def __init__(self, data=None, pad_geometry=None, mask=None, beam=None, **kwargs):
        r"""
        Ideally, a |DataFrame| instance is provided, which has |PADGeometry|, |Beam|, etc. information.

        Alternatively, data can be provided as a single |ndarray|, as a list of |ndarray|, or, for legacy purposes,
        as a dictionary with the 'pad_data' key in it.  If so, then you should at least provide the |PADGeometry| and
        |Beam| information, else some garbage numbers will be created for you.

        Args:
            dataframe:
            data:
            pad_geometry:
        """
        super().__init__()
        if isinstance(data, DataFrame):
            self.dataframe = data
            self.dataframe.validate()
        else:
            # This is a mess.  It is exactly the reason why the |DataFrame| class was created...
            if mask is None:
                mask = kwargs.pop('mask_data', None)
                if mask is not None:
                    utils.depreciate('The "mask_data" keyword argument is no longer used.  Use "mask" instead.')
            if data is None:
                data = kwargs.pop('pad_data', None)
                if data is not None:
                    utils.depreciate('The "pad_data" keyword argument is no longer used.  Use "data" instead.')
            if data is None:
                data = kwargs.pop('raw_data', None)
                if data is not None:
                    utils.depreciate('The "raw_data" keyword argument is no longer used.  Use "data" instead.')
            if data is None:
                # This should never happen...
                print('WARNING: No data was provided.  Making up some *GARBAGE* data.')
                if pad_geometry is None:
                    data = np.random.rand(100, 100)
                else:
                    pad_geometry = detector.PADGeometryList(pad_geometry)
                    data = np.random.rand(pad_geometry.n_pixels)
            if isinstance(data, dict):
                data = data['pad_data']
            if pad_geometry is None:
                if isinstance(data, np.ndarray):
                    data = [data]
                logger.warning('Making up some *GARBAGE* PAD geometry because you did not provide a geometry.')
                pad_geometry = []
                shft = 0
                for dat in data:
                    if len(dat.shape) == 1:
                        logger.warning('Your PAD data is a 1D array and you did not provide geometry information.')
                    pad = detector.PADGeometry(distance=1.0, pixel_size=1.0, shape=dat.shape)
                    pad.t_vec[0] += shft
                    shft += pad.shape()[0]
                    pad_geometry.append(pad)
            pad_geometry = detector.PADGeometryList(pad_geometry)
            # Handling of beam info:
            if beam is None:
                logger.warning('Making up some *GARBAGE* beam information because you provided no specification.')
                beam = source.Beam(photon_energy=9000 * 1.602e-19)
            self.dataframe = DataFrame(raw_data=data, pad_geometry=pad_geometry, beam=beam, mask=mask)
        self.n_frames = 1

    def get_data(self, frame_number=0):
        return self.dataframe
