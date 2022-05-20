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

import numpy as np
from pyqtgraph import QtGui, QtCore
from reborn.external.pyqtgraph import MultiHistogramLUTWidget
import pyqtgraph as pg
import reborn
# import tracemalloc
# tracemalloc.start(10)

concat = reborn.detector.concat_pad_data

class Plugin():

    widget = None

    def __init__(self, padview):
        self.widget = Widget(padview)
        m = max(self.widget.get_levels())
        self.widget.set_levels(levels=(-m, m))
        print('showing widget')
        self.widget.show()


class Widget(QtGui.QWidget):
    data_diff = None
    autoscale = True
    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.hbox = QtGui.QHBoxLayout()
        self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.graphics_view = pg.GraphicsView()
        self.viewbox = pg.ViewBox()
        self.viewbox.invertX()
        self.viewbox.setAspectLocked()
        self.graphics_view.setCentralItem(self.viewbox)
        self.splitter.addWidget(self.graphics_view)
        self.histogram = MultiHistogramLUTWidget()
        self.splitter.addWidget(self.histogram)
        self.hbox.addWidget(self.splitter)
        self.setLayout(self.hbox)
        self.setup_pads()
        self.setWindowTitle('Central symmetry')
        self.padview.sig_geometry_changed.connect(self.update_pad_geometry)

    @property
    def n_pads(self):
        return self.padview.n_pads

    @property
    def pad_geometry(self):
        return self.padview.dataframe.get_pad_geometry()

    def get_pad_display_data(self):
        r""" We subtract the Friedel mate from the current display data in padview. """
        if self.data_diff is not None:
            return self.data_diff
        pv = self.padview
        df = self.padview.dataframe
        return reborn.detector.subtract_pad_friedel_mate(pv.get_pad_display_data(), df.get_mask_list(),
                                                         df.get_pad_geometry())

    def setup_pads(self):
        # self.debug(get_caller(), 1)
        data = self.get_pad_display_data()
        geom = self.padview.dataframe.get_pad_geometry()
        self.images = []
        for i in range(0, self.padview.dataframe.n_pads):
            d = data[i]
            im = pg.ImageItem(d)
            self._apply_pad_transform(im, geom[i])
            self.images.append(im)
            self.viewbox.addItem(im)
        self.setup_histogram_tool()
        m = max(self.get_levels())
        self.set_levels(levels=(-m, m))
        self.set_colormap('bipolar2')

    def setup_histogram_tool(self):
        # self.debug(get_caller(), 1)
        self.histogram.setImageItems(self.images)

    def _apply_pad_transform(self, im, p):
        # self.debug(get_caller(), 2)
        f = p.fs_vec.copy()
        s = p.ss_vec.copy()
        t = p.t_vec.copy() - (f + s)/2.0
        trans = QtGui.QTransform()
        trans.setMatrix(s[0], s[1], s[2], f[0], f[1], f[2], t[0], t[1], t[2])
        im.setTransform(trans)

    def set_colormap(self, preset='flame'):
        r""" Change the colormap to one of the presets configured in pyqtgraph.  Right-click on the colorbar to find
        out what values are allowed.
        """
        # self.debug(get_caller(), 1)
        self.histogram.gradient.loadPreset(preset)
        self.histogram.setImageItems(self.images)
        pg.QtGui.QApplication.processEvents()

    def set_levels_by_percentiles(self, percents=(1, 99), colormap=None):
        r""" Set upper and lower levels according to percentiles.  This is based on :func:`numpy.percentile`. """
        # self.debug(get_caller(), 1)
        d = concat(self.get_pad_display_data())
        lower = np.percentile(d, percents[0])
        upper = np.percentile(d, percents[1])
        self.set_levels(lower, upper, colormap=colormap)

    def get_levels(self):
        r""" Get the minimum and maximum levels of the current image display. """
        return self.histogram.item.getLevels()

    def set_levels(self, min_value=None, max_value=None, levels=None, percentiles=None, colormap=None):
        r""" Set the minimum and maximum levels, same as sliding the yellow sliders on the histogram tool. """
        # self.debug(get_caller(), 1)
        if colormap is not None:
            self.set_colormap(colormap)
        if levels is not None:
            min_value = levels[0]
            max_value = levels[1]
        if (min_value is None) or (max_value is None):
            self.set_levels_by_percentiles(percents=percentiles)
        else:
            self.histogram.item.setLevels(float(min_value), float(max_value))

    def update_pads(self):
        levels = self.get_levels()
        processed_data = self.get_pad_display_data()
        for i in range(0, self.padview.dataframe.n_pads):
            self.images[i].setImage(processed_data[i])
        self.set_levels(levels=levels)

    @QtCore.pyqtSlot()
    def update_pad_geometry(self):
        self.data_diff = None
        levels = self.get_levels()
        self.update_pads()
        for i in range(0, self.padview.dataframe.n_pads):
            self._apply_pad_transform(self.images[i], self.pad_geometry[i])
        self.set_levels(levels=levels)  # FIXME: Why must this be done?  Very annoying...
