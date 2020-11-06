import numpy as np
from pyqtgraph import QtGui, QtCore
from reborn.external.pyqtgraph import MultiHistogramLUTWidget
import pyqtgraph as pg
import reborn
import tracemalloc
tracemalloc.start(10)

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
        return self.padview.pad_geometry

    def get_pad_display_data(self):
        r""" We subtract the Friedel mate from the current display data in padview. """
        if self.data_diff is not None:
            return self.data_diff
        data = self.padview.get_pad_display_data()
        mask = self.padview.mask_data
        for i in range(self.n_pads):
            data[i] = (data[i].copy()*mask[i]).astype(np.float32)
        pads = self.pad_geometry
        data_diff = [d.copy() for d in data]
        mask_diff = [p.zeros().astype(np.float32) for p in pads]
        for i in range(self.n_pads):
            vecs = pads[i].position_vecs()
            for j in range(self.n_pads):
                v = vecs.copy().astype(np.float32)
                v[:, 0:2] *= -1  # Invert vectors
                del vecs
                x, y = pads[j].vectors_to_indices(v, insist_in_pad=True, round=True)
                del v
                w = np.where(np.isfinite(x))
                x = x[w]
                y = y[w]
                data_diff[i].flat[w] -= data[j][x.astype(int), y.astype(int)]
                mask_diff[i].flat[w] += np.abs(data[j][x.astype(int), y.astype(int)])
            mask_diff[i] *= mask[i]
        for i in range(self.n_pads):
            m = mask_diff[i]
            m[m > 0] = 1
            data_diff[i] *= m
        return data_diff

    def setup_pads(self):
        # self.debug(get_caller(), 1)
        pad_data = self.get_pad_display_data()
        self.images = []
        for i in range(0, self.n_pads):
            d = pad_data[i]
            im = pg.ImageItem(d)
            self._apply_pad_transform(im, self.pad_geometry[i])
            self.images.append(im)
            self.viewbox.addItem(im)
        self.setup_histogram_tool()
        m = max(self.get_levels())
        self.set_levels(levels=(-m, m))
        self.set_preset_colormap('bipolar2')

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

    def set_preset_colormap(self, preset='flame'):
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
            self.set_preset_colormap(colormap)
        if levels is not None:
            min_value = levels[0]
            max_value = levels[1]
        if (min_value is None) or (max_value is None):
            self.set_levels_by_percentiles(percents=percentiles)
        else:
            self.histogram.item.setLevels(float(min_value), float(max_value))

    def update_pads(self):
        # self.debug(get_caller(), 1)
        # if self.images is None:
        #     self.setup_pads()
        processed_data = self.get_pad_display_data()
        for i in range(0, self.n_pads):
            self.images[i].setImage(processed_data[i])
        # self.histogram.regionChanged()

    @QtCore.pyqtSlot()
    def update_pad_geometry(self):
        self.data_diff = None
        time1 = tracemalloc.take_snapshot()
        self.update_pads()
        for i in range(0, self.n_pads):
            self._apply_pad_transform(self.images[i], self.pad_geometry[i])
        m = np.max(concat(self.get_pad_display_data()))
        time2 = tracemalloc.take_snapshot()
        stats = time2.compare_to(time1, 'lineno')
        for stat in stats[:3]:
            print(stat)
        self.set_levels(levels=(-m, m))


# class WidgetOld(QtGui.QWidget):
#     def __init__(self, padview):
#         super().__init__()
#         self.padview = padview
#         self.hbox = QtGui.QHBoxLayout()
#         self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
#         self.graphics_view = pg.GraphicsView()
#         self.viewbox = pg.ViewBox()
#         self.viewbox.invertX()
#         self.viewbox.setAspectLocked()
#         self.graphics_view.setCentralItem(self.viewbox)
#         self.splitter.addWidget(self.graphics_view)
#         self.histogram = MultiHistogramLUTWidget()
#         self.splitter.addWidget(self.histogram)
#         self.hbox.addWidget(self.splitter)
#         self.setLayout(self.hbox)
#         self.setup_pads()
#         self.setWindowTitle('Central symmetry')
#         # self.set_levels_by_percentiles((20, 80))
#         # m = max(self.get_levels())
#         # self.set_levels(-m, m)
#         # self.set_preset_colormap('bipolar')
#         self.padview.sig_geometry_changed.connect(self.update_pad_geometry)
#
#     @property
#     def n_pads(self):
#         return self.padview.n_pads
#
#     @property
#     def pad_geometry(self):
#         return self.padview.pad_geometry
#
#     def setup_pads(self):
#         # self.debug(get_caller(), 1)
#         pad_data = self.padview.get_pad_display_data()
#         self.images = []
#         self.images_inv = []
#         for i in range(0, self.n_pads):
#             d = pad_data[i]
#             im = pg.ImageItem(d)
#             # im.setOpts(opacity=0.5)
#             # im.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
#             self._apply_pad_transform(im, self.pad_geometry[i])
#             self.images.append(im)
#             self.viewbox.addItem(im)
#         for i in range(0, self.n_pads):
#             d = pad_data[i]
#             im = pg.ImageItem(-d)
#             im.setOpts(opacity=0.5)
#             # im.setCompositionMode(QtGui.QPainter.CompositionMode_)
#             self._apply_pad_transform_inv(im, self.pad_geometry[i])
#             self.images.append(im)
#             self.viewbox.addItem(im)
#             # self.histogram.regionChanged()
#         self.setup_histogram_tool()
#         m = max(self.get_levels())
#         self.set_levels(levels=(-m, m))
#         self.set_preset_colormap('bipolar2')
#         # self.setup_masks()
#         # self.set_levels(np.percentile(np.ravel(pad_data), 2), np.percentile(np.ravel(pad_data), 98))
#         # self.set_preset_colormap('grey')
#
#     def setup_histogram_tool(self):
#         # self.debug(get_caller(), 1)
#         self.histogram.setImageItems(self.images)
#
#     def _apply_pad_transform(self, im, p):
#         # self.debug(get_caller(), 2)
#         f = p.fs_vec.copy()
#         s = p.ss_vec.copy()
#         t = p.t_vec.copy() - (f + s)/2.0
#         trans = QtGui.QTransform()
#         trans.setMatrix(s[0], s[1], s[2], f[0], f[1], f[2], t[0], t[1], t[2])
#         im.setTransform(trans)
#
#     def _apply_pad_transform_inv(self, im, p):
#         # self.debug(get_caller(), 2)
#         f = p.fs_vec.copy()
#         s = p.ss_vec.copy()
#         t = p.t_vec.copy() - (f + s)/2.0
#         trans = QtGui.QTransform()
#         trans.setMatrix(-s[0], -s[1], s[2], -f[0], -f[1], f[2], -t[0], -t[1], t[2])
#         im.setTransform(trans)
#
#     def set_preset_colormap(self, preset='flame'):
#         r""" Change the colormap to one of the presets configured in pyqtgraph.  Right-click on the colorbar to find
#         out what values are allowed.
#         """
#         # self.debug(get_caller(), 1)
#         self.histogram.gradient.loadPreset(preset)
#         self.histogram.setImageItems(self.images)
#         pg.QtGui.QApplication.processEvents()
#
#     def set_levels_by_percentiles(self, percents=(1, 99), colormap=None):
#         r""" Set upper and lower levels according to percentiles.  This is based on :func:`numpy.percentile`. """
#         # self.debug(get_caller(), 1)
#         d = reborn.detector.concat_pad_data(self.padview.get_pad_display_data())
#         lower = np.percentile(d, percents[0])
#         upper = np.percentile(d, percents[1])
#         self.set_levels(lower, upper, colormap=colormap)
#
#     def get_levels(self):
#         r""" Get the minimum and maximum levels of the current image display. """
#         return self.histogram.item.getLevels()
#
#     def set_levels(self, min_value=None, max_value=None, levels=None, percentiles=None, colormap=None):
#         r""" Set the minimum and maximum levels, same as sliding the yellow sliders on the histogram tool. """
#         # self.debug(get_caller(), 1)
#         if colormap is not None:
#             self.set_preset_colormap(colormap)
#         if levels is not None:
#             min_value = levels[0]
#             max_value = levels[1]
#         if (min_value is None) or (max_value is None):
#             self.set_levels_by_percentiles(percents=percentiles)
#         else:
#             self.histogram.item.setLevels(float(min_value), float(max_value))
#         print(float(min_value), float(max_value))
#
#     @QtCore.pyqtSlot()
#     def update_pad_geometry(self):
#         for i in range(0, self.n_pads):
#             self._apply_pad_transform(self.images[i], self.pad_geometry[i])
#             self._apply_pad_transform_inv(self.images[i+self.n_pads], self.pad_geometry[i])
