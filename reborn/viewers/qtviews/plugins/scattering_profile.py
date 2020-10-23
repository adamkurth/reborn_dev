import reborn
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore


class Plugin():
    widget = None
    def __init__(self, padview):
        self.padview = padview
        self.widget = Widget(padview, self)
        self.update_profile()
    def update_profile(self):
        padview = self.padview
        profiler = reborn.detector.RadialProfiler(pad_geometry=padview.pad_geometry, beam=padview.beam, mask=padview.mask_data)
        profile = profiler.get_mean_profile(padview.get_pad_display_data())
        self.widget.plot_widget.plot(profiler.bin_centers, profile)
        pg.QtGui.QApplication.processEvents()


class Widget(QtGui.QWidget):
    def __init__(self, padview, plugin):
        super().__init__()
        self.hbox = QtGui.QHBoxLayout()
        self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.hbox.addWidget(self.splitter)
        self.padview = padview
        self.plugin = plugin
        self.plot_widget = pg.PlotWidget()
        self.splitter.addWidget(self.plot_widget)
        self.setLayout(self.hbox)