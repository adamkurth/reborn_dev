from time import time
import numpy as np
import pyqtgraph as pg
from reborn.viewers.qtviews.padviews import get_caller
from pyqtgraph.Qt import QtGui, QtCore
from reborn.analysis.peaks import boxsnr

class Plugin():
    widget = None
    def __init__(self, padview):
        self.widget = Widget(padview)
        self.widget.show()

class Widget(QtGui.QWidget):

    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.setWindowTitle('SNR Filter')
        self.layout = QtGui.QGridLayout()
        row = 0
        row += 1
        self.layout.addWidget(QtGui.QLabel('Update Colorbar'), row, 1)
        self.update_colorbar_button = QtGui.QCheckBox()
        self.update_colorbar_button.setChecked(True)
        self.layout.addWidget(self.update_colorbar_button, row, 2, alignment=QtCore.Qt.AlignCenter)
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
        self.outer_spinbox.setMinimum(1)
        self.outer_spinbox.setValue(10)
        self.layout.addWidget(self.outer_spinbox, row, 2)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Threshold'), row, 1)
        self.thresh_spinbox = QtGui.QDoubleSpinBox()
        self.thresh_spinbox.setMinimum(0)
        self.thresh_spinbox.setValue(8)
        self.layout.addWidget(self.thresh_spinbox, row, 2)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Iterations'), row, 1)
        self.iter_spinbox = QtGui.QSpinBox()
        self.iter_spinbox.setMinimum(1)
        self.iter_spinbox.setValue(2)
        self.layout.addWidget(self.iter_spinbox, row, 2)
        row += 1
        self.update_button = QtGui.QPushButton("Apply Filter")
        self.update_button.clicked.connect(self.apply_snr_filter)
        self.layout.addWidget(self.update_button, row, 1, 1, 2)
        self.setLayout(self.layout)

    def apply_snr_filter(self):
        self.padview.debug(get_caller(), 1)
        dataframe = self.padview.dataframe
        inner = self.inner_spinbox.value()
        center = self.center_spinbox.value()
        outer = self.outer_spinbox.value()
        threshold = self.thresh_spinbox.value()
        max_iterations = self.iter_spinbox.value()
        t = time()
        data = dataframe.get_processed_data_list()
        mask = dataframe.get_mask_list()
        processed_data = [None]*dataframe.n_pads
        for i in range(dataframe.n_pads):
            d = data[i]
            m1 = mask[i]
            m2 = mask[i].copy()
            for j in range(max_iterations):
                snr, _ = boxsnr(d, m1, m2, inner, center, outer)
                ab = snr > threshold
                ab *= snr < -threshold
                m2[ab] = 0
            processed_data[i] = snr
        dataframe.set_processed_data(processed_data)
        self.padview.dataframe = dataframe
        self.padview.update_display_data()
        if self.update_colorbar_button.isChecked():
            self.padview.set_levels(0, 10)
        self.padview.debug('%g seconds' % (time()-t,))
