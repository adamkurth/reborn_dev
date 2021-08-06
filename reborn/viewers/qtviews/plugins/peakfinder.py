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

from pyqtgraph import QtGui, QtCore
import numpy as np

class Plugin():
    widget = None
    def __init__(self, padview):
        self.widget = Widget(padview)
        self.widget.show()


class Widget(QtGui.QWidget):

    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.setWindowTitle('Peakfinder')
        self.layout = QtGui.QGridLayout()
        row = 0
        row += 1
        self.layout.addWidget(QtGui.QLabel('Activate Peakfinder'), row, 1)
        self.activate_peakfinder_button = QtGui.QCheckBox()
        self.activate_peakfinder_button.toggled.connect(self.do_action)
        self.layout.addWidget(self.activate_peakfinder_button, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Show SNR Transform'), row, 1)
        self.activate_snrview_button = QtGui.QCheckBox()
        self.activate_snrview_button.toggled.connect(self.do_action)
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
        self.update_button.clicked.connect(self.do_action)
        self.layout.addWidget(self.update_button, row, 1, 1, 2)
        self.setLayout(self.layout)

    def do_action(self):
        self.padview.debug('PeakfinderConfigWidget.get_values()', 1)
        dat = {}
        dat['activate'] = self.activate_peakfinder_button.isChecked()
        dat['show_snr'] = self.activate_snrview_button.isChecked()
        dat['inner'] = self.inner_spinbox.value()
        dat['center'] = self.center_spinbox.value()
        dat['outer'] = self.outer_spinbox.value()
        dat['snr_threshold'] = self.snr_spinbox.value()
        dat['max_iterations'] = self.iter_spinbox.value()
        print(dat)