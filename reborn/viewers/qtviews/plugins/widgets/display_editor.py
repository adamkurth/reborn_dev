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

from pyqtgraph import QtCore, QtGui
import pyqtgraph.Qt.QtWidgets as qwgt


class Plugin():
    widget = None
    def __init__(self, padview):
        self.widget = Widget(padview)
        self.widget.show()


class Widget(qwgt.QWidget):
    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.setWindowTitle('Display Editor')
        self.layout = qwgt.QVBoxLayout()
        bold = QtGui.QFont()
        bold.setBold(True)
        # ======  Upper Bound  ================
        label = qwgt.QLabel('Upper Bound')
        label.setFont(bold)
        self.layout.addWidget(label)
        layout = qwgt.QHBoxLayout()
        self.upper_fixed_checkbox = qwgt.QCheckBox()
        self.upper_fixed_checkbox.setText('Fixed Value')
        self.upper_fixed_checkbox.setChecked(False)
        self.upper_fixed_checkbox.toggled.connect(self.upper_fixed_checkbox_action)
        layout.addWidget(self.upper_fixed_checkbox)
        self.upper_fixed_spinbox = qwgt.QDoubleSpinBox()
        self.upper_fixed_spinbox.setMaximum(1e300)
        self.upper_fixed_spinbox.setMinimum(-1e300)
        if self.padview.default_levels[1] is not None:
            self.upper_fixed_spinbox.setValue(self.padview.default_levels[1])
        self.upper_fixed_spinbox.valueChanged.connect(self.upper_fixed_spinbox_action)
        layout.addWidget(self.upper_fixed_spinbox)
        self.layout.addLayout(layout)
        layout = qwgt.QHBoxLayout()
        self.upper_percentile_checkbox = qwgt.QCheckBox()
        self.upper_percentile_checkbox.setText('Percentile')
        self.upper_percentile_checkbox.setChecked(False)
        self.upper_percentile_checkbox.toggled.connect(self.upper_percentile_checkbox_action)
        layout.addWidget(self.upper_percentile_checkbox)
        self.upper_percentile_spinbox = qwgt.QDoubleSpinBox()
        self.upper_percentile_spinbox.setValue(98)
        self.upper_percentile_spinbox.setMaximum(100)
        self.upper_percentile_spinbox.setMinimum(0)
        self.upper_percentile_spinbox.valueChanged.connect(self.upper_percentile_spinbox_action)
        if self.padview.default_percentiles[1] is not None:
            self.upper_percentile_spinbox.setValue(self.padview.default_percentiles[1])
        layout.addWidget(self.upper_percentile_spinbox)
        self.layout.addLayout(layout)
        # ======  Lower Bound  ================
        label = qwgt.QLabel('Lower Bound')
        label.setFont(bold)
        self.layout.addWidget(label)
        layout = qwgt.QHBoxLayout()
        self.lower_fixed_checkbox = qwgt.QCheckBox()
        self.lower_fixed_checkbox.setText('Fixed Value')
        self.lower_fixed_checkbox.setChecked(False)
        self.lower_fixed_checkbox.toggled.connect(self.lower_fixed_checkbox_action)
        layout.addWidget(self.lower_fixed_checkbox)
        self.lower_fixed_spinbox = qwgt.QDoubleSpinBox()
        # self.lower_fixed_spinbox.setValue(6)
        self.lower_fixed_spinbox.setMaximum(1e300)
        self.lower_fixed_spinbox.setMinimum(-1e300)
        self.lower_fixed_spinbox.valueChanged.connect(self.lower_fixed_spinbox_action)
        layout.addWidget(self.lower_fixed_spinbox)
        self.layout.addLayout(layout)
        layout = qwgt.QHBoxLayout()
        self.lower_percentile_checkbox = qwgt.QCheckBox()
        self.lower_percentile_checkbox.setText('Percentile')
        self.lower_percentile_checkbox.setChecked(False)
        self.lower_percentile_checkbox.toggled.connect(self.lower_percentile_checkbox_action)
        layout.addWidget(self.lower_percentile_checkbox)
        self.lower_percentile_spinbox = qwgt.QDoubleSpinBox()
        self.lower_percentile_spinbox.setValue(2)
        self.lower_percentile_spinbox.setMaximum(100)
        self.lower_percentile_spinbox.setMinimum(0)
        self.lower_percentile_spinbox.valueChanged.connect(self.lower_percentile_spinbox_action)
        layout.addWidget(self.lower_percentile_spinbox)
        self.layout.addLayout(layout)
        # ======  General  ================
        label = qwgt.QLabel('General')
        label.setFont(bold)
        self.layout.addWidget(label)
        self.fix_button = qwgt.QPushButton("Fix Current Levels")
        self.fix_button.pressed.connect(self.fix_current_levels)
        self.layout.addWidget(self.fix_button)
        self.mirror_checkbox = qwgt.QCheckBox()
        self.mirror_checkbox.setText('Mirror Levels')
        self.mirror_checkbox.setChecked(False)
        self.mirror_checkbox.toggled.connect(self.mirror_checkbox_action)
        self.layout.addWidget(self.mirror_checkbox)
        self.ignore_masked_checkbox = qwgt.QCheckBox()
        self.ignore_masked_checkbox.setText('Ignore Masked')
        self.ignore_masked_checkbox.setChecked(True)
        self.ignore_masked_checkbox.toggled.connect(self.ignore_masked_checkbox_action)
        self.layout.addWidget(self.ignore_masked_checkbox)
        self.layout.addLayout(layout)
        self.setLayout(self.layout)
        if padview.default_levels[0] is not None:
            self.lower_fixed_spinbox.setValue(padview.default_levels[0])
        if padview.default_levels[1] is not None:
            self.upper_fixed_spinbox.setValue(padview.default_levels[1])

    def fix_current_levels(self):
        print('fix_current_levels')
        levels = self.padview.get_levels()
        self.lower_fixed_spinbox.setValue(levels[0])
        self.upper_fixed_spinbox.setValue(levels[1])
        self.lower_fixed_checkbox.setChecked(True)
        self.upper_fixed_checkbox.setChecked(True)
        self.mirror_checkbox.setChecked(False)
        self.padview.fixed_levels = self.padview.get_levels()
        self.padview.set_levels()

    def upper_fixed_checkbox_action(self):
        print('upper_fixed_checkbox_action')
        if self.upper_fixed_checkbox.isChecked():
            self.upper_percentile_checkbox.setChecked(False)
            self.padview.default_levels[1] = self.upper_fixed_spinbox.value()
            self.padview.default_percentiles[1] = None
        else:
            self.padview.default_levels[1] = None
        self.padview.previous_levels = self.padview.get_levels()
        self.padview.set_levels()

    def upper_fixed_spinbox_action(self):
        print('upper_fixed_spinbox_action')
        self.upper_fixed_checkbox_action()

    def upper_percentile_checkbox_action(self):
        print('upper_percentile_checkbox_action')
        if self.upper_percentile_checkbox.isChecked():
            self.upper_fixed_checkbox.setChecked(False)
            self.padview.default_levels[1] = None
            self.padview.default_percentiles[1] = self.upper_percentile_spinbox.value()
        else:
            self.padview.default_percentiles[1] = None
        self.padview.previous_levels = self.padview.get_levels()
        self.padview.set_levels()

    def upper_percentile_spinbox_action(self):
        print('upper_percentile_spinbox_action')
        self.upper_percentile_checkbox_action()

    def lower_fixed_checkbox_action(self):
        print('lower_fixed_checkbox_action')
        if self.lower_fixed_checkbox.isChecked():
            self.lower_percentile_checkbox.setChecked(False)
            self.padview.default_levels[0] = self.lower_fixed_spinbox.value()
            self.padview.default_percentiles[0] = None
        else:
            self.padview.default_levels[0] = None
        self.padview.previous_levels = self.padview.get_levels()
        self.padview.set_levels()

    def lower_fixed_spinbox_action(self):
        print('lower_fixed_spinbox_action')
        self.lower_fixed_checkbox_action()

    def lower_percentile_checkbox_action(self):
        print('lower_percentile_checkbox_action')
        if self.lower_percentile_checkbox.isChecked():
            self.lower_fixed_checkbox.setChecked(False)
            self.padview.default_levels[0] = None
            self.padview.default_percentiles[0] = self.lower_percentile_spinbox.value()
        else:
            self.padview.default_percentiles[0] = None
        self.padview.previous_levels = self.padview.get_levels()
        self.padview.set_levels()

    def lower_percentile_spinbox_action(self):
        print('lower_percentile_spinbox_action')
        self.lower_percentile_checkbox_action()

    def mirror_checkbox_action(self):
        print('mirror_checkbox_action')
        if self.mirror_checkbox.isChecked():
            self.upper_fixed_checkbox.setChecked(False)
            self.upper_percentile_checkbox.setChecked(False)
            self.padview.default_levels[1] = None
            self.padview.default_percentiles[1] = None
            self.padview.mirror_levels = True
        else:
            self.padview.mirror_levels = False
        self.padview.set_levels()

    def ignore_masked_checkbox_action(self):
        print('ignore_masked_checkbox_action')
        if self.ignore_masked_checkbox.isChecked():
            self.padview.levels_ignore_masked = True
        else:
            self.padview.levels_ignore_masked = False
        self.padview.set_levels()
