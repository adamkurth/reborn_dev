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

from reborn import detector
import numpy as np
import json
from pyqtgraph import QtGui
from functools import partial

class Plugin():

    widget = None

    def __init__(self, padview):
        self.widget = Widget(padview)
        print('showing widget')
        self.widget.show()


class Widget(QtGui.QWidget):
    editor_widget = None
    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.setWindowTitle('Shift Detector')
        self.layout = QtGui.QGridLayout()
        row = 0

        row += 1
        self.layout.addWidget(QtGui.QLabel('x shift (microns)'), row, 1)
        self.xsr_button = QtGui.QPushButton("-")
        self.xsr_button.clicked.connect(partial(self.shift_x, direction=-1.0))
        self.layout.addWidget(self.xsr_button, row, 2)
        sb = QtGui.QDoubleSpinBox()
        sb.setMinimum(-1e10)
        sb.setMaximum(+1e10)
        sb.setValue(1000)
        self.xs_spinbox = sb
        self.layout.addWidget(self.xs_spinbox, row, 3)
        self.xs_button = QtGui.QPushButton("+")
        self.xs_button.clicked.connect(partial(self.shift_x, direction=1.0))
        self.layout.addWidget(self.xs_button, row, 4)

        row += 1
        self.layout.addWidget(QtGui.QLabel('y shift (microns)'), row, 1)
        self.ysr_button = QtGui.QPushButton("-")
        self.ysr_button.clicked.connect(partial(self.shift_y, direction=-1.0))
        self.layout.addWidget(self.ysr_button, row, 2)
        sb = QtGui.QDoubleSpinBox()
        sb.setMinimum(-1e10)
        sb.setMaximum(+1e10)
        sb.setValue(1000)
        self.ys_spinbox = sb
        self.layout.addWidget(self.ys_spinbox, row, 3)
        self.ys_button = QtGui.QPushButton("+")
        self.ys_button.clicked.connect(partial(self.shift_y, direction=1.0))
        self.layout.addWidget(self.ys_button, row, 4)

        row += 1
        self.layout.addWidget(QtGui.QLabel('z shift (microns)'), row, 1)
        self.zsr_button = QtGui.QPushButton("-")
        self.zsr_button.clicked.connect(partial(self.shift_z, direction=-1.0))
        self.layout.addWidget(self.zsr_button, row, 2)
        sb = QtGui.QDoubleSpinBox()
        sb.setMinimum(-1e10)
        sb.setMaximum(+1e10)
        sb.setValue(1000)
        self.zs_spinbox = sb
        self.layout.addWidget(self.zs_spinbox, row, 3)
        self.zs_button = QtGui.QPushButton("+")
        self.zs_button.clicked.connect(partial(self.shift_z, direction=1.0))
        self.layout.addWidget(self.zs_button, row, 4)

        row += 1
        self.layout.addWidget(QtGui.QLabel('x rotation (degrees)'), row, 1)
        self.xrr_button = QtGui.QPushButton("-")
        self.xrr_button.clicked.connect(partial(self.rotate_x, direction=-1.0))
        self.layout.addWidget(self.xrr_button, row, 2)
        sb = QtGui.QDoubleSpinBox()
        sb.setMinimum(-90)
        sb.setMaximum(+90)
        sb.setValue(1)
        self.xr_spinbox = sb
        self.layout.addWidget(self.xr_spinbox, row, 3)
        self.xr_button = QtGui.QPushButton("+")
        self.xr_button.clicked.connect(partial(self.rotate_x, direction=1.0))
        self.layout.addWidget(self.xr_button, row, 4)

        row += 1
        self.layout.addWidget(QtGui.QLabel('y rotation (degrees)'), row, 1)
        self.yrr_button = QtGui.QPushButton("-")
        self.yrr_button.clicked.connect(partial(self.rotate_y, direction=-1.0))
        self.layout.addWidget(self.yrr_button, row, 2)
        sb = QtGui.QDoubleSpinBox()
        sb.setMinimum(-90)
        sb.setMaximum(+90)
        sb.setValue(1)
        self.yr_spinbox = sb
        self.layout.addWidget(self.yr_spinbox, row, 3)
        self.yr_button = QtGui.QPushButton("+")
        self.yr_button.clicked.connect(partial(self.rotate_y, direction=1.0))
        self.layout.addWidget(self.yr_button, row, 4)

        row += 1
        self.layout.addWidget(QtGui.QLabel('z rotation (degrees)'), row, 1)
        self.zrr_button = QtGui.QPushButton("-")
        self.zrr_button.clicked.connect(partial(self.rotate_z, direction=-1.0))
        self.layout.addWidget(self.zrr_button, row, 2)
        sb = QtGui.QDoubleSpinBox()
        sb.setMinimum(-90)
        sb.setMaximum(+90)
        sb.setValue(1)
        self.zr_spinbox = sb
        self.layout.addWidget(self.zr_spinbox, row, 3)
        self.zr_button = QtGui.QPushButton("+")
        self.zr_button.clicked.connect(partial(self.rotate_z, direction=1.0))
        self.layout.addWidget(self.zr_button, row, 4)

        row += 1
        self.direct_button = QtGui.QPushButton('Direct Edit Geometry')
        self.direct_button.clicked.connect(self.open_editor)
        self.layout.addWidget(self.direct_button, row, 1)

        self.setLayout(self.layout)
        QtGui.QShortcut(QtGui.QKeySequence('left'), self).activated.connect(partial(self.shift_x, direction=-1.0))
        QtGui.QShortcut(QtGui.QKeySequence('right'), self).activated.connect(partial(self.shift_x, direction=1.0))
        QtGui.QShortcut(QtGui.QKeySequence('up'), self).activated.connect(partial(self.shift_y, direction=-1.0))
        QtGui.QShortcut(QtGui.QKeySequence('down'), self).activated.connect(partial(self.shift_y, direction=1.0))
        
    def rotate_x(self, direction=1.0):
        ang = -float(self.xr_spinbox.value())*direction*np.pi/180
        c = np.cos(ang)
        s = np.sin(ang)
        R = np.array([[1,0,0],[0,c,s],[0,-s,c]])
        pads = self.padview.dataframe.get_pad_geometry()
        pads.rotate(R)
        self.padview.update_pad_geometry(pads)

    def rotate_y(self, direction=1.0):
        ang = float(self.yr_spinbox.value())*direction*np.pi/180
        c = np.cos(ang)
        s = np.sin(ang)
        R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        pads = self.padview.dataframe.get_pad_geometry()
        pads.rotate(R)
        self.padview.update_pad_geometry(pads)

    def rotate_z(self, direction=1.0):
        ang = -float(self.zr_spinbox.value())*direction*np.pi/180
        c = np.cos(ang)
        s = np.sin(ang)
        R = np.array([[c,s,0],[-s,c,0],[0,0,1]])
        pads = self.padview.dataframe.get_pad_geometry()
        pads.rotate(R)
        self.padview.update_pad_geometry(pads)

    def shift_x(self, direction=1.0):
        xs = float(self.xs_spinbox.value())*direction*1e-6
        ys, zs = 0, 0
        pads = self.padview.dataframe.get_pad_geometry()
        for p in pads:
            p.t_vec += np.array([xs, ys, zs])
        self.padview.update_pad_geometry(pads)

    def shift_y(self, direction=1):
        ys = float(self.ys_spinbox.value())*direction*1e-6
        xs, zs = 0, 0
        pads = self.padview.dataframe.get_pad_geometry()
        for p in pads:
            p.t_vec += np.array([xs, ys, zs])
        self.padview.update_pad_geometry(pads)

    def shift_z(self, direction=1):
        zs = float(self.zs_spinbox.value())*direction*1e-6
        xs, ys = 0, 0
        pads = self.padview.dataframe.get_pad_geometry()
        for p in pads:
            p.t_vec += np.array([xs, ys, zs])
        self.padview.update_pad_geometry(pads)

    def open_editor(self):
        if self.editor_widget is None:
            self.editor_widget = DirectEditor(padview=self.padview)
            self.editor_widget.show()


class DirectEditor(QtGui.QWidget):
    def __init__(self, padview=None):
        super().__init__()
        self.padview = padview
        self.padview.sig_geometry_changed.connect(self.geometry_updated)
        self.setWindowTitle("PAD Geometry Editor")
        # self.resize(300,270)
        self.editor = QtGui.QTextEdit()
        self.geometry_updated()
        self.update_button = QtGui.QPushButton("Update Geometry (Ctrl+Return)")
        self.update_button.setShortcut("Ctrl+Return")
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.editor)
        layout.addWidget(self.update_button)
        self.setLayout(layout)
        self.update_button.clicked.connect(self.update_geometry)

    def update_geometry(self):
        self.padview.debug()
        txt = self.editor.toPlainText()
        dicts = json.loads(txt)
        pads = []
        for d in dicts:
            p = detector.PADGeometry()
            p.from_dict(d)
            pads.append(p)
        self.padview.update_pad_geometry(pads)

    def geometry_updated(self):
        self.padview.debug()
        txt = json.dumps([g.to_dict() for g in self.padview.dataframe.get_pad_geometry()], sort_keys=True, indent=0)
        vsb = self.editor.verticalScrollBar()
        old_pos_ratio = vsb.value()
        self.editor.setPlainText(txt)
        vsb.setValue(old_pos_ratio)
