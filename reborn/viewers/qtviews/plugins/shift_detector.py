import numpy as np
from pyqtgraph import QtGui

class Plugin():

    widget = None

    def __init__(self, padview):
        self.widget = Widget(padview)
        print('showing widget')
        self.widget.show()


class Widget(QtGui.QWidget):
    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.setWindowTitle('Shift Detector')
        self.layout = QtGui.QGridLayout()
        row = 0
        # row += 1
        # self.layout.addWidget(QtGui.QLabel('Units'), row, 1)
        row += 1
        self.layout.addWidget(QtGui.QLabel('x shift (microns)'), row, 1)
        self.xsr_button = QtGui.QPushButton("<")
        self.xsr_button.clicked.connect(self.shift_xr)
        self.layout.addWidget(self.xsr_button, row, 2)
        sb = QtGui.QDoubleSpinBox()
        sb.setMinimum(-1e10)
        sb.setMaximum(+1e10)
        sb.setValue(0)
        self.xs_spinbox = sb
        self.layout.addWidget(self.xs_spinbox, row, 3)
        self.xs_button = QtGui.QPushButton(">")
        self.xs_button.clicked.connect(self.shift_x)
        self.layout.addWidget(self.xs_button, row, 4)
        row += 1
        self.layout.addWidget(QtGui.QLabel('y shift (microns)'), row, 1)
        sb = QtGui.QDoubleSpinBox()
        sb.setMinimum(-1e10)
        sb.setMaximum(+1e10)
        sb.setValue(0)
        self.ys_spinbox = sb
        self.layout.addWidget(self.ys_spinbox, row, 3)
        row += 1
        self.layout.addWidget(QtGui.QLabel('z shift (microns)'), row, 1)
        sb = QtGui.QDoubleSpinBox()
        sb.setMinimum(-1e10)
        sb.setMaximum(+1e10)
        sb.setValue(0)
        self.zs_spinbox = sb
        self.layout.addWidget(self.zs_spinbox, row, 3)
        self.setLayout(self.layout)

    def shift_x(self, dir=1):
        xs = self.xs_spinbox.value()*dir*1e-6
        ys, zs = 0, 0
        pads = self.padview.pad_geometry
        print('shift by', np.array([xs, ys, zs]))
        for p in pads:
            p.t_vec += np.array([xs, ys, zs])
        self.padview.update_pad_geometry(pads)

    def shift_xr(self):
        self.shift_x(dir=-1)

    def shift_y(self, dir=1):
        ys = self.ys_spinbox.value()*dir*1e-6
        xs, zs = 0, 0
        pads = self.padview.pad_geometry
        for p in pads:
            p.t_vec += np.array([xs, ys, zs])
        self.padview.update_pad_geometry(pads)

    def shift_yr(self):
        self.shift_y(dir=-1)

    def shift_z(self, dir=1):
        zs = self.zs_spinbox.value()*dir*1e-6
        xs, ys = 0, 0
        pads = self.padview.pad_geometry
        for p in pads:
            p.t_vec += np.array([xs, ys, zs])
        self.padview.update_pad_geometry(pads)

    def shift_zr(self):
        self.shift_z(dir=-1)
