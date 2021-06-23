import numpy as np
from pyqtgraph import QtGui
from functools import partial

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
        self.setLayout(self.layout)
        QtGui.QShortcut(QtGui.QKeySequence('left'), self).activated.connect(partial(self.shift_x, direction=-1.0))
        QtGui.QShortcut(QtGui.QKeySequence('right'), self).activated.connect(partial(self.shift_x, direction=1.0))
        QtGui.QShortcut(QtGui.QKeySequence('up'), self).activated.connect(partial(self.shift_y, direction=-1.0))
        QtGui.QShortcut(QtGui.QKeySequence('down'), self).activated.connect(partial(self.shift_y, direction=1.0))

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
