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

import os
import numpy as np
import reborn
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore
from reborn.analysis.optimize import fit_ellipse_pad


class Plugin():
    widget = None
    def __init__(self, padview):
        padview.debug(os.path.basename(__file__))
        self.widget = Widget(padview)


class Widget(QtGui.QWidget):

    show_ellipse = False
    ellipse_items = []
    ellipse_fits = []

    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.setWindowTitle('Fit Ellipse')
        self.layout = QtGui.QGridLayout()
        row = 0
        row += 1
        self.layout.addWidget(QtGui.QLabel('Show Ellipse'), row, 1)
        self.show_ellipse_button = QtGui.QCheckBox()
        self.show_ellipse_button.setChecked(True)
        self.show_ellipse = True
        self.show_ellipse_button.toggled.connect(self.toggle_show_ellipse)
        self.layout.addWidget(self.show_ellipse_button, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.update_button = QtGui.QPushButton("Fit Ellipse")
        self.update_button.clicked.connect(self.do_action)
        self.layout.addWidget(self.update_button, row, 1, 1, 2)
        row += 1
        self.update_geometry_button = QtGui.QPushButton("Update Geometry")
        self.update_geometry_button.clicked.connect(self.update_geometry)
        self.layout.addWidget(self.update_geometry_button, row, 1, 1, 2)
        self.setLayout(self.layout)

    def toggle_show_ellipse(self):
        if self.show_ellipse:
            self.show_ellipse = False
        else:
            self.show_ellipse = True

    def update_geometry(self):
        pads = self.padview.dataframe.get_pad_geometry()
        X0 = np.mean(np.array([f[8] for f in self.ellipse_fits]))
        Y0 = np.mean(np.array([f[9] for f in self.ellipse_fits]))
        for p in pads:
            p.t_vec[0] -= X0*p.t_vec[2]
            p.t_vec[1] -= Y0*p.t_vec[2]
        for e in self.ellipse_items:
            self.padview.viewbox.removeItem(e)
        self.ellipse_items = []
        self.padview.update_pad_geometry(pads)
        for f in self.ellipse_fits:
            f[8] -= X0
            f[9] -= Y0
            self.draw_ellipse(f)

    def do_action(self):
        pads = self.padview.dataframe.get_pad_geometry()
        masks = self.padview.dataframe.get_mask_list()
        efit = fit_ellipse_pad(pads, masks, threshold=0.5)
        self.ellipse_fits.append(efit)
        self.draw_ellipse(efit)

    def draw_ellipse(self, efit):
        a = efit[6]
        b = efit[7]
        X0 = efit[8]
        Y0 = efit[9]
        theta = -efit[10]
        phi = np.arange(1000) * 2 * np.pi / (999)
        X = a * np.cos(phi)
        Y = b * np.sin(phi)
        x = X * np.cos(theta) + Y * np.sin(theta) + X0
        y = -X * np.sin(theta) + Y * np.cos(theta) + Y0
        if self.show_ellipse:
            self.ellipse_items.append(self.padview.add_plot_item(x, y, pen=pg.mkPen(width=3, color='g')))
