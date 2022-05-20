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
import reborn
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore


class Plugin():
    widget = None
    profiler = None
    water_profile = None
    water_plot = None
    def __init__(self, padview):
        self.padview = padview
        self.widget = Widget(padview, self)
        self.update_profile()
        self.padview.sig_dataframe_changed.connect(self.update_profile)
        self.padview.sig_beam_changed.connect(self.update_geometry)
        self.padview.sig_geometry_changed.connect(self.update_geometry)
    def update_profile(self):
        self.padview.debug()
        padview = self.padview
        pads = self.padview.dataframe.get_pad_geometry()
        beam = self.padview.dataframe.get_beam()
        mask = self.padview.dataframe.get_mask_list()
        if self.profiler is None:
            self.profiler = reborn.detector.RadialProfiler(pad_geometry=pads, beam=beam, mask=mask)
        dat = pads.concat_data(padview.get_pad_display_data()).ravel()
        if self.widget.polarization_checkbox.isChecked():
            dat /= pads.polarization_factors(beam=beam)
        if self.widget.solid_angle_checkbox.isChecked():
            dat /= pads.solid_angles()
        self.current_profile = self.profiler.get_mean_profile(dat)
        self.current_q = self.profiler.bin_centers
        self.widget.plot_widget.plot(self.current_q/1e10, self.current_profile, clear=True)
        if self.widget.water_checkbox.isChecked():
            self.plot_water_profile()
        pg.QtGui.QApplication.processEvents()
    def update_geometry(self):
        self.padview.debug()
        pads = self.padview.dataframe.get_pad_geometry()
        beam = self.padview.dataframe.get_beam()
        mask = self.padview.dataframe.get_mask_list()
        self.profiler = reborn.detector.RadialProfiler(pad_geometry=pads, beam=beam, mask=mask)
        self.update_profile()
    def plot_water_profile(self):
        self.padview.debug()
        self.water_profile = reborn.simulate.solutions.water_scattering_factor_squared(self.profiler.bin_centers)
        w = np.where((self.current_q > 1.5e10)*(self.current_q < 2.5e10))
        if len(w[0] > 0):  # Try to normalize on water ring peak
            c = np.max(self.current_profile[w])/np.max(self.water_profile[w])
        else:  # Normalize based on maximum
            c = np.max(self.current_profile)/np.max(self.water_profile)
        self.water_profile *= c
        self.water_plot = self.widget.plot_widget.plot(self.profiler.bin_centers/1e10, self.water_profile, clear=False)
        print(self.water_plot)
        pg.QtGui.QApplication.processEvents()


class Widget(QtGui.QMainWindow):
    def __init__(self, padview, plugin):
        super().__init__()
        self.padview = padview
        self.plugin = plugin
        self.setWindowTitle('Scattering Profile')
        vbox = QtGui.QVBoxLayout()

        self.plot_widget = pg.PlotWidget()
        vbox.addWidget(self.plot_widget)

        hbox = QtGui.QHBoxLayout()

        b = QtGui.QPushButton("Update Profile")
        b.clicked.connect(self.plugin.update_profile)
        hbox.addWidget(b)

        h = QtGui.QHBoxLayout()
        self.polarization_checkbox = QtGui.QCheckBox()
        h.addWidget(self.polarization_checkbox, alignment=QtCore.Qt.AlignRight)
        h.addWidget(QtGui.QLabel('Correct Polarization'))
        hbox.addLayout(h)

        h = QtGui.QHBoxLayout()
        self.solid_angle_checkbox = QtGui.QCheckBox()
        h.addWidget(self.solid_angle_checkbox, alignment=QtCore.Qt.AlignRight)
        h.addWidget(QtGui.QLabel('Correct Solid Angles'))
        hbox.addLayout(h)

        h = QtGui.QHBoxLayout()
        self.water_checkbox = QtGui.QCheckBox()
        h.addWidget(self.water_checkbox, alignment=QtCore.Qt.AlignRight)
        h.addWidget(QtGui.QLabel('Add Water Profile'))
        hbox.addLayout(h)

        vbox.addLayout(hbox)
        main_widget = QtGui.QWidget()
        main_widget.setLayout(vbox)
        self.setCentralWidget(main_widget)
