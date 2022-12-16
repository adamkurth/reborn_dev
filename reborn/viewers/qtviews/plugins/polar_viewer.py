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

import reborn
import pyqtgraph as pg
from pyqtgraph import QtCore
import pyqtgraph.Qt.QtWidgets as qwgt


class Plugin():
    widget = None
    def __init__(self, padview):
        self.padview = padview
        self.widget = Widget(padview, self)
        self.update_profile()
    def update_profile(self):
        padview = self.padview
        dataframe = self.padview.dataframe
        profiler = reborn.detector.PolarPADAssembler(pad_geometry=dataframe.get_pad_geometry(), 
                        beam=dataframe.get_beam(), n_phi_bins=100)
        profile, _ = profiler.get_mean(padview.get_pad_display_data())
        self.widget.plot_widget.setImage(profile.T)
        qwgt.QApplication.processEvents()


class Widget(qwgt.QWidget):
    def __init__(self, padview, plugin):
        super().__init__()
        self.hbox = qwgt.QHBoxLayout()
        self.splitter = qwgt.QSplitter(QtCore.Qt.Horizontal)
        self.hbox.addWidget(self.splitter)
        self.padview = padview
        self.plugin = plugin
        self.plot_widget = pg.ImageView()#  PlotWidget()
        self.splitter.addWidget(self.plot_widget)
        self.setLayout(self.hbox)