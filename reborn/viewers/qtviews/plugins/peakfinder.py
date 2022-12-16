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

from pyqtgraph import QtCore
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
        self.setWindowTitle('Peakfinder')
        self.layout = qwgt.QGridLayout()
        row = 0
        row += 1
        self.layout.addWidget(qwgt.QLabel('Activate Peakfinder'), row, 1)
        self.activate_peakfinder_button = qwgt.QCheckBox()
        self.activate_peakfinder_button.toggled.connect(self.do_action)
        self.layout.addWidget(self.activate_peakfinder_button, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(qwgt.QLabel('Show SNR Transform'), row, 1)
        self.activate_snrview_button = qwgt.QCheckBox()
        self.activate_snrview_button.toggled.connect(self.do_action)
        self.layout.addWidget(self.activate_snrview_button, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(qwgt.QLabel('SNR Threshold'), row, 1)
        self.snr_spinbox = qwgt.QDoubleSpinBox()
        self.snr_spinbox.setMinimum(0)
        self.snr_spinbox.setValue(6)
        self.layout.addWidget(self.snr_spinbox, row, 2)
        row += 1
        self.layout.addWidget(qwgt.QLabel('Inner Size'), row, 1)
        self.inner_spinbox = qwgt.QSpinBox()
        self.inner_spinbox.setMinimum(1)
        self.inner_spinbox.setValue(1)
        self.layout.addWidget(self.inner_spinbox, row, 2)
        row += 1
        self.layout.addWidget(qwgt.QLabel('Center Size'), row, 1)
        self.center_spinbox = qwgt.QSpinBox()
        self.center_spinbox.setMinimum(1)
        self.center_spinbox.setValue(5)
        self.layout.addWidget(self.center_spinbox, row, 2)
        row += 1
        self.layout.addWidget(qwgt.QLabel('Outer Size'), row, 1)
        self.outer_spinbox = qwgt.QSpinBox()
        self.outer_spinbox.setMinimum(2)
        self.outer_spinbox.setValue(10)
        self.layout.addWidget(self.outer_spinbox, row, 2)
        row += 1
        self.layout.addWidget(qwgt.QLabel('Max Filter Iterations'), row, 1)
        self.iter_spinbox = qwgt.QSpinBox()
        self.iter_spinbox.setMinimum(3)
        self.iter_spinbox.setValue(3)
        self.layout.addWidget(self.iter_spinbox, row, 2)
        row += 1
        self.update_button = qwgt.QPushButton("Update Peakfinder")
        self.update_button.clicked.connect(self.do_action)
        self.layout.addWidget(self.update_button, row, 1, 1, 2)
        self.setLayout(self.layout)

    def display_peaks(self):
        r""" Scatter plot the peaks that are cached in the class instance. """
        self.debug()
        peaks = self.get_peak_data()
        if peaks is None:
            return
        centroids = peaks['centroids']
        for i in range(self.n_pads):
            c = centroids[i]
            if c is not None:
                self.panel_scatter_plot(i, c[:, 1], c[:, 0])

    def show_peaks(self):
        r""" Make peak scatter plots visible. """
        self.debug()
        self.display_peaks()
        self.peaks_visible = True

    def hide_peaks(self):
        r""" Make peak scatter plots invisible. """
        self.debug()
        self.remove_scatter_plots()
        self.peaks_visible = False

    def toggle_peaks_visible(self):
        r""" Toggle peak scatter plots visible/invisible. """
        self.debug()
        if self.peaks_visible == False:
            self.display_peaks()
            self.peaks_visible = True
        else:
            self.hide_peaks()
            self.peaks_visible = False

    # FIXME: This goes into peak finding widget
    def get_peak_data(self):
        r""" Fetch peak data, which might be stored in various places.
        FIXME: Need to simplify the data structure so that it is not a hassle to find peaks."""
        self.debug()
        # if self.processed_data is not None:
        #     self.debug('Getting processed peak data')
        #     if 'peaks' in self.processed_data.keys():
        #         return self.processed_data['peaks']
        # if self.raw_data is not None:
        #     self.debug('Getting raw peak data')
        #     if 'peaks' in self.raw_data.keys():
        #         return self.raw_data['peaks']
        return None

    def update_peakfinder_params(self):
        r""" Reset the peak finders with new parameters.  This also launges a peakfinding job.
        FIXME: Need to make this more intelligent so that unnecessary jobs are not launched."""
        self.peakfinder_params = self.widget_peakfinder_config.get_values()
        self.setup_peak_finders()
        self.find_peaks()
        self.hide_peaks()
        if self.peakfinder_params['activate']:
            self.show_peaks()
        else:
            self.hide_peaks()

    def find_peaks(self):
        r""" Launch a peak-finding job, and cache the results.  This will not display anything. """
        self.debug()
        if self.peak_finders is None:
            self.setup_peak_finders()
        centroids = [None]*self.n_pads
        n_peaks = 0
        for i in range(self.n_pads):
            pfind = self.peak_finders[i]
            pfind.find_peaks(data=self.raw_data['pad_data'][i], mask=self.mask_data[i])
            n_peaks += pfind.n_labels
            centroids[i] = pfind.centroids
        self.debug('Found %d peaks' % (n_peaks))
        self.raw_data['peaks'] = {'centroids': centroids, 'n_peaks': n_peaks}

    def toggle_peak_finding(self):
        r""" Toggle peakfinding on/off.  Set this to true if you want to automatically do peakfinding when a new
        image data is displayed. """
        self.debug()
        if self.do_peak_finding is False:
            self.do_peak_finding = True
        else:
            self.do_peak_finding = False
        self.update_display_data()

    def setup_peak_finders(self):
        r""" Create peakfinder class instances.  We use peakfinder classes rather than functions in order to tidy up
        the data structure. """
        self.debug()
        self.peak_finders = []
        a = self.peakfinder_params['inner']
        b = self.peakfinder_params['center']
        c = self.peakfinder_params['outer']
        t = self.peakfinder_params['snr_threshold']
        for i in range(self.n_pads):
            self.peak_finders.append(PeakFinder(mask=self.mask_data[i], radii=(a, b, c), snr_threshold=t))

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