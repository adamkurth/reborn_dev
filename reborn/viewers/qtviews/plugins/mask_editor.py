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
import pyqtgraph.Qt.QtWidgets as qwgt

import reborn
import numpy as np

class Plugin():
    widget = None
    def __init__(self, padview):
        self.widget = Widget(padview)
        self.widget.show()


class Widget(qwgt.QWidget):

    previous_mask = None

    def __init__(self, padview):
        super().__init__()
        bold = QtGui.QFont()
        bold.setBold(True)
        self.padview = padview
        self.setWindowTitle('Mask Editor')
        self.layout = qwgt.QGridLayout()
        row = 0
        row += 1
        bold = QtGui.QFont()
        bold.setBold(True)
        label = QtGui.QLabel('** Type spacebar to do mask action **')
        label.setFont(bold)
        padview.set_shortcut(QtCore.Qt.Key_Space, self.apply_mask)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self).activated.connect(self.apply_mask)
        self.layout.addWidget(label, row, 1)
        row += 1
        self.clear_button = QtGui.QPushButton("Clear mask")
        self.clear_button.clicked.connect(padview.clear_masks)
        self.layout.addWidget(self.clear_button, row, 1, 1, 2)
        row += 1
        self.undo_button = QtGui.QPushButton("Undo")
        self.undo_button.clicked.connect(padview.clear_masks)
        self.layout.addWidget(self.undo_button, row, 1, 1, 2)
        row += 1
        self.visible_button = QtGui.QPushButton("Toggle mask visible")
        self.visible_button.clicked.connect(padview.toggle_masks)
        self.layout.addWidget(self.visible_button, row, 1, 1, 2)
        row += 1
        self.color_button = QtGui.QPushButton("Choose mask color")
        self.color_button.clicked.connect(padview.choose_mask_color)
        self.layout.addWidget(self.color_button, row, 1, 1, 2)
        row += 1
        self.rroi_button = QtGui.QPushButton("Add rectangle ROI")
        self.rroi_button.clicked.connect(padview.add_rectangle_roi)
        self.layout.addWidget(self.rroi_button, row, 1, 1, 2)
        row += 1
        self.rroi_button = QtGui.QPushButton("Add circle ROI")
        self.rroi_button.clicked.connect(padview.add_circle_roi)
        self.layout.addWidget(self.rroi_button, row, 1, 1, 2)
        row += 1
        self.rroi_button = QtGui.QPushButton("Mask panel edges...")
        self.rroi_button.clicked.connect(self.mask_panel_edges)
        self.layout.addWidget(self.rroi_button, row, 1, 1, 2)
        row += 1
        self.rroi_button = QtGui.QPushButton("Mask panels by names...")
        self.rroi_button.clicked.connect(self.mask_pads_by_names)
        self.layout.addWidget(self.rroi_button, row, 1, 1, 2)
        row += 1
        label = QtGui.QLabel('What to do:')
        label.setFont(bold)
        self.layout.addWidget(label, row, 1)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Mask'), row, 1)
        self.mask_radio = QtGui.QRadioButton('')
        self.mask_radio.setChecked(True)
        self.layout.addWidget(self.mask_radio, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Unmask'), row, 1)
        self.unmask_radio = QtGui.QRadioButton('')
        self.layout.addWidget(self.unmask_radio, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Invert'), row, 1)
        self.invert_radio = QtGui.QRadioButton('')
        self.layout.addWidget(self.invert_radio, row, 2, alignment=QtCore.Qt.AlignCenter)
        self.what_group = QtGui.QButtonGroup(self)
        self.what_group.addButton(self.mask_radio)
        self.what_group.addButton(self.unmask_radio)
        self.what_group.addButton(self.invert_radio)
        row += 1
        label = QtGui.QLabel('Where to do it:')
        label.setFont(bold)
        self.layout.addWidget(label, row, 1)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Inside selected ROI'), row, 1)
        self.inside_radio = QtGui.QRadioButton('')
        self.inside_radio.setChecked(True)
        self.layout.addWidget(self.inside_radio, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Outside selected ROI'), row, 1)
        self.outside_radio = QtGui.QRadioButton('')
        self.layout.addWidget(self.outside_radio, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Everywhere'), row, 1)
        self.everywhere_radio = QtGui.QRadioButton('')
        self.layout.addWidget(self.everywhere_radio, row, 2, alignment=QtCore.Qt.AlignCenter)
        self.where_group = QtGui.QButtonGroup(self)
        self.where_group.addButton(self.inside_radio)
        self.where_group.addButton(self.outside_radio)
        self.where_group.addButton(self.everywhere_radio)
        row += 1
        label = QtGui.QLabel('Additional filters:')
        label.setFont(bold)
        self.layout.addWidget(label, row, 1)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Apply only above upper threshold'), row, 1)
        self.above_upper_checkbox = QtGui.QCheckBox()
        self.layout.addWidget(self.above_upper_checkbox, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Apply only below upper threshold'), row, 1)
        self.below_upper_checkbox = QtGui.QCheckBox()
        self.layout.addWidget(self.below_upper_checkbox, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Apply only above lower threshold'), row, 1)
        self.above_lower_checkbox = QtGui.QCheckBox()
        self.layout.addWidget(self.above_lower_checkbox, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Apply only below lower threshold'), row, 1)
        self.below_lower_checkbox = QtGui.QCheckBox()
        self.layout.addWidget(self.below_lower_checkbox, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.layout.addWidget(QtGui.QLabel('Apply only to PAD under mouse cursor'), row, 1)
        self.pad_under_mouse_checkbox = QtGui.QCheckBox()
        self.layout.addWidget(self.pad_under_mouse_checkbox, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.save_button = QtGui.QPushButton("Save masks...")
        self.save_button.clicked.connect(padview.save_masks)
        self.layout.addWidget(self.save_button, row, 1, 1, 2)
        row += 1
        self.load_button = QtGui.QPushButton("Load masks...")
        self.load_button.clicked.connect(padview.load_masks)
        self.layout.addWidget(self.load_button, row, 1, 1, 2)
        self.setLayout(self.layout)

    def undo(self):
        if self.previous_mask is not None:
            m = self.padview.dataframe.get_mask_flat()
            self.padview.dataframe.set_mask(self.previous_mask)
            self.padview.update_masks()
            self.previous_mask = m

    def apply_mask(self):
        self.padview.debug()
        mask = self.padview.dataframe.get_mask_flat()
        self.previous_mask = mask.copy()
        geom = self.padview.dataframe.get_pad_geometry()
        data = geom.concat_data(self.padview.get_pad_display_data())
        select = geom.zeros(dtype=int)
        if self.everywhere_radio.isChecked():
            select[:] = 1
        else:
            inds, typ = self.padview.get_hovering_roi_indices()
            if inds is None:
                self.padview.debug('No ROI selected')
                return
            select[inds] = 1
            if self.outside_radio.isChecked():
                select = -(select - 1)
        if self.above_upper_checkbox.isChecked():
            thresh = self.padview.get_levels()[1]
            select *= (data > thresh)
        if self.below_upper_checkbox.isChecked():
            thresh = self.padview.get_levels()[1]
            select *= (data < thresh)
        if self.above_lower_checkbox.isChecked():
            thresh = self.padview.get_levels()[0]
            select *= (data > thresh)
        if self.below_lower_checkbox.isChecked():
            thresh = self.padview.get_levels()[0]
            select *= (data < thresh)
        if self.pad_under_mouse_checkbox.isChecked():
            x, y, pid = self.padview.get_pad_coords_from_mouse_pos()
            pids = []
            for (i, p) in enumerate(geom):
                d = p.zeros() + i
                pids.append(d)
            pids = geom.concat_data(pids)
            select[pids != pid] = 0
        if self.mask_radio.isChecked():
            mask[select == 1] = 0
        elif self.unmask_radio.isChecked():
            mask[select == 1] = 1
        elif self.invert_radio.isChecked():
            mask[select == 1] = -(mask[select == 1] - 1)
        self.padview.dataframe.set_mask(mask)
        self.padview.update_masks()

    def mask_panel_edges(self, n_pixels=None):
        padview = self.padview
        if n_pixels is None or n_pixels is False:
            text, ok = QtGui.QInputDialog.getText(padview.main_window, "Edge mask", "Specify number of edge pixels to mask",
                                                  QtGui.QLineEdit.Normal, "1")
            if ok:
                if text == '':
                    return
                n_pixels = int(str(text).strip())
        mask = padview.dataframe.get_mask_list()
        for i in range(len(mask)):
            mask[i] *= reborn.detector.edge_mask(mask[i], n_pixels)
        padview.dataframe.set_mask(mask)
        padview.update_masks()

    def mask_pads_by_names(self):
        padview = self.padview
        clear_labels = False
        if padview.pad_labels is None:
            padview.show_pad_labels()
            clear_labels = True
        text, ok = QtGui.QInputDialog.getText(padview.main_window, "Enter PAD names (comma separated)", "PAD names",
                                              QtGui.QLineEdit.Normal, "")
        print('ok', ok)
        if clear_labels:
            padview.hide_pad_labels()
        if ok:
            if text == '':
                return
            names = text.split(',')
            geom = padview.dataframe.get_pad_geometry()
            mask = padview.dataframe.get_mask_list()
            for i in range(padview.dataframe.n_pads):
                print(geom[i].name)
                if geom[i].name in names:
                    mask[i] *= 0
            padview.dataframe.set_mask(mask)
            padview.update_masks()