from pyqtgraph import Qt, QtGui, QtCore
import reborn
import numpy as np

class Plugin():
    widget = None
    def __init__(self, padview):
        self.widget = Widget(padview)
        self.widget.show()


class Widget(QtGui.QWidget):

    def __init__(self, padview):
        super().__init__()
        bold = QtGui.QFont()
        bold.setBold(True)
        self.padview = padview
        self.setWindowTitle('Mask Editor')
        self.layout = QtGui.QGridLayout()
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

    def apply_mask(self):
        self.padview.debug('apply_mask', 1)
        mask = self.padview.dataframe.get_mask_flat()  # reborn.detector.concat_pad_data(self.padview.mask_data)
        data = None
        if self.mask_radio.isChecked():
            setval = 0
        elif self.unmask_radio.isChecked():
            setval = 1
        else:
            setval = None
        if self.everywhere_radio.isChecked():
            inds = np.arange(mask.size)
        else:
            inds, typ = self.padview.get_hovering_roi_indices()
            if inds is None:
                self.padview.debug('No ROI selected', 1)
                return
            if self.outside_radio.isChecked():
                inds = -(inds - 1)
        if self.above_upper_checkbox.isChecked():
            thresh = self.padview.get_levels()[1]
            if data is None:
                data = reborn.detector.concat_pad_data(self.padview.get_pad_display_data())
            inds[data <= thresh] = 0
        if self.below_upper_checkbox.isChecked():
            thresh = self.padview.get_levels()[1]
            if data is None:
                data = reborn.detector.concat_pad_data(self.padview.get_pad_display_data())
            inds[data > thresh] = 0
        if self.above_lower_checkbox.isChecked():
            thresh = self.padview.get_levels()[0]
            if data is None:
                data = reborn.detector.concat_pad_data(self.padview.get_pad_display_data())
            inds[data <= thresh] = 0
        if self.below_lower_checkbox.isChecked():
            thresh = self.padview.get_levels()[0]
            if data is None:
                data = reborn.detector.concat_pad_data(self.padview.get_pad_display_data())
            inds[data > thresh] = 0
        if self.pad_under_mouse_checkbox.isChecked():
            x, y, pid = self.padview.get_pad_coords_from_mouse_pos()
            inds = reborn.detector.split_pad_data(self.padview.pad_geometry, inds)
            for i in range(self.padview.n_pads):
                if i == pid:
                    continue
                inds[i][:, :] = 0
            inds = reborn.detector.concat_pad_data(inds)
        if setval is None:
            mask[inds] = -(mask[inds] - 1)
        else:
            mask[inds] = setval
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