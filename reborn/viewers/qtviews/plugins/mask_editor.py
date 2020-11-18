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
        self.color_button.clicked.connect(QtGui.QColorDialog.getColor)
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
        # number_group.addButton(r0)
        # r1 = QtGui.QRadioButton("1")
        # number_group.addButton(r1)
        # layout.addWidget(r0)
        # layout.addWidget(r1)
        # row += 1
        # self.
        # row += 1
        # self.layout.addWidget(QtGui.QLabel('Activate Peakfinder'), row, 1)
        # self.activate_peakfinder_button = QtGui.QCheckBox()
        # self.activate_peakfinder_button.toggled.connect(self.do_action)
        # self.layout.addWidget(self.activate_peakfinder_button, row, 2, alignment=QtCore.Qt.AlignCenter)
        # row += 1
        # self.layout.addWidget(QtGui.QLabel('Show SNR Transform'), row, 1)
        # self.activate_snrview_button = QtGui.QCheckBox()
        # self.activate_snrview_button.toggled.connect(self.do_action)
        # self.layout.addWidget(self.activate_snrview_button, row, 2, alignment=QtCore.Qt.AlignCenter)
        # row += 1
        # self.layout.addWidget(QtGui.QLabel('SNR Threshold'), row, 1)
        # self.snr_spinbox = QtGui.QDoubleSpinBox()
        # self.snr_spinbox.setMinimum(0)
        # self.snr_spinbox.setValue(6)
        # self.layout.addWidget(self.snr_spinbox, row, 2)
        # row += 1
        # self.layout.addWidget(QtGui.QLabel('Inner Size'), row, 1)
        # self.inner_spinbox = QtGui.QSpinBox()
        # self.inner_spinbox.setMinimum(1)
        # self.inner_spinbox.setValue(1)
        # self.layout.addWidget(self.inner_spinbox, row, 2)
        # row += 1
        # self.layout.addWidget(QtGui.QLabel('Center Size'), row, 1)
        # self.center_spinbox = QtGui.QSpinBox()
        # self.center_spinbox.setMinimum(1)
        # self.center_spinbox.setValue(5)
        # self.layout.addWidget(self.center_spinbox, row, 2)
        # row += 1
        # self.layout.addWidget(QtGui.QLabel('Outer Size'), row, 1)
        # self.outer_spinbox = QtGui.QSpinBox()
        # self.outer_spinbox.setMinimum(2)
        # self.outer_spinbox.setValue(10)
        # self.layout.addWidget(self.outer_spinbox, row, 2)
        # row += 1
        # self.layout.addWidget(QtGui.QLabel('Max Filter Iterations'), row, 1)
        # self.iter_spinbox = QtGui.QSpinBox()
        # self.iter_spinbox.setMinimum(3)
        # self.iter_spinbox.setValue(3)
        # self.layout.addWidget(self.iter_spinbox, row, 2)
        self.setLayout(self.layout)

    def apply_mask(self):
        self.padview.debug('apply_mask', 1)
        mask = reborn.detector.concat_pad_data(self.padview.mask_data)
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
            print('get inds')
            inds, typ = self.padview.get_hovering_roi_indices()
            print('got inds')
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
                inds[i][:,:] = 0
            inds = reborn.detector.concat_pad_data(inds)
        if setval is None:
            mask[inds] = -(mask[inds] - 1)
        else:
            mask[inds] = setval
        self.padview.update_masks(reborn.detector.split_pad_data(self.padview.pad_geometry, mask))
    # def do_action(self):
    #     self.padview.debug('PeakfinderConfigWidget.get_values()', 1)
    #     dat = {}
    #     dat['activate'] = self.activate_peakfinder_button.isChecked()
    #     dat['show_snr'] = self.activate_snrview_button.isChecked()
    #     dat['inner'] = self.inner_spinbox.value()
    #     dat['center'] = self.center_spinbox.value()
    #     dat['outer'] = self.outer_spinbox.value()
    #     dat['snr_threshold'] = self.snr_spinbox.value()
    #     dat['max_iterations'] = self.iter_spinbox.value()
    #     print(dat)