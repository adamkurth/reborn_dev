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

import multiprocessing
from pyqtgraph.Qt import QtCore
import pyqtgraph.Qt.QtWidgets as qwgt
from reborn.analysis.runstats import padstats, padstats_framegetter
from reborn.viewers.qtviews.padviews import PADView

cpu_count = multiprocessing.cpu_count()


class Plugin():
    widget = None
    def __init__(self, padview):
        self.widget = Widget(padview)
        self.widget.show()

class Widget(qwgt.QWidget):

    stats = None
    worker = None
    # sig_worker_done = QtCore.pyqtSignal()

    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.setWindowTitle('Run Stats')
        self.layout = qwgt.QGridLayout()
        row = 0
        row += 1
        self.layout.addWidget(qwgt.QLabel('Start Frame'), row, 1)
        self.start_frame_spinbox = qwgt.QSpinBox()
        self.start_frame_spinbox.setMinimum(0)
        self.start_frame_spinbox.setMaximum(self.padview.frame_getter.n_frames)
        self.start_frame_spinbox.setValue(0)
        self.layout.addWidget(self.start_frame_spinbox, row, 2)
        row += 1
        self.layout.addWidget(qwgt.QLabel('Stop Frame'), row, 1)
        self.stop_frame_spinbox = qwgt.QSpinBox()
        self.stop_frame_spinbox.setMinimum(1)
        self.stop_frame_spinbox.setMaximum(self.padview.frame_getter.n_frames)
        self.stop_frame_spinbox.setValue(self.padview.frame_getter.n_frames)
        self.layout.addWidget(self.stop_frame_spinbox, row, 2)
        row += 1
        self.layout.addWidget(qwgt.QLabel('N Processes'), row, 1)
        self.np_spinbox = qwgt.QSpinBox()
        self.np_spinbox.setMinimum(1)
        self.np_spinbox.setMaximum(cpu_count)
        self.np_spinbox.setValue(1)
        self.layout.addWidget(self.np_spinbox, row, 2)
        # row += 1
        # self.layout.addWidget(qwgt.QLabel('Threshold'), row, 1)
        # self.thresh_spinbox = qwgt.QDoubleSpinBox()
        # self.thresh_spinbox.setMinimum(0)
        # self.thresh_spinbox.setValue(8)
        # self.layout.addWidget(self.thresh_spinbox, row, 2)
        # row += 1
        # self.layout.addWidget(qwgt.QLabel('Iterations'), row, 1)
        # self.iter_spinbox = qwgt.QSpinBox()
        # self.iter_spinbox.setMinimum(1)
        # self.iter_spinbox.setValue(2)
        # self.layout.addWidget(self.iter_spinbox, row, 2)
        # row += 1
        # self.layout.addWidget(qwgt.QLabel('Start'), row, 1)
        # self.start_button = qwgt.QCheckBox()
        # self.start_button.setChecked(True)
        # self.layout.addWidget(self.start_button, row, 2, alignment=QtCore.Qt.AlignCenter)
        row += 1
        self.start_button = qwgt.QPushButton("Start")
        self.start_button.clicked.connect(self.get_padstats)
        self.layout.addWidget(self.start_button, row, 1, 1, 2)
        self.setLayout(self.layout)
        row += 1
        self.stop_button = qwgt.QPushButton("Stop")
        self.stop_button.clicked.connect(self.terminate_thread)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button, row, 1, 1, 2)
        self.setLayout(self.layout)
        row += 1
        self.show_button = qwgt.QPushButton("Show")
        self.show_button.clicked.connect(self.show_padstats)
        self.show_button.setEnabled(False)
        self.layout.addWidget(self.show_button, row, 1, 1, 2)
        self.setLayout(self.layout)

        # self.threadpool = QtCore.QThreadPool()

    def get_padstats(self):
        self.padview.debug()
        parallel = False
        np = int(self.np_spinbox.value())
        if np > 1:
            parallel = True
        start = int(self.start_frame_spinbox.value())
        stop = int(self.stop_frame_spinbox.value())
        # self.stats = padstats(framegetter=self.padview.frame_getter, start=start, stop=stop, parallel=parallel,
        #                       n_processes=np, verbose=1)
        self.worker = Worker(self, framegetter=self.padview.frame_getter, start=start, stop=stop, parallel=parallel,
                              n_processes=np, verbose=1)
        #self.start_button.setEnabled(False)
        #self.stop_button.setEnabled(True)
        #self.show_button.setEnabled(False)
        self.worker.start()

    def show_padstats(self):
        self.padview.debug()
        if self.stats is None:
            print('No stats to show yet... wait for the job to finish...')
            return
        fg = padstats_framegetter(self.stats)
        pv = PADView(frame_getter=fg, main=False)
        pv.start()

    def terminate_thread(self):
        if self.worker is not None:
            print('Attempting to exit thread...')
            self.worker.exit()


class Worker(QtCore.QThread):
    def __init__(self, parent, *args, **kwargs):
        super().__init__()
        self.parent = parent
        self.args = args
        self.kwargs = kwargs
        self.stats = None
    def run(self):
        self.parent.start_button.setEnabled(False)
        self.parent.show_button.setEnabled(False)
        self.parent.stop_button.setEnabled(True)
        self.parent.stats = padstats(*self.args, **self.kwargs)
        self.parent.start_button.setEnabled(True)
        self.parent.show_button.setEnabled(True)
        self.parent.stop_button.setEnabled(False)
        print('DEBUG:pad_stats:Worker: Thread is done')
        # self.parent.sig_worker_done.emit()
        # self.parent.show_padstats()
        # self.quit()
        # self.wait()
