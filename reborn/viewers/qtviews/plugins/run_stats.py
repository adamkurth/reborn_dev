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

import tempfile
import multiprocessing
from pyqtgraph.Qt import QtCore
import pyqtgraph.Qt.QtWidgets as qwgt
from reborn.analysis.runstats import padstats, padstats_framegetter, view_padstats
from reborn.viewers.qtviews.padviews import PADView

cpu_count = multiprocessing.cpu_count()


class Plugin:
    widget = None

    def __init__(self, padview):
        self.widget = Widget(padview)
        self.widget.show()


class Widget(qwgt.QWidget):
    stats = None
    worker = None
    pv = None
    # sig_worker_done = QtCore.pyqtSignal()
    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.setWindowTitle("Run Stats")
        self.layout = qwgt.QGridLayout()
        row = 0
        row += 1
        self.layout.addWidget(qwgt.QLabel("Start Frame"), row, 1)
        self.start_frame_spinbox = qwgt.QSpinBox()
        self.start_frame_spinbox.setMinimum(0)
        self.start_frame_spinbox.setMaximum(self.padview.frame_getter.n_frames)
        self.start_frame_spinbox.setValue(0)
        self.layout.addWidget(self.start_frame_spinbox, row, 2)
        row += 1
        self.layout.addWidget(qwgt.QLabel("Stop Frame"), row, 1)
        self.stop_frame_spinbox = qwgt.QSpinBox()
        self.stop_frame_spinbox.setMinimum(1)
        self.stop_frame_spinbox.setMaximum(self.padview.frame_getter.n_frames)
        self.stop_frame_spinbox.setValue(self.padview.frame_getter.n_frames)
        self.layout.addWidget(self.stop_frame_spinbox, row, 2)
        row += 1
        self.layout.addWidget(qwgt.QLabel("Step Frame"), row, 1)
        self.step_frame_spinbox = qwgt.QSpinBox()
        self.step_frame_spinbox.setMinimum(1)
        self.step_frame_spinbox.setMaximum(self.padview.frame_getter.n_frames)
        self.step_frame_spinbox.setValue(1)
        self.layout.addWidget(self.step_frame_spinbox, row, 2)
        row += 1
        self.layout.addWidget(qwgt.QLabel("N Processes"), row, 1)
        self.np_spinbox = qwgt.QSpinBox()
        self.np_spinbox.setMinimum(1)
        self.np_spinbox.setMaximum(cpu_count)
        self.np_spinbox.setValue(1)
        self.layout.addWidget(self.np_spinbox, row, 2)
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
        tmp = tempfile.gettempdir()
        n_processes = int(self.np_spinbox.value())
        start = int(self.start_frame_spinbox.value())
        stop = int(self.stop_frame_spinbox.value())
        step = int(self.step_frame_spinbox.value())
        config = dict()
        # TODO: Get the config options into the GUI
        # config = dict(log_file=tmp+'/runstats/logs/',
        #               checkpoint_interval=250,
        #               checkpoint_file=tmp+'/runstats/checkpoints/',
        #               message_prefix="PADView:runstats")
        self.start_button.setEnabled(False)
        self.show_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker = Worker(
            self,
            framegetter=self.padview.frame_getter,
            start=start,
            stop=stop,
            step=step,
            n_processes=n_processes,
            config=config,
        )
        self.worker.start()
        self.padview.debug("runstats - Started worker")

    def padstats_completed(self):
        self.start_button.setEnabled(True)
        self.show_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.padview.debug("runstats - Worker is done")

    def show_padstats(self):
        self.padview.debug("runstats - show_padstats")
        if self.stats is None:
            self.padview.debug("runstats - No stats to show yet...")
            return
        self.stats["pad_geometry"] = self.padview.dataframe.get_pad_geometry()
        self.stats["beam"] = self.padview.dataframe.get_beam()
        if self.pv is None:
            self.pv = view_padstats(self.stats, start=False, main=False)
        else:
            self.pv.show()

    def terminate_thread(self):
        if self.worker is not None:
            self.padview.debug("runstats - Cannot terminate processes...")
            # self.worker.exit()


class Worker(QtCore.QThread):
    def __init__(self, parent, *args, **kwargs):
        super().__init__()
        self.parent = parent
        self.args = args
        self.kwargs = kwargs
        self.stats = None

    def run(self):
        self.parent.stats = padstats(*self.args, **self.kwargs)
        self.parent.padstats_completed()
        self.parent.padview.debug("runstats - Quitting thread...")
        self.quit()
