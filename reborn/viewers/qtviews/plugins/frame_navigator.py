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


class Plugin():

    widget = None

    def __init__(self, padview):
        self.widget = Widget(padview)
        print('showing widget')
        self.widget.show()


# Todo(adeore2) : Add regex based input validator to force numeric values
#               : Property binding of frame number label and frame number variable
#               : Shuffle Functionality using Timer/Thread

def push(label, func, layout):
    r""" Make QPushButton, connect to function, add to layout"""
    b = QtGui.QPushButton(label, None)
    b.clicked.connect(func)
    layout.addWidget(b)
    return b

def label(label, layout):
    b = QtGui.QLabel(None)
    b.setText(str(label))
    layout.addWidget(b)
    return b

def text(label, layout):
    b = QtGui.QLineEdit(None)
    b.setText(str(label))
    layout.addWidget(b)
    return b

class Widget(QtGui.QWidget):
    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.setWindowTitle('Frame Navigator')
        self.layout = QtGui.QVBoxLayout()
        # ======  Back  |  Next  |  Rand  ================
        layout = QtGui.QHBoxLayout()
        push("Back", self.show_previous_frame, layout)
        push("Next", self.show_next_frame, layout)
        push("Rand", padview.show_random_frame, layout)
        self.layout.addLayout(layout)
        # ====== Go to frame  =============================
        layout = QtGui.QHBoxLayout()
        push("Go to frame", self.show_frame, layout)
        self.goto = text("1", layout)
        self.goto.returnPressed.connect(self.show_frame)
        self.layout.addLayout(layout)
        # ====== Play  | Rate  ============================
        layout = QtGui.QHBoxLayout()
        self.play = push("Play", self.toggle_play, layout)
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.show_next_frame)
        label("Rate (Hz):", layout)
        self.rate = text("1", layout)
        self.layout.addLayout(layout)
        # ====== Frame X of Y   ===========================
        layout = QtGui.QHBoxLayout()
        self.frame_num = label('', layout)
        self.layout.addLayout(layout)
        # ====== Frame ID: Z  ==============================
        layout = QtGui.QHBoxLayout()
        self.frame_id = label('', layout)
        self.layout.addLayout(layout)
        # ====== Skip  |              |  ===================
        layout = QtGui.QHBoxLayout()
        label('Skip:', layout)
        self.skip = text('1', layout)
        self.layout.addLayout(layout)
        self.dataframe_updated()
        padview.sig_dataframe_changed.connect(self.dataframe_updated)
        self.setLayout(self.layout)

    def dataframe_updated(self):
        r""" What to do when PADView DataFrame has been updated. """
        pv = self.padview
        text = 'Frame %d of %d' % (pv.frame_getter.current_frame, pv.frame_getter.n_frames)
        self.frame_num.setText(text)
        self.frame_id.setText('Frame ID: ' + str(pv.dataframe.get_frame_id()))

    def show_next_frame(self):
        self.padview.show_next_frame(skip=int(self.skip.text()))

    def show_previous_frame(self):
        self.padview.show_previous_frame(skip=int(self.skip.text()))

    def show_frame(self):
        self.padview.show_frame(int(self.goto.text()))

    def toggle_play(self):
        if self.play.text() == "Play":
            self.play.setText("Pause")
            self.play_timer.start(1000/float(self.rate.text()))
            print("Play")
        elif self.play.text() == "Pause":
            self.play.setText("Play")
            self.play_timer.stop()
            print("Pause")
        else:
            raise ValueError("Value should be Play or Pause")
