import numpy as np
from pyqtgraph import QtGui
from functools import partial

class Plugin():

    widget = None

    def __init__(self, padview):
        self.widget = Widget(padview)
        print('showing widget')
        self.widget.show()


# Todo(adeore2) : Add regex based input validator to force numeric values
#               : Property binding of frame number label and frame number variable
#               : Shuffle Functionality using Timer/Thread

class Widget(QtGui.QWidget):
    
    def __init__(self, padview):
        super().__init__()

        self.frame_number = 0
        self.frame_skip_count = 1
        self.total_frames = 100
        self.time_duration_seconds = 3
        self.autoplay_mode = False

        self.padview = padview
        self.setWindowTitle('Frame Navigator')
        self.layout = QtGui.QGridLayout()

        self.playpause_button = QtGui.QPushButton(u"⏯️", None)
        self.playpause_button.clicked.connect(self.toggle_play_pause)

        self.next_button = QtGui.QPushButton(u"⏭️", None)
        self.next_button.clicked.connect(self.get_next_frame)

        self.prev_button = QtGui.QPushButton(u"⏮", None)
        self.prev_button.clicked.connect(self.get_prev_frame)

        self.shuffle_button = QtGui.QPushButton(u"🔀", None)
        self.shuffle_button.clicked.connect(self.get_random_frame)

        self.frame_no_label = QtGui.QLabel(None)
        self.frame_no_label.setText(str(self.frame_number))

        self.frame_count_textfield = QtGui.QLineEdit(None)
        self.time_duration_textfield = QtGui.QLineEdit(None)

        row = 1
        self.layout.addWidget(self.prev_button, row, 1)
        self.layout.addWidget(self.playpause_button, row, 2)
        self.layout.addWidget(self.shuffle_button, row, 3)
        self.layout.addWidget(self.next_button, row, 4)

        row = 2
        self.layout.addWidget(QtGui.QLabel("Frame No.: ", None), row, 1)
        self.layout.addWidget(self.frame_no_label, row, 2)
        self.layout.addWidget(QtGui.QLabel("Frame Count: ", None), row, 3)
        self.layout.addWidget(self.frame_count_textfield, row, 4)
        self.layout.addWidget(QtGui.QLabel("Seconds: ", None), row, 5)
        self.layout.addWidget(self.time_duration_textfield, row, 6)

        self.setLayout(self.layout)

        QtGui.QShortcut(QtGui.QKeySequence('f'), self).activated.connect(self.get_next_frame)
        QtGui.QShortcut(QtGui.QKeySequence('b'), self).activated.connect(self.get_prev_frame)
        QtGui.QShortcut(QtGui.QKeySequence('r'), self).activated.connect(self.get_random_frame)
        QtGui.QShortcut(QtGui.QKeySequence('p'), self).activated.connect(self.toggle_play_pause)

    def get_next_frame(self):
        self.frame_number += self.frame_skip_count
        self.frame_no_label.setText(str(self.frame_number))

    def get_prev_frame(self):
        self.frame_number -= self.frame_skip_count
        self.frame_no_label.setText(str(self.frame_number))

    def get_random_frame(self):
        self.frame_number = np.random.randint(0, self.total_frames)
        self.frame_no_label.setText(str(self.frame_number))

    def toggle_play_pause(self):
        self.autoplay_mode = not self.autoplay_mode
        self.playpause_button.setEnabled(self.autoplay_mode)

    def set_time_duration(self):
        self.time_duration_seconds = int(self.time_duration_textfield.getText())

    def set_frame_count(self):
        self.frame_skip_count = int(self.frame_count_textfield.getText())
