import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


class Widget(QtGui.QWidget):

    def __init__(self, padview=None):
        super().__init__()

        self.setWindowTitle("SNR Filter")

        layout = QtGui.QVBoxLayout()
        widgets = [QtGui.QCheckBox,
                   QtGui.QComboBox,
                   QtGui.QDateEdit,
                   QtGui.QDateTimeEdit,
                   QtGui.QDial,
                   QtGui.QDoubleSpinBox,
                   QtGui.QFontComboBox,
                   QtGui.QLCDNumber,
                   QtGui.QLabel,
                   QtGui.QLineEdit,
                   QtGui.QProgressBar,
                   QtGui.QPushButton,
                   QtGui.QRadioButton,
                   QtGui.QSlider,
                   QtGui.QSpinBox,
                   QtGui.QTimeEdit]

        for w in widgets:
            layout.addWidget(w())

        self.setLayout(layout)
        self.show()

app = pg.mkQApp()
widget = Widget()
app.exec_()