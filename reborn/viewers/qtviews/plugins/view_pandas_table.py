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

from pyqtgraph.Qt import QtGui
from reborn.viewers.pandaviews import DataFrameWidget

class Plugin():
    widget = None
    def __init__(self, padview):
        self.widget = Widget(padview)
        self.widget.show()

class DFW(DataFrameWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Widget(QtGui.QWidget):

    def __init__(self, padview):
        super().__init__()
        self.padview = padview
        self.setWindowTitle('Pandas Table')
        self.layout = QtGui.QGridLayout()
        row = 0
        row += 1
        self.pandas_widget = DFW(self.padview.main_window, self.padview.frame_getter.pandas_dataframe)
        self.pandas_widget.doubleClicked.connect(self.on_double_click)
        self.layout.addWidget(self.pandas_widget, row, 1)
        self.setLayout(self.layout)

    def on_double_click(self, model_index):
        row = model_index.row()
        # column = model_index.column()
        self.padview.show_frame(frame_number=row)

