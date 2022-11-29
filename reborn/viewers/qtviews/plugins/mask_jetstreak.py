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
import numpy as np
import reborn
from reborn.analysis.masking import StreakMasker
from reborn import detector
from reborn.external.pyqtgraph import ImView
from reborn.viewers.qtviews.padviews import PADView
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore


class Plugin():
    masker = None
    def __init__(self, padview: PADView):
        self.padview = padview
        self.padview.debug_level = 1
        self.make_mask()
        padview.sig_dataframe_changed.connect(self.make_mask)
    def make_mask(self):
        dataframe = self.padview.dataframe
        mask = dataframe.get_mask_flat()
        if self.masker is None:
            geom = dataframe.get_pad_geometry()
            beam = dataframe.get_beam()
            self.masker = StreakMasker(geom=geom, beam=beam)
        data = self.padview.get_pad_display_data()
        smask = self.masker.get_mask(data, mask)
        print(np.min(smask), np.max(smask))
        # self.padview.dataframe.set_mask(smask*mask)
        self.padview.update_masks(smask*mask) #mask*smask)
        # self.padview.show_masks()
        # self.padview.set_pad_display_data(smask)
        pg.QtGui.QApplication.processEvents()
        print('Jetstreak masked')