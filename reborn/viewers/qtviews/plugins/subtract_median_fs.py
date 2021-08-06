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


def plugin(self):
    r""" Plugin for PADView. """
    self.debug('plugin(subtract_median_ss)')
    data = self.get_pad_display_data()
    for i in range(len(data)):
        data[i] -= np.median(data[i], axis=0).reshape((1, data[i].shape[1]))
    self.set_pad_display_data(data, update_display=True, percentiles=(2, 98), colormap='bipolar')
