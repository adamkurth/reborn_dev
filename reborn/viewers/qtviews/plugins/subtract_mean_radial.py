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

from time import time
import reborn


class Plugin():
    def __init__(self, padview):
        self.padview = padview
        self.profiler = reborn.detector.RadialProfiler(pad_geometry=padview.dataframe.get_pad_geometry(),
                                                       beam=padview.dataframe.get_beam())
        self.action()
    def action(self):
        padview = self.padview
        padview.debug('Calculating mean profile...', 1)
        t = time()
        data = self.profiler.subtract_profile(padview.get_pad_display_data(),
                                              mask=padview.dataframe.get_mask_list(), statistic='mean')
        padview.debug('Done (%g seconds)' % (time()-t), 1)
        padview.set_pad_display_data(data, update_display=True)
