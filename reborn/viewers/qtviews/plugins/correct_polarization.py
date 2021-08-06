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

from reborn import detector
def plugin(padview):
    r""" Plugin for PADView. """
    data = detector.concat_pad_data(padview.get_pad_display_data())
    data /= detector.concat_pad_data([p.polarization_factors(beam=padview.dataframe.get_beam()) for p in
                                      padview.dataframe.get_pad_geometry()])
    padview.set_pad_display_data(data, auto_levels=True, update_display=True)
