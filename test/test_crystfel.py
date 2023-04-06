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

from reborn.data import cspad_geom_file
from reborn.external import crystfel


def test_crystfel():
    geom_dict = crystfel.geometry_file_to_pad_geometry_list(cspad_geom_file)
    assert isinstance(geom_dict, list)
    assert len(geom_dict) == 64


def test_02():
    streamfile = crystfel.example_stream_file_path
    fg = crystfel.StreamfileFrameGetter(stream_file=streamfile)
