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

from reborn.source import Beam
import os

def test_json():
    fname = 'test.beam'
    b1 = Beam()
    b1.save_json(fname)
    b2 = Beam()
    b2.load_json(fname)
    os.remove(fname)
    d1 = b1.to_dict()
    d2 = b2.to_dict()
    for k in list(d1.keys()):
        assert(d1[k] == d2[k])
