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

r"""
This is the documentation for the reborn package.
"""
import os
import tempfile
from . import utils
from . import source
from . import detector
from . import dataframe
from . import simulate
from . import target
from . import fileio
from . import misc
from . import fortran
from . import const

temp_dir = os.path.join(tempfile.gettempdir(), 'reborn')
