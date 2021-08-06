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

import pkg_resources

# PDB files
lysozyme_pdb_file = pkg_resources.resource_filename('reborn', 'data/pdb/2LYZ.pdb')
psi_pdb_file = pkg_resources.resource_filename('reborn', 'data/pdb/1jb0.pdb')

# CrystFEL geom files
pnccd_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/pnccd_front.geom')
cspad_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/cspad.geom')
