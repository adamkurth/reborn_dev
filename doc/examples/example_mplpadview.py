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
View detector with matplotlib
=============================

Simple example of how to view multiple detector panels with matplotlib.

Contributed by Richard Kirian.

First the imports:
"""

from reborn import detector, data
from reborn.external import crystfel
from reborn.viewers.mplviews import view_pad_data

# %%
# In this example we will load a CrystFEL geometry file:
geomfile = data.cspad_geom_file
pads = crystfel.geometry_file_to_pad_geometry_list(geomfile)
# %%
# Since we don't have diffraction data handy, we will display the solid angles:
dat = pads.solid_angles()
# %%
# Here's the convenience function for viewing a PAD.  It doesn't have many features, but should suffice to
# have a quick look at your geometry:
view_pad_data(pad_data=dat, pad_geometry=pads, pad_numbers=True, show_coords=True, show_scans=True)
# %%
# Note that the `show_scans=True` option displays arrows to indicate the locations of the corners of the 2D numpy
# arrays and the directions of the "fast scan" vectors.  The `show_coords` keyword will display the coordinate
# axes. We follow the convention rgb <=> xyz, so red corresponds to the first
# component of the coordinate vectors (i.e. the "x" component), green corresponds to the second component, etc.  We
# can print the information about the PAD with index 0:
print(pads[0])
# %%
# Now, look at the detector: identify the number 0, and look at the scan arrow (blue arrow with red border).  It points
# downward.  From the output above, we see that the fast-scan vector `fs_vec` points along the positive "y" direction.
# Now notice that this corresponds to the green coordinate vector as expected.  As for the translation of the panel,
# we see that the corner pixel is shifted in the positive "x" direction, and a little bit in the negative "y" direction.
# Indeed, the translation vector `t_vec` printed above agrees with this.

