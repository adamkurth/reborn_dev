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

from reborn.external import crystfel
from reborn.data import cspad_geom_file
from reborn.viewers.mplviews import view_pad_data

# %%
# In this example we will load a CrystFEL geometry file.  We use a file that is included with reborn as an example:
pads = crystfel.geometry_file_to_pad_geometry_list(cspad_geom_file)
# %%
# We will display the q-vector magnitudes, so we make a list of 2D arrays with those values.
q_mags = [p.reshape(p.q_mags(wavelength=1.5e-10, beam_vec=[0, 0, 1])) for p in pads]
# %%
# Here's the convenience function for viewing a PAD.  It doesn't have many features at this time, but should suffice to
# have a quick look at your geometry.  If you want something more elaborate, you should look at the source code and
# copy/modify as needed.
view_pad_data(pad_data=q_mags, pad_geometry=pads, pad_numbers=True, show_coords=True, show_scans=True)
# %%
# Now let's do a little sanity check to make sure that we understand what we are looking at.
# Note that the `show_scans` keyword will indicate the locations of the corners of the 2D numpy arrays, and also the
# directions of the "fast scan" vectors.  The `show_coords` keyword will display the coordinate axes, which are
# indicated by the red and green arrows.  We follow the convention rgb <=> xyz, so red corresponds to the first
# component of the coordinate vectors (i.e. the "x" component), green corresponds to the second component, etc.  We
# can print the information about the PAD with index 0:
print(pads[0])
# %%
# Now, look at the detector: identify the number 0, and look at the scan arrow (blue arrow with red border).  It points
# downward.  From the output above, we see that the fast-scan vector `fs_vec` points along the positive "y" direction.
# Now notice that this corresponds to the green coordinate vector as expected.  As for the translation of the panel,
# we see that the corner pixel is shifted in the positive "x" direction, and a little bit in the negative "y" direction.
# Indeed, the translation vector `t_vec` printed above agrees with this.
#
# It is important to note that the units in this display are screwed up.  We are presently using "pixel units" for our
# coordinate system.  This is a very bad idea, because a multi-panel PAD could have more than one pixel size.  Moreover,
# if multiple detectors are situated at different distances (e.g. a small-angle and wide-angle detector), then this
# display is going to have ridiculous detector overlaps.  Hopefully the coordinates are fixed before many people read
# this, but that requires that Rick (or someone else) comes to an understanding of 2D Affine transforms in matplotlib,
# and then thinks hard about how the the 3D detector panels *should* be projected onto a 2D plane.  If multiple
# detectors are going to look reasonable, the point of view should be the location of the sample.
