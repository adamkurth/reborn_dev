r"""
View detector with matplotlib
=============================

Simple example of how to view multiple detector panels with matplotlib.

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
view_pad_data(pad_data=q_mags, pad_geometry=pads, pad_numbers=True, circle_radii=10)
