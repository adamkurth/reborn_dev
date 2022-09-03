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
The PADView GUI
===============

Testing.

Contributed by Richard Kirian.

"""
import numpy as np
from reborn import source, detector, dataframe, simulate, fileio
from reborn.viewers.qtviews import view_pad_data, PADView

# %%
# The PADView class provides a GUI to help visualize diffraction data that consists of multiple PADs.  Let's begin by
# simulating some data to look at:

beam = source.Beam(wavelength=1.5e-10)
pads = detector.cspad_pad_geometry_list(detector_distance=0.1)
dat = simulate.solutions.water_scattering_factor_squared(pads.q_mags(beam=beam))
dat *= pads.polarization_factors(beam=beam)
dat = np.double(np.random.poisson(dat))

# %%
# The simplest way to display the data is as follows:

view_pad_data(data=dat, pad_geometry=pads, beam=beam)

# %%
# Although the above function exists for convenience, a better way is to create an instance of the PADView class.
# This way, you can customize a few things before launching the viewer.  We will call a few  methods that help us
# understand the geometry layout.  The beam origin and axes are shown at the center of the pattern (x: red, y: green,
# z: blue), the PAD names (or numbers) are overlaid on the pads, and the fast-scan directions are indicated by the
# red arrows.

pv = PADView(data=dat, pad_geometry=pads, beam=beam)
pv.show_coordinate_axes()
pv.show_fast_scan_directions()
pv.show_pad_labels()
pv.start()
del pv

# %%
# If you are using iPython, tab completion on `pv.show_` should list various things you can make visible.
#
# PADView has a mechanism to create plugins for processing data.  You can retrieve the list of plugins as follows:

pv = PADView(data=dat, pad_geometry=pads, beam=beam)
for p in pv.list_plugins():
    print(p)

# %%
# To run a plugin, use the run_plugin method.  Here we will correct the polarization factor:

pv.run_plugin('correct_polarization')
pv.start()
del pv

# %%
# Creating a plugin is sometimes very straightforward.  For example, the polarization correction
# plugin used above consists of the following code:
#
# .. code-block:: python
#
#     def plugin(padview):
#         r""" Plugin for PADView. Divides out the beam polarization factor. """
#         data = padview.dataframe.get_processed_data_flat()
#         beam = padview.dataframe.get_beam()
#         geom = padview.dataframe.get_pad_geometry()
#         data /= geom.polarization_factors(beam=beam)
#         padview.set_pad_display_data(data, auto_levels=True, update_display=True)

# %%
# A more advanced plugin can also offer a separate Qt widget.  This is the case for the masking utility in PADView:

pv = PADView(data=dat, pad_geometry=pads, beam=beam)
pv.run_plugin('mask_editor')
pv.start()
del pv

# %%
# It is difficult to demonstrate the mouse clicking that you can do with the mask editor widget, but hopefully there is
# enough information displayed to figure it out.  Note that you can save your mask, but it is in the reborn mask format.
# If you want a different format, you can load the mask in a python script using the
# :func:`load_pad_masks <reborn.detector.load_pad_masks>` function and then save it in whatever format you like.

# %%
# If you want to flip through a series of diffraction patterns, you can create a |FrameGetter| class for your data.
# Here is an example in which we build a |FrameGetter| from a normal list.  First, we make a list of |DataFrame|
# instances, each with different Poisson noise in the simulated data:

dat = simulate.solutions.water_scattering_factor_squared(pads.q_mags(beam=beam))
dat *= pads.polarization_factors(beam=beam)
n_frames = 10
dataframes = []  # Initialize list of dataframes
for i in range(n_frames):
    dat = np.random.poisson(dat)
    dataframes.append(dataframe.DataFrame(raw_data=np.random.poisson(dat), pad_geometry=pads, beam=beam))

# %%
# Next, we create the |FrameGetter| and pass it to |PADView|:

mfg = fileio.getters.ListFrameGetter(dataframes)
pv = PADView(frame_getter=mfg)
pv.start()
del pv

# %%
# As you can see in the bottom left of the display, the frame that is shown is number 1 out of 10.  If you hit the arrow
# keys, the frame will skip forward or backward.

# %%
# We will try to add to this example as needs arise.  Beware that some of the plugins are still under development.  Let
# us know if something is broken or if you have an idea for something that would be nice to have.
