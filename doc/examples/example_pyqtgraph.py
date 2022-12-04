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
.. _plot_pyqtgraph:

Tips on using pyqtgraph
=======================

Many tasks in pyqtgraph require the use of undocumented parameters or methods, or they involve
the use of Qt stuff that most people are unfamiliar with.  This example is meant to help with
that.

Contributed by Richard A. Kirian
"""

import time
import numpy as np
import pyqtgraph as pg
np.random.seed(1)

# %%
# If you run a script and find that the plot window appears for an instant and vanishes,
# you need to execute a Qt app to block further executions

dat2d = np.random.rand(25, 15)
im = pg.image(dat2d)
# pg.mkQApp().exec_()

# %%
# If you want a plot to update within a for loop so you can watch something change:

p = pg.plot()
app = pg.mkQApp()
for i in range(2):
    p.plot(np.random.rand(10), clear=True)
    app.processEvents()
    time.sleep(0.5)

# %%
# If you want your image to have a plot axis:

p = pg.PlotItem()
imv = pg.ImageView(view=p)
imv.setImage(dat2d)
imv.show()

# %%
# Make the y-axis point upwards:

imv.view.invertY(False)

# %%
# Set the range of the plot axes (with bounds corresponding to pixel centers):

fs_lims = [0, 14]
ss_lims = [1, 250]
fs_scale = (fs_lims[1] - fs_lims[0])/(dat2d.shape[1]-1)
ss_scale = (ss_lims[1] - ss_lims[0])/(dat2d.shape[0]-1)
position = (ss_lims[0]-0.5*ss_scale, fs_lims[0]-0.5*fs_scale)
imv.setImage(dat2d, pos=position, scale=(ss_scale, fs_scale))

# %%
# Add title and axes labels:

imv.view.setLabel('left', 'Fast scan axis')
imv.view.setLabel('bottom', 'Slow scan axis')
imv.view.setTitle('Example Title')

# %%
# Set the levels of the colormap

imv.setLevels(0.5, 1)

# %%
# Unlock the aspect ratio so the image can be stretched

imv.view.setAspectLocked(False)

# %%
# Change the colormap

colormaps = ("thermal", "flame", "yellowy", "bipolar", "spectrum", "cyclic", "greyclip", "grey", "viridis", "inferno",
"plasma", "magma")
# imv.setPredefinedGradient(colormaps[0])

# %%
# Remove the buttons that you probably don't use

imv.ui.roiBtn.hide()
imv.ui.menuBtn.hide()

# %%
# Remove the histogram if you don't want it

# imv.ui.histogram.hide()

# %%
# Use a colormap from matplotlib

from matplotlib import cm
colormap = cm.get_cmap("magma")
colormap._init()
lut = (colormap._lut * 255).view(np.ndarray)[:colormap.N]
imv.ui.histogram.gradient.setColorMap(pg.ColorMap(np.arange(0, 1, 1/256), lut))

# %%
# Remove the big triangular histogram ticks

for tick in imv.ui.histogram.gradient.ticks:
      tick.hide()

# %%
# Make the image visible

imv.show()

pg.mkQApp().exec_()
