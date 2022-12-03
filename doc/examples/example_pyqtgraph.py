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
# A totally incomplete summary of plot settings:

plot = pg.plot()
legend = plot.addLegend()
x = np.linspace(0, 2*np.pi, 40)

y1 = np.cos(x)
pen1 = pg.mkPen(width=4.5, color='r', style=pg.QtCore.Qt.DashLine)
symbol1 = dict(symbol='o', symbolBrush=pg.mkBrush(0, 127, 0, 200), symbolSize=np.arange(40), symbolPen='b')
plot.plot(x, y1, pen=pen1, **symbol1, name='Cos(x)')

y2 = np.sin(x)
plot.plot(x, y2, name='Sin(x)')

plot.setXRange(0, 2 * np.pi, padding=0.1)
plot.setLabel('bottom', "x")
plot.setLabel('left', "y")
plot.setTitle('A Plot!')

legend.setBrush([0, 0, 0, 200])
legend.setPen(plot.getAxis("left").pen())
legend.setPos(100, 0)

# %%
# Formatting fonts:

font = plot.getAxis("left").label.font()
font.setFamily("Serif")
font.setPixelSize(20)
plot.getAxis("left").label.setFont(font)
plot.getAxis("left").setTickFont(font)
plot.getAxis("bottom").label.setFont(font)
plot.getAxis("bottom").setTickFont(font)
plot.getPlotItem().titleLabel.item.setFont(font)
plot.show()

# %%
# If you run a script and find that the plot window appears for an instant and vanishes,
# you need to execute a Qt app:

pg.mkQApp().exec_()

# %%
# If you want a plot to update within a for loop:

plot = pg.plot()
app = pg.mkQApp()
for i in range(2):
    plot.plot(np.random.rand(10), clear=True)
    app.processEvents()
    time.sleep(0.1)

# %%
# If you want to put an image within a plot axis:

dat2d = np.random.rand(100, 150)
plot = pg.PlotItem()
imv = pg.ImageView(view=plot)
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
