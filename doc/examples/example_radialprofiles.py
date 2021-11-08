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
.. _example_radialprofiles:

Making Radial Profiles
=========================================

Demonstration of how to use `reborn`'s RadialProfiler class.

Contributed by Konstantinos Karpos

"""

# %%
# Overview
# --------
#
# Radial profiles are an essential step in X-Ray diffraction data analysis. This example 
# goes over a few basic examples on how to use the built in `RadialProfiler` class within `reborn`. 


# %%
# Initializing the RadialProfiler class
# -------------------------------------
#
# As is the case in every diffraction analysis or simulation, the geometry of your detector is required before
# any further steps are taken. For the sake of simplicity, we can use a built in PADGeometry. We should also initialize
# beam class while we're setting things up.

import reborn
from scipy import constants

eV = constants.value('electron volt')  # J

# Note that detector_geometry is in terms of meters
rayonix_geom = reborn.detector.rayonix_mx340_xfel_pad_geometry_list(detector_distance=0.1)
beam = reborn.source.Beam(photon_energy=10000*eV)

# %%
# Next, we should simulate a water pattern for use throughout the example. 

from reborn.simulate import solutions

water_pattern = solutions.get_pad_solution_intensity(pad_geometry=rayonix_geom,
                                                            beam=beam,
                                                            thickness=5e-6,
                                                            liquid='water',
                                                            temperature=293.15,
                                                            poisson=False)

# %%
# Let's display the pattern so we can get an idea of what the pattern looks like.
from reborn.viewers.mplviews import view_pad_data

view_pad_data(pad_geometry=rayonix_geom, pad_data=water_pattern)

# %%
# Now that we have our basic setup, we can calculate our radial profiles.
# Before doing so, we'll need to initialize the RadialProfiler class.

from reborn.detector import RadialProfiler
import numpy as np

profiler = RadialProfiler(beam=beam,
                            pad_geometry=rayonix_geom,
                            n_bins=1000, # number of histogram bins, 1000 is usually our default
                            q_range=np.array([0, 3.2]) * 1e10, # reborn is SI, so the qrange is in units of meters
                            mask=None)


# %%
# Basic Usage
# -----------
#
# This class instance now handles all the complicated steps in setting up your geometries and your binning.
# Now, a radial profiler allows you to calculate any statistic on a given q-bin. Be it the mean, variance, standard
# deviation, etc. Examples of the built-in functions are shown below, with the final example covering how 
# to use your own statistic.

import pylab as plt

# calculating the mean of each q bin
mean_radial = profiler.get_mean_profile(water_pattern)
# get the standard deviation in each bin
standard_deviation_radial = profiler.get_sdev_profile(water_pattern)
# and now the sum
sum_radial = profiler.get_sum_profile(water_pattern)
# continuing on with the photon counts per q bin
counts_radial = profiler.get_counts_profile(water_pattern)

# Finally, to calculate any statistic, the following function is available. 
# Note that for the sake of the example, np.var() and np.median are used. Any function can be used here.
statistic_radial_1 = profiler.get_profile_statistic(water_pattern, statistic=np.var)
statistic_radial_2 = profiler.get_profile_statistic(water_pattern, statistic=np.median)

# let's plot each statistic for the fun of it.
qrange = np.linspace(0, 3.2, 1000) # converting from bins to q
fig, ax = plt.subplots(2, 3, figsize=(12,6), sharex=True)

ax[0][0].plot(qrange, mean_radial)
ax[0][0].set_title("Mean Radial")
ax[1][0].plot(qrange, standard_deviation_radial)
ax[1][0].set_title("Standard Deviation Radial")
ax[0][1].plot(qrange, sum_radial)
ax[0][1].set_title("Sum Radial")
ax[1][1].plot(qrange, counts_radial)
ax[1][1].set_title("Counts Radial")
ax[0][2].plot(qrange, statistic_radial_1)
ax[0][2].set_title("Variance Radial")
ax[1][2].plot(qrange, statistic_radial_2)
ax[1][2].set_title("Median Radial")
fig.text(0.5, 0.04, r'q=4$\pi\sin\theta / \lambda$ [$\AA$]', ha='center')

plt.show()














