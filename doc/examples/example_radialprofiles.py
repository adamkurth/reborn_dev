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
======================

Demonstration of how to use the |RadialProfiler| class.

Contributed by Konstantinos Karpos

Updated by Richard A. Kirian

"""

# %%
# Radial profiles are an essential step in solution and powder diffraction analysis. This example
# goes over a few basic examples on how to use the built-in |RadialProfiler| class within |reborn|.
#
# We start with the usual import block:

import time
import numpy as np
import matplotlib.pyplot as plt
from reborn import source, detector, const
from reborn.simulate import solutions
from reborn.viewers.qtviews import PADView

# %%
# As is the case in every diffraction analysis or simulation, the geometry of your detector is essential. For this
# example we use a built-in |PADGeometryList| for a Rayonix detector. We also create |Beam| class instance.  Note
# that the detector distance is in SI units like everything else in reborn.

geom = detector.rayonix_mx340_xfel_pad_geometry_list(detector_distance=0.2, binning=10)
beam = source.Beam(photon_energy=8000*const.eV)

# %%
# Simulate a water pattern and make sure it looks reasonable:

water_pattern = solutions.get_pad_solution_intensity(pad_geometry=geom, beam=beam, thickness=5e-6, liquid='water',
                                                     temperature=293.15, poisson=False)
pv = PADView(pad_geometry=geom, data=water_pattern)
pv.start()

# %%
# Before calculating radial profiles, we create a |RadialProfiler| class instance:

profiler = detector.RadialProfiler(beam=beam, pad_geometry=geom, n_bins=100, q_range=np.array([0, 3.2]) * 1e10)

# %%
# The reason for the class is that it allows for a tidy way to maintain a cache of pre-computed indices that speed up
# our calculations.  It also maintains the information about bin boundaries, the detector mask, and so on.
#
# The |RadialProfiler| calculates the usual weighted mean and standard deviation using python code that wraps around
# optimized fortran code.  It also has the capability to compute arbitrary statistics using any function handle that
# operates on a numpy array.
#
# The fastest way to get radial profiles is to use the ``quickstats`` method, which calculates everything at once.  The
# first step in the underlying code computes the sums
#
# .. math::
#
#     S &= \sum_{i=1}^N w_i I_i \\
#     S2 &= \sum_{i=1}^N w_i I_i^2 \\
#     W &= \sum_{i=1}^N w_i
#
# where :math:`w_i` are weights (equal to the binary mask by default), and :math:`I_i` are the intensity values that
# lie within a particular :math:`q` bin.  From these sums, the weighted average and standard deviation are computed:
#
# .. math::
#
#     \langle I \rangle &= S / W \\
#     \sigma &= \sqrt{\langle I^2 \rangle - \langle I \rangle^2}
#
# For bins in which :math:`W = 0`, the values are set to zero by default.  In some cases you may wish to specify the
# weights as something other than the binary pixel mask; for example, you may use the polarization factor and pixel
# solid angles since floating point arrays are allowed.

t = time.time()
stats = profiler.quickstats(water_pattern)
print(f"{(time.time()-t)*1000} milliseconds to calculate the following:")
print(stats.keys())
sum_radial = stats['sum']
sum_squared_radial = stats['sum2']
counts_radial = stats['weight_sum']
mean_radial = stats['mean']
standard_deviation_radial = stats['sdev']

# %%
# Finally, to calculate an arbitrary statistic, the ``get_profile_statistic`` method is available.  This is presently
# the way that we calculate median profiles.  You may of course calculate a mean profile in this way, but it is about
# 10-fold slower than the ``quickstats`` method.

t = time.time()
slow_mean_radial = profiler.get_profile_statistic(water_pattern, statistic=np.mean)
print(f"{(time.time()-t)*1000} milliseconds to calculate the mean (slowly).")
t = time.time()
median_radial = profiler.get_profile_statistic(water_pattern, statistic=np.median)
print(f"{(time.time()-t)*1000} milliseconds to calculate the median.")

# %%
# Here are some plots of the above results:

qrange = profiler.q_bin_centers*1e-10
fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

ax[0][0].plot(qrange, mean_radial)
ax[0][0].set_title("Mean Radial")
ax[1][0].plot(qrange, standard_deviation_radial)
ax[1][0].set_title("Standard Deviation Radial")
ax[0][1].plot(qrange, sum_radial)
ax[0][1].set_title("Sum Radial")
ax[1][1].plot(qrange, counts_radial)
ax[1][1].set_title("Counts Radial")
ax[0][2].plot(qrange, slow_mean_radial)
ax[0][2].set_title("Mean Radial (slow method)")
ax[1][2].plot(qrange, median_radial)
ax[1][2].set_title("Median Radial")
fig.text(0.5, 0.04, r'q=4$\pi\sin\theta / \lambda$ [$\AA$]', ha='center')

plt.show()
