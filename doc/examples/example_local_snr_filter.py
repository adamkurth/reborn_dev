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
Signal-to-Noise Filter
======================

Simple example of how to transform intensities into a map of local signal-to-noise ratio.

Contributed by Richard A. Kirian.
"""

# %%
# First our imports:

import numpy as np
from reborn import detector, source
from reborn.simulate import solutions
from reborn.viewers.mplviews import view_pad_data
from reborn.analysis.peaks import snr_filter

np.random.seed(0)

# %%
# Some basic configurations of the data simulation (simulated water diffraction pattern).

detector_distance = 0.1
photon_energy = 7000*1.602e-19
beam_diameter = 5e-6
pulse_energy = 0.1e-3
jet_diameter = 3e-6
jet_temperature = 300
binning = 4

# %%
# Standard beam class instance:

beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=pulse_energy)

# %%
# We will use the epix detector layout, but we will bin the pixels to reduce the computational times for this example.

pads = detector.epix10k_pad_geometry_list(detector_distance=detector_distance)
pads = pads.binned(2)

# %%
# Here is the intensity with Poisson noise

intensity = solutions.get_pad_solution_intensity(pad_geometry=pads, beam=beam, thickness=jet_diameter,
                                                 liquid='water', temperature=298, poisson=True)

# %%
# Now we add a giant "cosmic ray" (narrow streak of lit pixels) to PAD number 0:

maxint = np.max(np.concatenate(intensity))
intensity[0][20:21, 5:30] = maxint*10

# %%
# We'll add some random "hot pixels" to PAD number 1:

a = intensity[1]
a.flat[np.random.choice(np.arange(a.size), 50)] = maxint * 10

# %%
# We'll add some random "dead pixels" to PAD number 2:

a = intensity[2]
a.flat[np.random.choice(np.arange(a.size), 50)] = 0

# %%
# Here is the whole detector.  Can you spot the messed-up pixels?  The dead pixels may be hard to spot.

view_pad_data(pad_geometry=pads, pad_data=intensity)

# %%
# Here are 4 of the detectors alone

view_pad_data(pad_geometry=pads[0:4], pad_data=intensity[0:4], pad_numbers=True)

# %%
# Now we seek to find all the messed-up pixels.  First let's correct for the polarization factor and subtract off
# the median water ring value:

intmod = pads.concat_data(intensity)
intmod /= pads.polarization_factors(beam=beam)
radial_profiler = detector.RadialProfiler(pad_geometry=pads, mask=None, beam=beam)
intmod = radial_profiler.subtract_profile(intmod, statistic=np.median)

# %%
# The intensities should be very flat now, except for the outlier pixels:

view_pad_data(pad_geometry=pads, pad_data=intmod)

# %%
# Note that there is a little subtlety that we should pay attention to.  The above intmod array
# (i.e. "modified intensities") has been flattened to a 1D array with all data PADs concatenated.  Look at the shape:

print(intmod.shape)

# %%
# For some operations, such as the SNR filter we are about to apply, we need a list of 2D arrays, because the shape
# of the data matters when doing image analysis.  A convenient way to split up the concatenated data is to use the
# methods of the PADGeometryList class created above:

intmod = pads.split_data(intmod)
print(type(intmod))

# %%
# To identify outliers, we can run a signal-to-noise filter over the data.  Details are in the documentation, but in
# brief, this filter estimates the noise by computing the standard deviation in the pixel values surrounding each pixel.
# The signal is estimated by subtracting off the local signal level.  The result of the filter is a local
# signal-to-noise ratio.

mask = [np.ones_like(d) for d in intmod]
snr = snr_filter(intmod, mask, 0, 5, 12, threshold=6, mask_negative=True, max_iterations=3)
snr = [np.abs(d) for d in snr]
view_pad_data(pad_geometry=pads[0:4], pad_data=snr[0:4], vmax=6)

# %%
# To be clear, one can also use a simple threshold to identify outlier pixels, but the downside to that method is that
# you will need to tune the threshold for each dataset.  With an SNR filter, a threshold of 6 is almost always a good
# number.

# %%
# Finally, let's make a bad-pixel mask:

mask = [np.ones_like(d) for d in snr]
for i in range(len(pads)):
    m = mask[i]
    s = snr[i]
    m[s > 6] = 0

view_pad_data(pad_geometry=pads[0:4], pad_data=mask[0:4], cmap='gray')

