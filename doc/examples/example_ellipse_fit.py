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
Beam center from ellipse fit
============================

Simulate water scatter and then fit an ellipse to pixels associated to the water ring.

Contributed by Richard Kirian.

Edited by Kosta Karpos.

Note that this example
only works on single-panel detectors.  It could be extended to multi-panel detectors if the need arises, but in that
case it is important that the relative coordinates of all panels is known very well.

"""

# %%
# First we import the needed modules and configure the simulation parameters:

import numpy as np
from reborn import source, detector, const
from reborn.viewers.qtviews import PADView
from reborn.analysis import optimize
from reborn.simulate import solutions

# Configure the detector
detector_shape = (200, 200)
beam_center = np.array([110, 105], dtype=np.double)
pixel_size = 750e-6
detector_distance = 0.1
# Configure x-ray beam
n_photons = 1e12
photon_energy = 8000*const.eV
beam_diameter = 5e-6
# Configure water
water_thickness = 1e-6
water_ring_thresh = 50000

# %%
# Set up the PAD geometry:
pad = detector.PADGeometry(distance=detector_distance, shape=detector_shape, pixel_size=pixel_size)
pad.t_vec[0] = -beam_center[0]*pixel_size
pad.t_vec[1] = -beam_center[1]*pixel_size
# %%
# Set up the x-ray beam:
beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=n_photons*photon_energy)
# %%
# Simulate the water scatter.  We use a built-in example based on experimental lookup table of water scatter.  Note that
# this includes scattering cross-section, polarization, pixel solid angles.
I = pad.reshape(solutions.get_pad_solution_intensity(pad_geometry=pad, beam=beam, thickness=100e-6)[0])
# %%
# Fit an ellipse to pixel coordinates above a water-ring threshold:
mask = I > water_ring_thresh
x, y = np.nonzero(I > water_ring_thresh)
beam_center_fit = optimize.ellipse_center(optimize.fit_ellipse(y, x))
# %%
# Display results:
print('True beam center:', beam_center)
print('Beam center from ellipse fit:', beam_center_fit)
print('Fractional errors:', np.abs(beam_center-beam_center_fit)/beam_center)
pv = PADView(data=I, mask=mask, pad_geometry=pad)
pv.start()
