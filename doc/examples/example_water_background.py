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
Water background scatter
========================

Simple example of how to simulate water background scatter.

Contributed by Richard A. Kirian.
"""

import numpy as np
from reborn import detector, source
from reborn.simulate import solutions
from reborn.viewers.qtviews import PADView

np.random.seed(0)

detector_distance = 0.1
photon_energy = 6000*1.602e-19
beam_diameter = 5e-6
pulse_energy = 5e-3
jet_diameter = 10e-6
jet_temperature = 300

pads = detector.epix10k_pad_geometry_list(detector_distance=detector_distance, binning=10)
beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=pulse_energy)
intensity = solutions.get_pad_solution_intensity(pad_geometry=pads, beam=beam, thickness=jet_diameter,
                                                 liquid='water', temperature=298, poisson=True)
pv = PADView(pad_geometry=pads, data=intensity)
pv.start()
