r"""
Water background scatter
========================

Simple example of how to simulate water background scatter.

Contributed by Richard A. Kirian.
"""

import numpy as np
from reborn import detector, source
from reborn.simulate import solutions
from reborn.viewers.mplviews import view_pad_data

np.random.seed(0)

detector_distance = 0.1
photon_energy = 6000*1.602e-19
beam_diameter = 5e-6
pulse_energy = 5e-3
jet_diameter = 10e-6
jet_temperature = 300

pads = detector.epix10k_pad_geometry_list(detector_distance=detector_distance)
beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=pulse_energy)
intensity = solutions.get_pad_solution_intensity(pad_geometry=pads, beam=beam, thickness=jet_diameter,
                                                 liquid='water', temperature=298, poisson=True)
view_pad_data(pad_geometry=pads, pad_data=intensity)
