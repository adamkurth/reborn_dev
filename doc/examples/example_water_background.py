r"""
Water background scatter
========================

Simple example of how to simulate water background scatter.

Contributed by Richard A. Kirian.
"""

import numpy as np
from reborn import detector, source
from reborn.simulate import solutions
from scipy import constants as const
from reborn.viewers.mplviews import view_pad_data

np.random.seed(0)
r_e = const.value('classical electron radius')
eV = const.value('electron volt')
concat = detector.concat_pad_data

detector_distance = 0.1
photon_energy = 6000*eV
beam_diameter = 5e-6
pulse_energy = 5e-3
jet_diameter = 10e-6
jet_temperature = 300

pads = detector.epix10k_pad_geometry_list(detector_distance=detector_distance)
beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=pulse_energy)
I = solutions.get_pad_solution_intensity(pad_geometry=pads, beam=beam, thickness=jet_diameter,
                                         liquid='water', temperature=298, poisson=True)
view_pad_data(pad_geometry=pads, pad_data=I)

# n_water_molecules = jet_diameter * beam.diameter_fwhm**2 * solutions.water_number_density()
# q_mags = pads.q_mags(beam)
# J = beam.photon_number_fluence
# P = concat([p.polarization_factors(beam=beam) for p in pads])
# SA = concat([p.solid_angles() for p in pads])
# F = solutions.get_water_profile(q_mags, temperature=jet_temperature)
# F2 = F**2*n_water_molecules
# I = r_e**2 * J * P * SA * F2
# I = np.random.poisson(I)
# I = detector.split_pad_data(pads, I)
# view_pad_data(pad_geometry=pads, pad_data=I)
