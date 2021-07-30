import numpy as np
from reborn import detector
from reborn.source import Beam
from reborn.target import atoms
from reborn.simulate import solutions
from reborn.simulate.form_factors import sphere_form_factor
from reborn.viewers.qtviews import view_pad_data
import scipy.constants as const
np.random.seed(0)
eV = const.value('electron volt')
r_e = const.value('classical electron radius')
h = const.h
c = const.c
photon_energy = 7000*eV
detector_distance = 2.4
pulse_energy = 0.5e-3
drop_diameter = 100e-9
beam_diameter = 0.5e-6
beam = Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=pulse_energy)
fluence = beam.photon_number_fluence
pads = detector.cspad_2x2_pad_geometry_list(detector_distance=detector_distance)
q_mags = pads.q_mags(beam=beam)
solid_angles = pads.solid_angles()
polarization_factors = pads.polarization_factors(beam=beam)
rho = atoms.xraylib_scattering_density(compound='H2O', density=1000, beam=beam)
# rho should be close to the electron number density of water:
# 10 electrons * (1000 kg/m^3) / (18 g/mol / 6.022e23) = 3.35e29 / m^3
# print(rho, 3.35e29)
amps = r_e*rho*sphere_form_factor(radius=drop_diameter/2, q_mags=q_mags)
intensities = np.random.poisson(np.abs(amps)**2*solid_angles*polarization_factors*fluence)
view_pad_data(pad_geometry=pads, pad_data=intensities, show=False, title='%3.0f nm drop' % (drop_diameter*1e9))
intensities = solutions.get_pad_solution_intensity(pad_geometry=pads, beam=beam, thickness=drop_diameter,
                                                 liquid='water', poisson=True)
view_pad_data(pad_geometry=pads, pad_data=intensities, title='%3.0f nm sheet' % (drop_diameter*1e9))
