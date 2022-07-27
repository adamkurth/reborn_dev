import numpy as np
from reborn import source, detector, const
from reborn.target import atoms
from reborn.simulate import solutions
from reborn.simulate.form_factors import sphere_form_factor
from reborn.viewers.qtviews import view_pad_data
photon_energy = 7000*const.eV
detector_distance = 0.58
pulse_energy = 10000e-3
drop_diameter = 100e-9
beam_diameter = 0.5e-6
beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=pulse_energy)
pads = detector.jungfrau4m_pad_geometry_list(detector_distance=detector_distance)
f2p = pads.f2phot(beam)
q_mags = pads.q_mags(beam=beam)
rho = atoms.xraylib_scattering_density(compound='H2O', density=1000, beam=beam)
# rho should be close to the electron number density of water:
# 10 electrons * (1000 kg/m^3) / (18 g/mol / 6.022e23) = 3.35e29 / m^3
amps = rho*sphere_form_factor(radius=drop_diameter/2, q_mags=q_mags)
intensities = np.random.poisson(np.abs(amps)**2*f2p)
view_pad_data(pad_geometry=pads, data=intensities, show=True, title='%3.0f nm drop' % (drop_diameter*1e9))
intensities = solutions.get_pad_solution_intensity(pad_geometry=pads, beam=beam, thickness=drop_diameter,
                                                 liquid='water', poisson=True)

#view_pad_data(pad_geometry=pads, data=intensities, title='%3.0f nm sheet' % (drop_diameter*1e9))
