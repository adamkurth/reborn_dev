""" Example """
import numpy as np
from scipy import constants
from reborn import source, detector
from reborn.target import molecule, crystal, atoms
from reborn.simulate import gas, solutions
from reborn.viewers.qtviews import PADView
eV = constants.value('electron volt')
r_e = constants.value('classical electron radius')
k = constants.value('Boltzmann constant')
gas_length = 0.3
temperature = 300
pressure = 101325
poisson = True
water_thickness = 10e-6
beam = source.Beam(photon_energy=9e3*eV, pulse_energy=1e-3)
pads = detector.mpccd_pad_geometry_list(detector_distance=0.08)
q_mags = pads.q_mags(beam=beam)
h2o_intensity = solutions.get_pad_solution_intensity(beam=beam, pad_geometry=pads,
                                           thickness=water_thickness, liquid='water', poisson=poisson)
h2o_intensity = pads.concat_data(h2o_intensity)
he_mol = molecule.Molecule(coordinates=np.array([[0, 0, 0]]), atomic_numbers=[2, 2])
q_profile = np.linspace(q_mags.min(), q_mags.max(), 100)
he_profile = gas.isotropic_gas_intensity_profile(molecule=he_mol, beam=beam, q_mags=q_profile)
n_gas_molecules = np.pi * (beam.diameter_fwhm/2)**2 * gas_length * pressure / k / temperature
gas_intensity = beam.photon_number_fluence * n_gas_molecules * r_e**2 * np.interp(q_mags, q_profile, he_profile) * \
                pads.solid_angles() * pads.polarization_factors(beam=beam)
if poisson:
    gas_intensity = np.random.poisson(gas_intensity).astype(np.double)
total_intensity = h2o_intensity + gas_intensity
pv = PADView(pad_geometry=pads, raw_data=total_intensity)
pv.add_rings(q_mags=[2.1e10])
pv.start()
