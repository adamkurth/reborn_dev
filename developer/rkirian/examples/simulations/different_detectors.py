import numpy as np
from reborn import source, detector
from reborn.simulate.solutions import get_pad_solution_intensity
from reborn.target import gas, molecule
from reborn.viewers.qtviews import PADView
from scipy import constants
eV = constants.value('electron volt')
r_e = constants.value('classical electron radius')
k = constants.value('Boltzmann constant')
gas_length = 1
temperature = 300
pressure = 101325

beam = source.Beam(photon_energy=9e3*eV, pulse_energy=1e-3)
pads1 = detector.tiled_pad_geometry_list(pixel_size=8.9e-5, pad_shape=(3840, 3840), tiling_shape=(1, 1), distance=1.5)
pads2 = detector.epix10k_pad_geometry_list(detector_distance=0)
theta = np.pi/4
R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
for p in pads2:
    p.fs_vec = np.dot(p.fs_vec, R.T)
    p.ss_vec = np.dot(p.ss_vec, R.T)
    p.t_vec = np.dot(p.t_vec, R.T)
    p.t_vec[0] += 100e-3
    p.t_vec[2] += 0.2
pads = detector.PADGeometryList(pads1 + pads2)
q_mags = pads.q_mags(beam=beam)
h2o_intensity = get_pad_solution_intensity(beam=beam, pad_geometry=pads, thickness=60e-6, liquid='water', poisson=False)
h2o_intensity = pads.concat_data(h2o_intensity)
h2o_intensity /= pads.solid_angles() * pads.polarization_factors(beam=beam)
# Build an N2 molecule
n2_mol = molecule.Molecule(coordinates=np.array([[0, 0, 0], [0, 0, 1.07e-10]]), atomic_numbers=[7, 7])
o2_mol = molecule.Molecule(coordinates=np.array([[0, 0, 0], [0, 0, 1.21e-10]]), atomic_numbers=[8, 8])
he_mol = molecule.Molecule(coordinates=np.array([[0, 0, 0]]), atomic_numbers=[2, 2])
q_profile = np.linspace(q_mags.min(), q_mags.max(), 1000)
n2_profile = gas.isotropic_gas_intensity_profile(molecule=n2_mol, beam=beam, q_mags=q_profile)
o2_profile = gas.isotropic_gas_intensity_profile(molecule=o2_mol, beam=beam, q_mags=q_profile)
he_profile = gas.isotropic_gas_intensity_profile(molecule=he_mol, beam=beam, q_mags=q_profile)
n_gas_molecules = np.pi * (beam.diameter_fwhm/2)**2 * gas_length * pressure / k / temperature
n2_partial = 0.79  # N2 and O2 partial pressures are combined to form "air"
o2_partial = 0.21
air_profile = n2_profile*n2_partial + o2_profile*o2_partial
he_partial = 0  # He partial pressure combined with air to form "gas"
gas_profile = air_profile*(1-he_partial) + he_profile*he_partial
gas_intensity = n_gas_molecules * r_e**2 * np.interp(q_mags, q_profile, gas_profile)
total_intensity = gas_intensity + h2o_intensity
total_intensity = h2o_intensity / gas_intensity
pv = PADView(pad_geometry=pads, raw_data=total_intensity)
pv.start()