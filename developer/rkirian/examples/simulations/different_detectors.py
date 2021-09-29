import sys
from time import time
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from reborn import source, detector
from reborn.target import molecule, crystal
from reborn.simulate import gas, solutions
from reborn.viewers.qtviews import PADView
eV = constants.value('electron volt')
r_e = constants.value('classical electron radius')
k = constants.value('Boltzmann constant')
gas_length = 1
temperature = 300
pressure = 101325
poisson = False
water_thickness = 4e-6
beam = source.Beam(photon_energy=9e3*eV, pulse_energy=1e-3)
pads1 = detector.tiled_pad_geometry_list(pixel_size=8.9e-5, pad_shape=(3840, 3840), tiling_shape=(1, 1), distance=1.5)
pads2 = detector.epix10k_pad_geometry_list(detector_distance=0)
theta = np.pi/4
R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
for p in pads2:
    p.fs_vec = np.dot(p.fs_vec, R.T)
    p.ss_vec = np.dot(p.ss_vec, R.T)
    p.t_vec = np.dot(p.t_vec, R.T)
    p.t_vec[0] += 30e-3
    p.t_vec[2] += 0.2
pads = detector.PADGeometryList(pads1 + pads2)
q_mags = pads.q_mags(beam=beam)
h2o_intensity = solutions.get_pad_solution_intensity(beam=beam, pad_geometry=pads,
                                           thickness=water_thickness, liquid='water', poisson=poisson)
h2o_intensity = pads.concat_data(h2o_intensity)
h2o_intensity /= pads.solid_angles() * pads.polarization_factors(beam=beam)
# Build a rhodopsin molecule
rhod_mol = crystal.CrystalStructure('1f88').molecule
# Build N2, O2, and he molecules
n2_mol = molecule.Molecule(coordinates=np.array([[0, 0, 0], [0, 0, 1.07e-10]]), atomic_numbers=[7, 7])
o2_mol = molecule.Molecule(coordinates=np.array([[0, 0, 0], [0, 0, 1.21e-10]]), atomic_numbers=[8, 8])
he_mol = molecule.Molecule(coordinates=np.array([[0, 0, 0]]), atomic_numbers=[2, 2])
q_profile = np.linspace(q_mags.min(), q_mags.max(), 100)
# print('rhod')
# rhod_profile = gas.isotropic_gas_intensity_profile(molecule=rhod_mol, beam=beam, q_mags=q_profile)
print('n2')
n2_profile = gas.isotropic_gas_intensity_profile(molecule=n2_mol, beam=beam, q_mags=q_profile)
# plt.plot(n2_profile)
# plt.show()
# sys.exit()
o2_profile = gas.isotropic_gas_intensity_profile(molecule=o2_mol, beam=beam, q_mags=q_profile)
he_profile = gas.isotropic_gas_intensity_profile(molecule=he_mol, beam=beam, q_mags=q_profile)
n_gas_molecules = np.pi * (beam.diameter_fwhm/2)**2 * gas_length * pressure / k / temperature
n2_partial = 0.79  # N2 and O2 partial pressures are combined to form "air"
o2_partial = 0.21
air_profile = n2_profile*n2_partial + o2_profile*o2_partial
he_partial = 0  # He partial pressure combined with air to form "gas"
gas_profile = air_profile*(1-he_partial) + he_profile*he_partial
gas_intensity = n_gas_molecules * r_e**2 * np.interp(q_mags, q_profile, gas_profile)
gas_intensity *= beam.photon_number_fluence
if poisson:
    gas_intensity = np.random.poisson(gas_intensity).astype(np.double)
# print(n_gas_molecules)
# total_intensity = gas_intensity + h2o_intensity
total_intensity = h2o_intensity / gas_intensity
print(total_intensity[0:5])
pv = PADView(pad_geometry=pads, raw_data=total_intensity)
pv.start()
