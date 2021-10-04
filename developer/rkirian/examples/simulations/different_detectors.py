import sys
from time import time
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from reborn import source, detector
from reborn.target import molecule, crystal
from reborn.simulate import gas, solutions
from reborn.viewers.qtviews import PADView

# === CONFIGURATION ==============================================================
eV = constants.value('electron volt')
r_e = constants.value('classical electron radius')
k = constants.value('Boltzmann constant')
gas_length = 1
temperature = 300
pressure = 101325
poisson = True
sample_thickness = 4e-6
pulse_energy = 1e-3
photon_energy = 9e3*eV
rayonix_distance = 1.5
epix_distance = 0.3
epix_angle = np.pi/6         # Tilt angle of the epix
helium_partial_pressure = 0  # Helium partial pressure
map_resolution = 0.2e-9  # Minimum resolution for 3D density map
map_oversample = 2  # Oversampling factor for 3D density map
cell_size = 200e-10  # Unit cell size (assume P1, cubic)
pdb_file = '1f88'
protein_concentration = 10  # Protein concentration in mg/ml = kg/m^3
hit_frac = 0.5  # Hit fraction
freq = 120  # XFEL frequency
runtime = 12*60*60  # Run time in seconds
random_seed = 120  # Seed for random number generator (choose None to make it random)
cl_double_precision = True
cl_group_size = 32
# ======================================================================================

if random_seed is not None:
    np.random.seed(random_seed)
beam = source.Beam(photon_energy=photon_energy, pulse_energy=pulse_energy)
pads_rayonix, mask = detector.rayonix_mx340_xfel_pad_geometry_list(detector_distance=rayonix_distance, return_mask=True)
pads_epix = detector.epix10k_pad_geometry_list(detector_distance=0)
R = np.array([[np.cos(epix_angle), 0, np.sin(epix_angle)], [0, 1, 0], [-np.sin(epix_angle), 0, np.cos(epix_angle)]])
for p in pads_epix:
    p.fs_vec = np.dot(p.fs_vec, R.T)
    p.ss_vec = np.dot(p.ss_vec, R.T)
    p.t_vec = np.dot(p.t_vec, R.T)
    p.t_vec += np.dot(np.array([0, 0, epix_distance]), R.T)
pads = detector.PADGeometryList(pads_rayonix + pads_epix)
mask = pads.concat_data([mask] + pads_epix.split_data(pads_epix.ones()))
q_mags = pads.q_mags(beam=beam)
sa = pads.solid_angles()
pol = pads.polarization_factors(beam=beam)
J0 = beam.photon_number_fluence
water_intensity = solutions.get_pad_solution_intensity(beam=beam, pad_geometry=pads,
                                                       thickness=sample_thickness, liquid='water', poisson=poisson)
water_intensity = pads.concat_data(water_intensity)
# Build a rhodopsin molecule
rhod_mol = crystal.CrystalStructure(pdb_file).molecule
q_profile = np.linspace(q_mags.min(), q_mags.max(), 100)
# print('rhod')
# rhod_profile = gas.isotropic_gas_intensity_profile(molecule=rhod_mol, beam=beam, q_mags=q_profile)
he_profile = gas.isotropic_gas_intensity_profile(molecule='He', beam=beam, q_mags=q_profile)
n_gas_molecules = np.pi * (beam.diameter_fwhm/2)**2 * gas_length * pressure / k / temperature
air_profile = gas.air_intensity_profile(q_mags=q_profile, beam=beam)
gas_profile = air_profile*(1-helium_partial_pressure) + he_profile*helium_partial_pressure
gas_intensity = n_gas_molecules * J0 * sa * pol * r_e**2 * np.interp(q_mags, q_profile, gas_profile)

if poisson:
    gas_intensity = np.random.poisson(gas_intensity).astype(np.double)

if False:
    water_intensity /= pads.solid_angles() * pads.polarization_factors(beam=beam)

total_intensity = gas_intensity + water_intensity
print(total_intensity[0:10])
pv = PADView(pad_geometry=pads, raw_data=total_intensity, mask_data=mask)
pv.start()
