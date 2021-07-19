import numpy as np
import matplotlib.pyplot as plt
from reborn import detector, source
import reborn.target.crystal as crystal
import reborn.simulate.clcore as simcore

# INPUT PARAMETERS ================================
n_pixels = 1000
pixel_size = 100e-6
detector_distance = 0.05
photon_energies = np.linspace(283, 286, 10)*1.602e-19
pulse_energy = 32e-9
beam_diameter = 1e-6
sample_thickness = 100e-9
n_frames = 1000
pdb_id = '1JB0'
concentration = 10  # mg/ml = kg/m^3
# ==================================================

cryst = crystal.CrystalStructure(pdb_id)
pad_geometry = detector.PADGeometry(shape=(n_pixels, n_pixels), pixel_size=pixel_size, distance=detector_distance)
sample_volume = sample_thickness * np.pi * (beam_diameter / 2) ** 2
protein_number_density = concentration/cryst.molecule.get_molecular_weight()
n_proteins = sample_volume*protein_number_density

print(np.unique(cryst.molecule.atomic_symbols))

for photon_energy in photon_energies:

    beam = source.Beam(photon_energy=photon_energy, pulse_energy=pulse_energy, diameter_fwhm=beam_diameter)
    q_vecs = pad_geometry.q_vecs(beam=beam)
    r_vecs = cryst.molecule.coordinates  # These are atomic coordinates (Nx3 array)
    f = cryst.molecule.get_scattering_factors(beam=beam)
    print(photon_energy/1.6022e-19, np.abs(np.unique(f)))

    A = simcore.phase_factor_qrf(q_vecs, r_vecs, f)
    F2 = np.abs(A) ** 2
    I = detector.f2_to_photon_counts(F2, beam=beam, pad_geometry=pad_geometry)
    I *= n_frames * n_proteins
    I = np.random.poisson(I)
    I = pad_geometry.reshape(I)
    profile1, qmags = detector.get_radial_profile(I, beam=beam, pad_geometry=pad_geometry, n_bins=200)

    plt.figure(1)
    plt.plot(qmags[10:]/1e10, profile1[10:], label='%6.2f eV'%(photon_energy/1.602e-19))
    plt.ylabel('Scattering intensity [photons/pixel]')
    plt.xlabel('Q [Angstrom]')
    plt.figure(2)
    plt.imshow(I, interpolation='nearest', cmap='gray', origin='lower')

plt.figure(1)
plt.legend()
plt.show()
