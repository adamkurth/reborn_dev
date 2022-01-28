"""
Simulate Ag Behenate diffraction patterns from protein pdb_id on a Jungfrau 4M detector.

Date Created: 21 Nov 2021
Last Modified: 21 Nov 2021
Author: RAK, JC

     
"""
import numpy as np
from scipy.spatial.transform import Rotation
from reborn import source, detector, const
from reborn.target import crystal, atoms
from reborn.simulate import clcore
from reborn.viewers.mplviews import view_pad_data
#------------------------------
np.random.seed(2021)  # Make random numbers that are reproducible
photon_energy = 12000     # eV
diameter_fwhm = 0.2e-6   # m
pulse_energy = 100         # J
detector_distance = 0.2  # m
pdb_id = '1507774.pdb'
N_patterns = 3           # Number of patterns to simulate
is_poisson_noise = False  # Turn Poisson noise on or off
#------------------------------
eV = const.eV
r_e = const.r_e
beam = source.Beam(photon_energy=photon_energy*eV, diameter_fwhm=diameter_fwhm, pulse_energy=pulse_energy)
fluence = beam.photon_number_fluence
pads = detector.jungfrau4m_pad_geometry_list(detector_distance=detector_distance)
pads = pads.binned(16)
q_vecs = pads.q_vecs(beam=beam)
solid_angles = pads.solid_angles()
polarization_factors = pads.polarization_factors(beam=beam)
q_mags = pads.q_mags(beam=beam)
cryst = crystal.CrystalStructure(pdb_id)
r_vecs = cryst.molecule.coordinates
r_vecs -= np.mean(r_vecs, axis=0)
cryst.a = 4.1769
cryst.b = 4.7218
cryst.c = 58.3385
cryst.al = 89.440 / 180 * np.pi
cryst.be = 89.634 / 180 * np.pi
cryst.ga = 75.854 / 180 * np.pi
simcore = clcore.ClCore()
uniq_z = np.unique(cryst.molecule.atomic_numbers)
grouped_r_vecs = []
grouped_fs = []
for z in uniq_z:
    subr = np.squeeze(r_vecs[np.where(cryst.molecule.atomic_numbers == z), :])
    grouped_r_vecs.append(subr)
    grouped_fs.append(atoms.hubbel_henke_scattering_factors(q_mags=q_mags, photon_energy=beam.photon_energy,
                                                            atomic_number=z))
for n in range(N_patterns):
    print(f'Simulating pattern {n+1}')
    R = Rotation.random().as_matrix()  # Just for fun, let's rotate the molecule
    amps = 0
    for j in range(len(grouped_fs)):
        f = grouped_fs[j]
        r = grouped_r_vecs[j]
        a = simcore.phase_factor_qrf(q_vecs, r, R=R)
        amps += a*f
    ints = r_e**2*fluence*solid_angles*polarization_factors*np.abs(amps)**2
    if is_poisson_noise == True:
        ints = np.double(np.random.poisson(ints))
    dispim = np.log10(ints+1)
    view_pad_data(pad_data=dispim, pad_geometry=pads)
