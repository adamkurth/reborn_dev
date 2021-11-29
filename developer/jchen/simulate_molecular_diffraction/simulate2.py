"""
Simulate N_patterns diffraction patterns from protein pdb_id on a Jungfrau 4M detector.

Date Created: 21 Nov 2021
Last Modified: 21 Nov 2021
Author: RAK, JC
"""

import numpy as np
import scipy.constants as const
from scipy.spatial.transform import Rotation
from reborn import detector
from reborn.source import Beam
from reborn.target import crystal, atoms
from reborn.simulate import clcore
from reborn.viewers.mplviews import view_pad_data

#------------------------------
# Inputs
file_name_tosave = 'test.npz'
photon_energy = 9000     # eV
diameter_fwhm = 0.2e-6   # m
pulse_energy = 0.1         # J
detector_distance = 0.2  # m
pdb_id = '2LYZ'
N_patterns = 5           # Number of patterns to simulate
is_poisson_noise = True # Turn Poisson noise on or off
#------------------------------

# Define some constants and other parameters that we'll need:
eV = const.value('electron volt')
r_e = const.value('classical electron radius')
np.random.seed(2021)  # Make random numbers that are reproducible


# Setup the beam and detector.  Note that the Jungfrau 4M detector has multiple panels, so we need to deal with lists
# of detector panels for things like scattering vectors, pixel solid angles, etc.
beam = Beam(photon_energy=photon_energy*eV, diameter_fwhm=diameter_fwhm, pulse_energy=pulse_energy)
fluence = beam.photon_number_fluence
pads = detector.jungfrau4m_pad_geometry_list(detector_distance=detector_distance)


# Speed up simulations by binning pixels 16x16
pads = pads.binned(16)
q_vecs = [pad.q_vecs(beam=beam) for pad in pads]
solid_angles = [pad.solid_angles() for pad in pads]
polarization_factors = [pad.polarization_factors(beam=beam) for pad in pads]
q_mags = [p.q_mags(beam=beam) for p in pads]


# Here is the resolution range:
qmin = np.min(np.array([np.min(q) for q in q_mags]))
qmax = np.max(np.array([np.max(q) for q in q_mags]))
print('resolution range (Angstrom): ', 2*np.pi*1e10/qmax, ' - ', 2*np.pi*1e10/qmin)


# Load a lysozyme pdb_id file and transform to a CrystalStructure object.  This particular pdb_id file is included in reborn
# for examples like this.  You can try a different pdb_id ID, and reborn will attempt to download the file for you.
cryst = crystal.CrystalStructure(pdb_id)
r_vecs = cryst.molecule.coordinates  # Atomic coordinates of the asymmetric unit
r_vecs -= np.mean(r_vecs, axis=0)  # Roughly center the molecule
atomic_numbers = cryst.molecule.atomic_numbers


# Create a ClCore instance.  In reborn, the ClCore class helps maintain a context and queue with a GPU device.
# It has the functions you probably need in order to do simulations.
simcore = clcore.ClCore()

# Let's see what kind of device we are running on.  If it is not a GPU, the simulations will not be very fast...
simcore.print_device_info()


# We will use the following formula for diffraction intensities:
#
# .. math::
#
#     I(\vec{q}) = J_0 \Delta \Omega r_e^2 P(\vec{q})\left| \sum_n f_n(q) \sum_m \exp(i \vec{q}\cdot\vec{r}_{mn}) \right|^2
#
# The double sum allows us to compute the atomic form factors :math:`f_n(q)` just once for each atom type :math:`n`.
# We must search through the atom types and group them according to atomic number.
uniq_z = np.unique(atomic_numbers)
grouped_r_vecs = []
grouped_fs = []
for z in uniq_z:
    subr = np.squeeze(r_vecs[np.where(atomic_numbers == z), :])
    grouped_r_vecs.append(subr)
    grouped_fs.append(atoms.hubbel_henke_scattering_factors(q_mags=q_mags, photon_energy=beam.photon_energy,
                                                            atomic_number=z))

# Now we have the atomic coordinates :math:`\vec{r}_{mn}` and scattering factors :math:`f_n(q)` for atom type
# :math:`n` and atom number :math:`m`.  Note that we also have a list of detector panels (PADs) that we must loop over.
# Here we go!

intensities_list = []
for n in range(N_patterns):
    print(f'Simulating pattern {n+1}')

    R = Rotation.random().as_matrix()  # Just for fun, let's rotate the molecule
    intensities = []
    for i in range(len(pads)):
        pad = pads[i]
        sa = solid_angles[i]
        p = polarization_factors[i]
        q = q_vecs[i]
        amps = 0
        for j in range(len(grouped_fs)):
            f = grouped_fs[j][i]
            r = grouped_r_vecs[j]
            a = simcore.phase_factor_qrf(q, r, R=R)
            amps += a*f
        ints = r_e**2*fluence*sa*p*np.abs(amps)**2

        if is_poisson_noise == True:
            ints = np.random.poisson(ints) # Add some Poisson noise

        intensities.append(pad.reshape(ints))

    # Let's see how many photons hit the detector:
    print('Total number of photons on detector: %d' % np.round(np.sum(detector.concat_pad_data(intensities))))

    # Finally, display the pattern:
    dispim = [np.log10(d+1) for d in intensities]
    view_pad_data(pad_data=dispim, pad_geometry=pads)

    # Append each simulated pattern into a list
    intensities_list.append(intensities)



# Save the intensities and the inputs to the program
# np.savez(file_name_tosave, intensities_list=intensities_list, 
#                            q_vecs=q_vecs,
#                            photon_energy=photon_energy, 
#                            diameter_fwhm=diameter_fwhm,
#                            pulse_energy=pulse_energy,
#                            detector_distance=detector_distance,
#                            pdb_id=pdb_id,
#                            N_patterns=N_patterns,
#                            is_poisson_noise=is_poisson_noise)


