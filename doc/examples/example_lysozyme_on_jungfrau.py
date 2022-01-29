# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

r"""
Molecule diffraction from a PDB file
====================================

Simulate lysozyme diffraction on a Jungfrau 4M detector.

Contributed by Richard A. Kirian.

Imports:
"""

import numpy as np
from scipy.spatial.transform import Rotation
from reborn import source, detector, const
from reborn.target import crystal, atoms
from reborn.simulate import clcore
from reborn.viewers.mplviews import view_pad_data

# %%
# Define some constants and other parameters that we'll need:
eV = const.eV
r_e = const.r_e
np.random.seed(0)  # Make random numbers that are reproducible

# %%
# Setup the beam and detector.  Note that the Jungfrau 4M detector has multiple panels, so we need to deal with lists
# of detector panels for things like scattering vectors, pixel solid angles, etc.
beam = source.Beam(photon_energy=9000*eV, diameter_fwhm=0.2e-6, pulse_energy=2)
fluence = beam.photon_number_fluence
pads = detector.jungfrau4m_pad_geometry_list(detector_distance=0.1)
# Speed up simulations by binning pixels 16x16
pads = pads.binned(16)
q_vecs = pads.q_vecs(beam=beam)
solid_angles = pads.solid_angles()
polarization_factors = pads.polarization_factors(beam=beam)
q_mags = pads.q_mags(beam=beam)

# %%
# Here is the resolution range:
qmin = np.min(np.array([np.min(q) for q in q_mags]))
qmax = np.max(np.array([np.max(q) for q in q_mags]))
print('resolution range (Angstrom): ', 2*np.pi*1e10/qmax, ' - ', 2*np.pi*1e10/qmin)

# %%
# Load a lysozyme PDB file and transform to a CrystalStructure object.  This particular PDB file is included in reborn
# for examples like this.  You can try a different PDB ID, and reborn will attempt to download the file for you.
cryst = crystal.CrystalStructure('2LYZ')
r_vecs = cryst.molecule.coordinates  # Atomic coordinates of the asymmetric unit
r_vecs -= np.mean(r_vecs, axis=0)  # Roughly center the molecule
atomic_numbers = cryst.molecule.atomic_numbers

# %%
# Create a ClCore instance.  In reborn, the ClCore class helps maintain a context and queue with a GPU device.
# It has the functions you probably need in order to do simulations.
simcore = clcore.ClCore()

# %%
# Let's see what kind of device we are running on.  If it is not a GPU, the simulations will not be very fast...
simcore.print_device_info()

# %%
# We will use the following formula for diffraction intensities:
#
# .. math::
#
#     I(\vec{q}) = J_0 \Delta \Omega r_e^2 P(\vec{q})\left| \sum_n f_n(q) \sum_m \exp(i \vec{q}\cdot\vec{r}_{mn}) \right|^2
#
# The double sum allows us to compute the atomic form factors :math:`f_n(q)` just once for each atom type :math:`n`.
# We must search through the atom types and group them according to atomic number.  In this example we will use the
# |Hubbel1975| q-dependent form factors mixed with the energy-dependent dispersion corrections from |Henke1993|.
uniq_z = np.unique(atomic_numbers)
grouped_r_vecs = []
grouped_fs = []
for z in uniq_z:
    subr = np.squeeze(r_vecs[np.where(atomic_numbers == z), :])
    grouped_r_vecs.append(subr)
    grouped_fs.append(atoms.hubbel_henke_scattering_factors(q_mags=q_mags, photon_energy=beam.photon_energy,
                                                            atomic_number=z))

# %%
# Now we have the atomic coordinates :math:`\vec{r}_{mn}` and scattering factors :math:`f_n(q)` for atom type
# :math:`n` and atom number :math:`m`.
R = Rotation.random().as_matrix()  # Just for fun, let's rotate the molecule
amps = 0
for j in range(len(grouped_fs)):
    print('element z =', uniq_z[j])
    amps += grouped_fs[j] * simcore.phase_factor_qrf(q_vecs, grouped_r_vecs[j], R=R)

# %%
# Convert to photons per pixel then add sPoisson noise
ints = r_e**2*fluence*solid_angles*polarization_factors*np.abs(amps)**2
ints = np.double(np.random.poisson(ints))
print('# photons total: %d' % np.round(np.sum(detector.concat_pad_data(ints))))

# %%
# Finally, display the pattern (plus 1, log scale):
view_pad_data(pad_data=np.log10(ints+1), pad_geometry=pads)
