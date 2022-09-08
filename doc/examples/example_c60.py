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
GPU Simulation (C60 molecules)
==============================

Examples of how to simulate C60 molecules from xyz file using GPU devices.

Contributed by Richard A. Kirian.

Imports:
"""
import numpy as np
import reborn
from reborn.simulate.clcore import ClCore
from reborn.viewers.qtviews import PADView
np.random.seed(0)
# %%
# Our agenda here is to simulate atomistic diffraction from multiple C60 molecules.  We begin by creating the |ClCore|
# class instance that handles the GPU configurations:
simcore = ClCore(group_size=32, double_precision=False)
# %%
# Set up the |PAD| and |Beam|, which contain the experiment parameters:
beam = reborn.source.Beam(wavelength=1e-10, pulse_energy=1e-3)
geom = reborn.detector.jungfrau4m_pad_geometry_list(detector_distance=0.07, binning=10)
q_vecs = geom.q_vecs(beam=beam)
# %%
# Load the atomic coordinates and species from an "xyz" file:
xyz = reborn.fileio.misc.load_xyz('files/C60.xyz')
r_vecs = xyz['position_vecs']
z = xyz['atomic_numbers']
n_atoms = len(z)
# %%
# Pre-allocating GPU arrays helps speed up computations:
q_vecs_gpu = simcore.to_device(q_vecs)
r_vecs_gpu = simcore.to_device(r_vecs)
amps_gpu = simcore.to_device(shape=(geom.n_pixels,), dtype=simcore.complex_t)
# %%
# We first simulate the atomistic diffraction assuming atomic scattering factors are all :math:`f(q)=1`.
simcore.phase_factor_qrf(q_vecs_gpu, r_vecs_gpu, a=amps_gpu)
# %%
# Since all the atoms are carbon, we can take a shortcut and multiply the amplitudes by the carbon scattering factor
# :math:`f(q)`.  There are various sources of scattering factors -- here we will combine the |Cromer1968| form factors
# with the |Henke1993| dispersion corrections:
q_mags = geom.q_mags(beam=beam)
f = reborn.target.atoms.cmann_henke_scattering_factors(q_mags, atomic_number=z[0], beam=beam)
# %%
# Multiply the amplitudes by the form factor.  Be careful here... the amplitudes are on the GPU still while the form
# factors are still in the ordinary RAM.
f_gpu = simcore.to_device(f)
amps_gpu *= f_gpu
# %%
# In order to convert to photon counts, we need the Thompson scattering cross-section, polarization factor, solid angles
# of the pixels, and incident fluence.  At this point we will move the arrays to ordinary RAM.
amps = amps_gpu.get()  # This copies GPU memory to RAM
intensity = np.abs(amps)**2
intensity *= geom.solid_angles()
intensity *= geom.polarization_factors(beam=beam)
intensity *= beam.photon_number_fluence
intensity *= reborn.const.r_e**2
# %%
# Now we can view the resulting pattern.  Since we have a segmented detector with multiple PADs, we can use |PADView|
# to lay out the geometry correctly:
pv = PADView(data=np.log10(intensity), beam=beam, pad_geometry=geom, title='test')
pv.start()
# %%
# Now that we have done all of the above, it is straightforward to sum up amplitudes from many molecules in different
# positions and orientations.  The GPU functions can accept translation vectors and rotation matrices:
amps_gpu *= 0
n_molecules = 5
for i in range(n_molecules):
    R = reborn.utils.random_rotation()
    U = np.random.normal(size=3)*10e-10
    simcore.phase_factor_qrf(q_vecs_gpu, r_vecs_gpu, a=amps_gpu, R=R, U=U, add=True)
amps_gpu = amps_gpu*f_gpu
# %%
# For convenience, the Thompson scattering cross-section, solid angles, etc. are combined in the f2phot method of
# |PADGeometry|:
intensity = np.abs(amps_gpu.get())**2 * geom.f2phot(beam=beam)
# %%
# Let's see what we created:
pv = PADView(data=np.log10(intensity), beam=beam, pad_geometry=geom, title='test')
pv.start()
# %%
# On my IBM Thinkpad laptop it takes about 1 millisecond per molecule.
