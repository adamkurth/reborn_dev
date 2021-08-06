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
Diffraction from three spheres
==============================

Simple example of diffraction simulation of three spheres in a plane.

Contributed by Richard A. Kirian.

Imports:
"""
import numpy as np
import matplotlib.pyplot as plt
import reborn
from reborn import simulate
from reborn.simulate.form_factors import sphere_form_factor
from reborn.simulate.clcore import ClCore

# %%
# We will first create an ensemble of spheres in a plane.
# The real-space scattering density is:
#
# .. math::
#
#    \rho_\text{tot}(\vec{r}) = \sum_n \rho_\text{sph}(|\vec{r} - \vec{s}_n|)
#
# where :math:`\vec{s}_n` is the position of the :math:`n` th sphere.  The Fourier-space density is:
#
# .. math::
#
#    F_\text{tot}(\vec{q}) = \sum_n F_\text{sph}(q) \exp(i \vec{q}\cdot \vec{s}_n)
#
# Notably, the form factor of the sphere, :math:`F_\text{sph}(q)`, can come out of the sum:
#
# .. math::
#
#     F_\text{tot}(\vec{q}) = F_\text{sph}(q) \sum_n \exp(i \vec{q}\cdot\vec{s}_n)
#
# We simulate the form factor of the sphere once, and do the sum over the position phasors separately.

# %%
# Create the beam and pixel array detector (PAD)
beam = reborn.source.Beam(wavelength=5e-9, beam_vec=[0, 0, 1])
pad = reborn.detector.PADGeometry(shape=(1000, 1000), pixel_size=10e-6, distance=0.1)
q_vecs = pad.q_vecs(beam=beam)
q_mags = pad.q_mags(beam=beam)
print(beam)

# %%
# Construct the arrangement of sphere coordinates:
radius = 0.5e-6
pos = np.array([[-3, -3, 0], [-3, 4, 0], [0, 2, 0]])*1e-6
print(pos)

# %%
# Create an instance of the simulation engine:
simcore = simulate.clcore.ClCore()

# %%
# Compute the scattering amplitudes:
scat = simcore.phase_factor_qrf(q=q_vecs, r=pos)
scat *= simulate.form_factors.sphere_form_factor(radius=radius, q_mags=q_mags)*1e30
scat = np.abs(scat)**2

# %%
# Mask the direct beam
beam_mask = pad.beamstop_mask(beam=beam, min_angle=0.001)

# %%
# Need to re-shape our simulated intensities to 2D arrays:
scat = pad.reshape(scat)
beam_mask = pad.reshape(beam_mask)

# %%
# Display the diffraction intensities along with the Fourier transform of the intensities.  The FT of the intensities
# is equal to the autocorrelation of the scattering density, however, there are artifacts caused by Ewald curvature
# and the beamstop.
cmap = 'CMRmap'
fig = plt.figure(2)
dispim1 = scat
dispim1 *= beam_mask
dispim1 /= np.max(dispim1)
dispim1 = np.log10(dispim1*1e5 + 1)
ax1 = fig.add_subplot(121)
ax1.imshow(dispim1, cmap=plt.get_cmap(cmap))
plt.title('Diffraction Intensity')
dispim2 = np.fft.fftshift(np.abs(np.fft.fftn(scat)))
dispim2 /= np.max(dispim2)
dispim2 = np.log10(dispim2 + 1e-1)
ax2 = fig.add_subplot(122)
ax2.imshow(dispim2, cmap=plt.get_cmap(cmap))
plt.title('Diffraction Autocorrelation')
plt.show()
