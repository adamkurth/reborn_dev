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
Diffraction from two point scatterers
=====================================

Simple simulation of two point scatterers on a GPU.

Contributed by Richard A. Kirian.

Imports:
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import reborn
from reborn.simulate import clcore

# %%
# First we create the ClCore instance that manages compute devices (GPUs and CPUs):
simcore = clcore.ClCore()

# %%
# Let's see what kind of device we are computing with.  If it is not a GPU, computations may be slow:
print(simcore.get_device_name())

# %%
# Create a beam and detector:
pad = reborn.detector.PADGeometry(shape=(1000, 1000), pixel_size=100e-6, distance=0.05)
beam = reborn.source.Beam(wavelength=1e-10)
print(pad)
print(beam)

# %%
# Scattering vectors:
q = pad.q_vecs(beam=beam)

# %%
# Coordinates of the point scatterers:
r = np.zeros([2, 3])
r[1, 0] = 20e-10
print(r)

# %%
# Scattering factors:
f = np.ones([2])

# %%
# Compute diffraction amplitudes and intensities using the formula
#
# .. math::
#
#   I(\vec{q}) = \left| \sum_n f_n \exp(i\vec{q}\cdot\vec{r}) \right|^2
A = simcore.phase_factor_qrf(q, r, f)
I = np.abs(A)**2

# %%
# We can add in the polarization factor due to classical dipole radiation:
I *= pad.polarization_factors(beam=beam)  # Polarization factors

# %%
# Display diffraction intensities.  We first need to re-shape our intensities into a 2D array.
I = pad.reshape(I)
plt.imshow(I, interpolation='nearest', cmap='gray', origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()

# %%
# Finally, let's just have a look at what compute devices are available to us:
clcore.help()
