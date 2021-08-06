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
Diffraction from finite lattice
===============================

Simple diffraction simulation from lattice of point scatterers.

Contributed by Richard A. Kirian

Imports:
"""

import numpy as np
import matplotlib.pyplot as plt
import reborn as ba
from reborn.simulate import clcore

# %%
# Simulation core:
simcore = clcore.ClCore()

# %%
# Detector and beam:
pad = ba.detector.PADGeometry(shape=(1000, 1000), pixel_size=100e-6, distance=0.05)
beam = ba.source.Beam(wavelength=1e-10)
q = pad.q_vecs(beam=beam)

# %%
# Atomic coordinates:
N = 2
x = np.arange(0, N) * 10e-10
[xx, yy, zz] = np.meshgrid(x, x, x, indexing='ij')
r = np.zeros([N ** 3, 3])
r[:, 0] = zz.flatten()
r[:, 1] = yy.flatten()
r[:, 2] = xx.flatten()

# %%
# Scattering factors:
f = np.ones([N ** 3])

# %%
# Compute diffraction amplitudes:
A = simcore.phase_factor_qrf(q, r, f)

# %%
# Display diffraction intensities
I = pad.reshape(np.abs(A) ** 2)
dispim = np.log10(I + 0.1)
plt.imshow(dispim, interpolation='nearest', cmap='gray', origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()
