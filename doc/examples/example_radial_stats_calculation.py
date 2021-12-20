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
Radial statistics calculation
=============================

Calculate the radial statistics of a 3-dimensional array of numbers.

Contributed by Joe Chen.

"""

# %%
# First we import the needed modules and configure the simulation parameters:

import numpy as np
import matplotlib.pyplot as plt
from reborn.utils import make_label_radial_shell, radial_stats

# Shape of the numpy array
Nx = 20
Ny = 20
Nz = 20

# Number of radial bins we want
N_radials = 6

# Make a 3D array of random numbers
f = np.random.rand(Nx, Ny, Nz)

# Values of the radius for the start of each shell
r_bin_vec = np.linspace(0, int(Nx/2), N_radials)

# Set up the radial labels
labels_radial = make_label_radial_shell(r_bin_vec, n_vec=(Nx, Ny, Nz))


# %%
# Now we calculate the radial median of the array, as an example:

radial_median = radial_stats(f=f, labels_radial=labels_radial, n_radials=N_radials, mode="median")


# %%
# Display results:

# Radial labels
fig = plt.figure()
ax = fig.add_subplot(131)
plt.imshow(labels_radial[:,:,int(Nz/2)])
plt.colorbar()
ax = fig.add_subplot(132)
plt.imshow(labels_radial[:,int(Ny/2),:])
plt.colorbar()
ax = fig.add_subplot(133)
plt.imshow(labels_radial[int(Nx/2),:,:])
plt.colorbar()
plt.tight_layout()
plt.show()

# Radial median
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(r_bin_vec, radial_median, 'o-', markersize=7, lw=2)
ax.grid()
plt.show()
