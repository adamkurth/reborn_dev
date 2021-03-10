r"""
Radial statistics calculation
=============================

Calculate radial statistics

Contributed by Joe Chen.

"""

# %%
# First we import the needed modules and configure the simulation parameters:

import numpy as np
import matplotlib.pyplot as plt
from reborn.utils import make_label_radialShell, radial_stats

Nx = 20
Ny = 20
Nz = 20
N_radials = 6

# Make a 3D array of radnom numbers
f = np.random.rand(Nx, Ny, Nz)

# Values of the radius for the start of each shell
r_bin_vec = np.linspace(0, int(Nx/2), N_radials)

# Set up the radial labels
labels_radial = make_label_radialShell(r_bin_vec, N_vec=(Nx, Ny, Nz))

# Calculate the radial median
radial_median = radial_stats(f=f, labels_radial=labels_radial, N_radials=N_radials, mode="median")


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
plt.show()

# Radial median
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(r_bin_vec, radial_median, 'o-', markersize=7, lw=2)
ax.grid()
plt.show()
