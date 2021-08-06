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
3D rotation of a density map
===============================

Rotate a 3D array via three Euler angles.

Contributed by Joe Chen and Kevin Schmidt

Imports:
"""

import reborn as ba

import numpy as np
from reborn.target import crystal
import scipy.constants as const

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from reborn.utils import rotate3D

eV = const.value('electron volt')

# %%
# Make a density map.
pdb_file = '1jb0.pdb'
resolution = 3e-10
oversampling = 1
photon_energy_ev = 12000


cryst = crystal.CrystalStructure(pdb_file, tight_packing=True)
uc = cryst.unitcell
sg = cryst.spacegroup
print(uc)
print(sg)


cdmap = crystal.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=oversampling)


f = cryst.molecule.get_scattering_factors(photon_energy=photon_energy_ev*eV)
print('sum of f', np.sum(f))

rho_au = cdmap.place_atoms_in_map(cryst.fractional_coordinates, f, mode='trilinear')
print('sum of rho_au', np.sum(rho_au))



# %%
# Define plotting functions and display the density map.

CMAP = "viridis"

print(rho_au.shape)
Nx, Ny, Nz = rho_au.shape
Nx_cent = int(np.round(Nx/2))
Ny_cent = int(np.round(Ny/2))
Nz_cent = int(np.round(Nz/2))

def show_projection(disp_map, disp_str):
    """
    Projection
    """
    fig = plt.figure()
    ax = fig.add_subplot(131)
    im = ax.imshow(np.sum(disp_map,axis=0), interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[Nx_cent,:,:]')
    ax = fig.add_subplot(132)
    im = ax.imshow(np.sum(disp_map,axis=1), interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[:,Ny_cent,:]')
    ax = fig.add_subplot(133)
    im = ax.imshow(np.sum(disp_map,axis=2), interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[:,:,Nz_cent]')

    plt.suptitle(disp_str)
    plt.tight_layout()
    plt.show()


disp_map = np.abs(rho_au)
disp_str = 'Asymmetric unit: projection'
show_projection(disp_map, disp_str)


# %%
# Rotate.

x_rot = rotate3D(f=rho_au,
                 euler_angles=np.array([0, 0, 60]) * (np.pi / 180))
# x_rot = rotate3D(f=x_rot, 
#                  Euler_angles=np.array([0,0,30])*(np.pi/180))
disp_map = (np.real(x_rot))
disp_str = 'Asymmetric unit rotated: projection'
show_projection(disp_map, disp_str)




x_rot = rotate3D(f=rho_au,
                 euler_angles=np.array([0, 0, 60]) * (np.pi / 180))
disp_map = (np.abs(x_rot))
disp_str = 'Asymmetric unit rotated: projection'
show_projection(disp_map, disp_str)

