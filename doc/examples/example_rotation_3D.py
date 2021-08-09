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
import numpy as np
from reborn.target import crystal
import scipy.constants as const
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from reborn.misc.rotate import Rotate3D
eV = const.value('electron volt')
# %%
# Make a density map.
pdb_file = '1LYZ.pdb'
resolution = 3e-10
oversampling = 1
photon_energy_ev = 12000
cell_size = 100e-10
uc = crystal.UnitCell(cell_size, cell_size, cell_size, np.pi / 2, np.pi / 2, np.pi / 2)
sg = crystal.SpaceGroup('P1', [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])], [np.zeros(3)])
cryst = crystal.CrystalStructure(pdb_file, spacegroup=sg, unitcell=uc)
cdmap = crystal.CrystalDensityMap(cryst, resolution, oversampling)
f = cryst.molecule.get_scattering_factors(photon_energy=photon_energy_ev*eV)
x = cryst.unitcell.r2x(cryst.molecule.get_centered_coordinates())
rho_au = cdmap.place_atoms_in_map(x, f, mode='trilinear')
rho_au = np.fft.fftshift(rho_au)
# %%
# Define plotting functions and display the density map.
Nx, Ny, Nz = rho_au.shape
Nx_cent = int(np.round(Nx/2))
Ny_cent = int(np.round(Ny/2))
Nz_cent = int(np.round(Nz/2))
def show_projection(disp_map, disp_str):
    fig = plt.figure()
    ax = fig.add_subplot(131)
    im = ax.imshow(np.sum(disp_map.astype(float), axis=0), interpolation='nearest', origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[Nx_cent,:,:]')
    ax = fig.add_subplot(132)
    im = ax.imshow(np.sum(disp_map,axis=1), interpolation='nearest', origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[:,Ny_cent,:]')
    ax = fig.add_subplot(133)
    im = ax.imshow(np.sum(disp_map,axis=2), interpolation='nearest', origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[:,:,Nz_cent]')
    plt.suptitle(disp_str)
    plt.tight_layout()
    # plt.show()

disp_map = np.abs(rho_au)
# disp_str = 'Asymmetric unit: projection'
# show_projection(disp_map, disp_str)
# %%
# Rotate.
rot = Rotate3D(rho_au)
disp_map = (np.real(rho_au))
disp_str = 'Asymmetric unit projections'
show_projection(disp_map, disp_str)
phi = 30*np.pi/180.0
c = np.cos(phi)
s = np.sin(phi)
R = np.array([[c, 0, s],
              [0, 1, 0],
              [-s, 0, c]])
Rs = scipy.spatial.transform.Rotation.from_matrix(R)
rot.rotation(Rs)
disp_map = (np.abs(rot.f))
disp_str = 'Rotated asymmetric unit projections'
show_projection(disp_map, disp_str)

plt.show()