r"""
Making real space density maps
==============================
"""

# %%
# The tools for producing real-space scattering density maps are still under development.  Here we show just one way
# to make density maps with reborn.

import numpy as np
from reborn.target import crystal
import scipy.constants as const
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

eV = const.value('electron volt')

pdb_file = '1jb0.pdb'
resolution = 5e-10
oversampling = 1
photon_energy_ev = 12000

# %%
# Create a crystal object. This object has molecule, unit cell and spacegroup info.
# Tight packing lets the asymmetric units be positioned in a physical way.
cryst = crystal.CrystalStructure(pdb_file, tight_packing=True)
uc = cryst.unitcell
sg = cryst.spacegroup
print(uc)
print(sg)

# %%
# Create a crystal density map object. This object has spacegroup info and helps with spacegroup symmetry
# transformations.
cdmap = crystal.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=oversampling)

# %%
# Scattering factors.  These are the the :math:`f(q=0)` scattering factors that come from the Henke Tables.
f = cryst.molecule.get_scattering_factors(photon_energy=photon_energy_ev*eV)
print('sum of f', np.sum(f))

# %%
# Make the density map for the asymmetric unit. 
# Here we do a trilinear insertion of the scattering factors, giving a 3D array of complex-valued numbers.
rho_au = cdmap.place_atoms_in_map(cryst.fractional_coordinates, f, mode='trilinear')
print('sum of rho_au', np.sum(rho_au))

# %%
# Assemble the unit cell density by generating all symmetry partners of the asymmetric unit and adding them up.
rho_uc = 0
for k in range(cryst.spacegroup.n_operations):
    rho_uc += cdmap.au_to_k(k, rho_au)
print('sum of rho_uc', np.sum(rho_uc))


# %%
# Have a look at the results!

CMAP = "viridis"

print(rho_au.shape)
Nx, Ny, Nz = rho_au.shape
Nx_cent = int(np.round(Nx/2))
Ny_cent = int(np.round(Ny/2))
Nz_cent = int(np.round(Nz/2))

def show_slice(disp_map, disp_str):
    """
    Slice
    """
    fig = plt.figure()
    ax = fig.add_subplot(131)
    im = ax.imshow(disp_map[Nx_cent,:,:], interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[Nx_cent,:,:]')
    ax = fig.add_subplot(132)
    im = ax.imshow(disp_map[:,Ny_cent,:], interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[:,Ny_cent,:]')
    ax = fig.add_subplot(133)
    im = ax.imshow(disp_map[:,:,Nz_cent], interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[:,:,Nz_cent]')

    plt.suptitle(disp_str)
    plt.tight_layout()
    plt.show()

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
disp_str = 'Asymmetric unit: central slice'
show_slice(disp_map, disp_str)

disp_map = np.abs(rho_au)
disp_str = 'Asymmetric unit: projection'
show_projection(disp_map, disp_str)

disp_map = np.abs(rho_uc)
disp_str = 'Unit cell: central slice'
show_slice(disp_map, disp_str)

disp_map = np.abs(rho_uc)
disp_str = 'Unit cell: projection'
show_projection(disp_map, disp_str)
