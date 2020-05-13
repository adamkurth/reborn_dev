r"""
Making real space density maps
===============
"""

# %%
# Here we show how real space density maps can be generated with reborn.


import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from reborn.simulate.clcore import ClCore
from reborn.target import crystal
import scipy.constants as const

np.random.seed(42)
eV = const.value('electron volt')

resolution = 5e-10
oversampling = 1
photon_energy_ev = 12000
pdb_file = '1JB0.pdb'

# Create a crystal object. This has molecule, unit cell, and spacegroup info
cryst = crystal.CrystalStructure(pdb_file, tight_packing=True)  # Tight packing: put molecule COMs inside unit cell
uc = cryst.unitcell
sg = cryst.spacegroup
print(uc)
print(sg)


# Density map configuration with spacegroup considerations
cdmap = crystal.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=oversampling)


# GPU simulation engine
clcore = ClCore()

# Scattering factors
f = cryst.molecule.get_scattering_factors(photon_energy=photon_energy_ev*eV)
print('sum over f', np.sum(f))

# au_map = cdmap.place_atoms_in_map(cryst.fractional_coordinates, f, mode='gaussian', fixed_atom_sigma=1e-10)
au_map = cdmap.place_atoms_in_map(cryst.fractional_coordinates, f, mode='trilinear')
print('sum over au_map', np.sum(au_map))

# Assemble the unit cell density
rho = 0
for k in range(cryst.spacegroup.n_operations):
    rho += cdmap.au_to_k(k, au_map)

print('sum over rho', np.sum(rho))


#================================================================================
# Plotting stuff
# import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# font_size = 12
# mpl.rcParams['xtick.labelsize'] = font_size
# mpl.rcParams['ytick.labelsize'] = font_size
CMAP = "viridis"

plt.close('all')


print(au_map.shape)
Nx,Ny,Nz = au_map.shape
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


disp_map = np.abs(au_map)
disp_str = 'Asymmetric unit map - central slice'
show_slice(disp_map, disp_str)

disp_map = np.abs(au_map)
disp_str = 'Asymmetric unit map- projection'
show_projection(disp_map, disp_str)


disp_map = np.abs(rho)
disp_str = 'Unit cell unit map - central slice'
show_slice(disp_map, disp_str)

disp_map = np.abs(rho)
disp_str = 'Unit cell unit map- projection'
show_projection(disp_map, disp_str)


