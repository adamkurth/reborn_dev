"""
# Example of how to simulate a finite crystal with reborn utilities, specially made for Joe!
Simulate diffraction patterns from disordered materials.
This script simulates 3D Fourier volumes directly and then merges them, 
as opposed to simulating individual 2D patterns then merging.

To do:
- Stacking fault

Date Created: 2020
Last Modified: 3 Jan 2022
Humans responsible: Rick Kirian, Joe Chen
"""

import numpy as np
import scipy.constants as const
from scipy.spatial.transform import Rotation

from reborn.simulate.clcore import ClCore
from reborn.target import crystal


np.random.seed(42)

CMAP = 'viridis'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.close('all')
keV = 1e3 * const.value('electron volt')
r_e = const.value("classical electron radius")

#==========================================================================

# Some parameters to get us started:
pdb_id = '1Jb0'
photon_energy = 9 * keV
resolution = 4e-10
oversampling = 10

#==========================================================================
# Helper functions

def colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6) 


def show_slice(disp_map, N_cent, clim, disp_str):
    """
    Plot the three orthogonal slices
    """
    fig = plt.figure()
    ax = fig.add_subplot(131)
    im = ax.imshow(disp_map[N_cent[0],:,:], clim=clim, cmap=CMAP, origin='lower')
    colorbar(ax, im)
    ax.set_title('[Nx_cent,:,:]')
    ax = fig.add_subplot(132)
    im = ax.imshow(disp_map[:,N_cent[1],:], clim=clim, cmap=CMAP, origin='lower')
    colorbar(ax, im)
    ax.set_title('[:,Ny_cent,:]')
    ax = fig.add_subplot(133)
    im = ax.imshow(disp_map[:,:,N_cent[2]], clim=clim, cmap=CMAP, origin='lower')
    colorbar(ax, im)
    ax.set_title('[:,:,Nz_cent]')

    plt.suptitle(disp_str)
    plt.tight_layout()
    plt.show(block=False)

def show_proj(disp_map, N_cent, disp_str):
    """
    Plot the three orthogonal projections
    """
    fig = plt.figure()
    ax = fig.add_subplot(131)
    im = ax.imshow(np.sum(disp_map, axis=0), interpolation='nearest', cmap=CMAP, origin='lower')
    colorbar(ax, im)
    ax.set_title('[Nx_cent,:,:]')
    ax = fig.add_subplot(132)
    im = ax.imshow(np.sum(disp_map, axis=1), interpolation='nearest', cmap=CMAP, origin='lower')
    colorbar(ax, im)
    ax.set_title('[:,Ny_cent,:]')
    ax = fig.add_subplot(133)
    im = ax.imshow(np.sum(disp_map, axis=2), interpolation='nearest', cmap=CMAP, origin='lower')
    colorbar(ax, im)
    ax.set_title('[:,:,Nz_cent]')

    plt.suptitle(disp_str)
    plt.tight_layout()
    plt.show(block=False)

#==========================================================================
# Start the GPU Engine
clcore = ClCore()

# Make the CrystalStructure.  Recall: this combines atoms + spacegroup + unit cell.  No lattice info.
# Note: tight_packing option => molecule COMs will be in unit cell, which is nice for displays.
crystal_structure = crystal.CrystalStructure(pdb_id, tight_packing=True)

# yay

# We want to make a hexagonal lattice with ABC stacking fault


#------------
# Turn the crystal into a P1 manually - this is just a label - does not affect anything
crystal_structure.spacegroup.spacegroup_symbol = 'P 1'


# Variables we have to play with
# a = None  #: Lattice constant (float)
# b = None  #: Lattice constant (float)
# c = None  #: Lattice constant (float)
# alpha = None  #: Lattice angle in radians (float)
# beta = None  #: Lattice angle in radians (float)
# gamma = None  #: Lattice angle in radians (float)
# volume = None  #: Unit cell volume (float)
# o_mat = None  #: Orthogonalization matrix (3x3 array).  Does the transform r = O.x on fractional coordinates x.
# o_mat_inv = None  #: Inverse orthogonalization matrix (3x3 array)
# a_mat = None  #: Orthogonalization matrix transpose (3x3 array). Does the transform q = A.h, with Miller indices h.
# a_mat_inv = None  #: A inverse


a = crystal_structure.unitcell.a
b = crystal_structure.unitcell.b
c = crystal_structure.unitcell.c

al = crystal_structure.unitcell.alpha
be = crystal_structure.unitcell.beta
ga = np.pi/3

# Set gamma to 60 degs
crystal_structure.unitcell.gamma = ga

# Re-calculate volume and o matrices
vol = a * b * c * np.sqrt(1 - np.cos(al)**2 - np.cos(be) **
              2 - np.cos(ga)**2 + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
crystal_structure.unitcell.volume = vol

o_mat = np.array([
        [a, b * np.cos(ga), c * np.cos(be)],
        [0, b * np.sin(ga), c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)],
        [0, 0, vol / (a * b * np.sin(ga))]
        ])
crystal_structure.unitcell.o_mat = o_mat

o_inv = np.array([
        [1 / a, -np.cos(ga) / (a * np.sin(ga)), 0],
        [0, 1 / (b * np.sin(ga)), 0],
        [0, 0, a * b * np.sin(ga) / vol]
        ])
crystal_structure.unitcell.o_inv = o_inv




# Make rotations - all three rotations are the identity operator
ok = []
for i in range(3):
    ok.append(np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]))
crystal_structure.spacegroup.sym_rotations = ok

# Make translations
a_vec = crystal_structure.unitcell.a_vec
b_vec = crystal_structure.unitcell.b_vec

crystal_structure.spacegroup.sym_translations = [0 for i in range(3)]
crystal_structure.spacegroup.sym_translations[0] = crystal_structure.unitcell.r2x(np.array([0.0,0.0,0.0])) 
crystal_structure.spacegroup.sym_translations[1] = crystal_structure.unitcell.r2x((a_vec+b_vec)/3)
crystal_structure.spacegroup.sym_translations[2] = crystal_structure.unitcell.r2x((2*b_vec-a_vec)/3)


#------------

print('Number of symmetry operations: %d' % (crystal_structure.spacegroup.n_operations,))
print(crystal_structure.unitcell)
print(crystal_structure.spacegroup)



# Make the FiniteCrystal tool, which stores lattice coordinates + occupancies, helps create facets and more.
finite_crystal = crystal.FiniteCrystal(crystal_structure, max_size=10)



# Visualise the lattice


l1 = finite_crystal.lattices[0].occupied_r_coordinates
l2 = finite_crystal.lattices[1].occupied_r_coordinates

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(l1[:,0], l1[:,1], l1[:,2])
ax.scatter(l2[:,0], l2[:,1], l2[:,2])
plt.show(block=False)


# yay2


A = finite_crystal.lattices[0].occupancies
A[:,1,:] = 0
# A[:,2,:] = 0
# A[:,4,:] = 0
# A[:,6,:] = 0
finite_crystal.lattices[0].occupancies = A

B = finite_crystal.lattices[1].occupancies
# B[:,0,:] = 0
# B[:,2,:] = 0
# B[:,3,:] = 0
# B[:,5,:] = 0
finite_crystal.lattices[1].occupancies = B

C = finite_crystal.lattices[2].occupancies
# C[:,0,:] = 0
# C[:,1,:] = 0
# C[:,3,:] = 0
# C[:,4,:] = 0
# C[:,5,:] = 0
# C[:,6,:] = 0
finite_crystal.lattices[2].occupancies = C


from reborn.viewers.qtviews import view_finite_crystal
view_finite_crystal(finite_crystal)



yay






# Make the CrystalDensityMap tool, which chooses sensible grid points along with symmetry transform operators.
density_map = crystal.CrystalDensityMap(cryst=crystal_structure, resolution=resolution, oversampling=oversampling)

# Generate the actual density map of the asymmetric unit (a numpy array).  Trilinear insertion is probably ok for now...
f = np.real(crystal_structure.molecule.get_scattering_factors(photon_energy=photon_energy))
au_map = density_map.place_atoms_in_map(crystal_structure.fractional_coordinates, f, mode='trilinear', fixed_atom_sigma=10e-10)

# Generate (and save) molecular transforms of each symmetry partner by (1) re-mapping asymmetric unit then (2) FFT
mol_amps = []
for k in range(crystal_structure.spacegroup.n_operations):
    # rho = density_map.au_to_k(k, au_map)  # This and the line below give the same result
    rho = density_map.symmetry_transform(0, k, au_map)
    mol_amps.append(clcore.to_device(np.fft.fftshift(np.fft.fftn(rho)), dtype=clcore.complex_t))

# Use the FiniteCrystal tool to form a particular crystal.  We can of course generate lots of them and merge together.
# For Lysozyme, we make a simple parallelepiped.  There is a disorder option if desired, and also we could randomly
# set some occupancies to zero to make a holey crystal.  A remaining task is to make some stacking-fault crystals, which
# can probably be done by manually inserting more symmetry partners into the CrystalStructure instance, and then
# tweaking the FiniteLattice occupancies to create ABC bands without overlapping molecules.  Note that if we add
# symmetry partners to the CrystalStructure, then you will not need to change *anything* in the IPA.  You simply have
# more transform operators.
# finite_crystal.make_hexagonal_prism(width=2, length=2)
finite_crystal.make_parallelepiped((5, 7, 3))

# ====================================================================================================================
# Now we try to make a crappy crystal surface, with missing molecules.
print('Making crystal surface')
x_crappy_surface = []
for k in range(len(finite_crystal.lattices)):

    # Here are the occupied coordinates of the kth lattice
    x = finite_crystal.lattices[k].occupied_x_coordinates.copy()

    # The above lattice is centered at the origin, but that's not representative of where the center of mass of the
    # molecules is.  However, the FiniteCrystal class stores those COMs for us -- here's the COM of the kth molecule:
    com = finite_crystal.au_x_coms[k]

    # At this point you have all the locations of the molecules: they are located at the coords: x + com .  Let's now
    # Randomly eliminate some lattice points at the surface of the parallelepiped-shaped crystal.  I will do just one
    # of the six facets:
    d1 = np.dot(x + com, np.array([1, 0, 0]))  # projection of coordinates onto facet normal vector
    occ = np.ones(d1.shape, dtype=int)  # Start with all vectors occupied
    w = np.where(d1 == np.max(d1))[0]  # Find the ones at the surface
    w = w[np.random.choice([True, False], w.shape, p=[0.4, 0.6])]  # Randomly kill off some of these surface molecules
    occ[w] = 0
    x = np.squeeze(x[np.where(occ), :])

    # Now get just the coordinates that are occupied.
    x_crappy_surface.append(x)
    
    # Note that we've killed molecules from the surface of this sub lattice without considering the other sub lattices.
    # We will therefore get "floating" molecules... this should be done more intelligently using all of the coordinates
    # from the whole crystal.

# =====================================================================================================================
# Now that FiniteCrystal has occupancies set up as desired, we will compute lattice transforms and form the amplitudes
# from the whole crystal.  If you want more crystals, you can "reset" the FiniteCrystal occupancies and repeat.
lattice_amps = clcore.to_device(shape=density_map.shape, dtype=clcore.complex_t)
intensity = clcore.to_device(shape=density_map.shape, dtype=clcore.real_t) * 0
crystal_amps = 0
lims = density_map.h_limits * 2 * np.pi

print('Calculating the diffraction') 
for k in range(crystal_structure.spacegroup.n_molecules):
    print(k)

    # Use the coordinates worked on above to calculate the diffraction 
    x = x_crappy_surface[k]
    # x = finite_crystal.lattices[k].occupied_x_coordinates

    clcore.phase_factor_mesh(x, N=density_map.shape, q_min=lims[:, 0], q_max=lims[:, 1], a=lattice_amps, add=False)
    crystal_amps += lattice_amps * mol_amps[k]  # <--- Easy formula to write down on paper, but tons of bookkeeping!


print('Calculating crystal in real space')
# Now the ** BIG TEST **: if we inverse FFT, will we see a sensible crystal?  We've mixed the FFT method for the
# molecular transform along with the direct GPU computation for the lattice transforms, with attention to how
# translation vectors of symmetry partners and lattice points are handled... not a simple task!!!  The benefit of this
# complexity is that we can have arbitrarily large crystals than the oversampled real-space grid would allow.
F = crystal_amps.get().reshape(density_map.shape)

crystal_density = np.real(np.fft.ifftn(np.fft.ifftshift(F)))
crystal_density = np.fft.fftshift(crystal_density)

# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(np.sum(np.fft.fftshift(crystal_density), axis=0))
# plt.subplot(1, 3, 2)
# plt.imshow(np.sum(np.fft.fftshift(crystal_density), axis=1))
# plt.subplot(1, 3, 3)
# plt.imshow(np.sum(np.fft.fftshift(crystal_density), axis=2))
# plt.show(block=False)
# Looks good to me!  For the reconstruction algorithm, you probably just need the density_map.au_to_k method.






Nx_os, Ny_os, Nz_os = crystal_density.shape
Nx_os_cent = int(Nx_os/2)
Ny_os_cent = int(Ny_os/2)
Nz_os_cent = int(Nz_os/2)
N_os_cent_vec = [Nx_os_cent, Ny_os_cent, Nz_os_cent]

# Intensity
I = np.abs(F)**2
show_slice(I, N_cent=N_os_cent_vec, clim=[0,1e9], disp_str='Diffracted intensity slice')
show_proj(np.log10(I), disp_str='Diffracted intensity projection', N_cent=N_os_cent_vec)


# N_cent, clim, disp_str

show_slice(crystal_density, N_cent=N_os_cent_vec, clim=None, disp_str='crystal_density')
show_proj(crystal_density, disp_str='crystal_density', N_cent=N_os_cent_vec)








