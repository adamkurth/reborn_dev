import numpy as np
from reborn.simulate.clcore import ClCore
from reborn.target import crystal
import matplotlib.pyplot as plt
# Example of how to simulate a finite crystal with reborn utilities, specially made for Joe!
# Some parameters to get us started:
pdb_id = '2LYZ'
photon_energy = 9000*1.602e-19
resolution = 5e-10
oversampling = 6
# Start the GPU Engine
clcore = ClCore()
# Make the CrystalStructure.  Recall: this combines atoms + spacegroup + unit cell.  No lattice info.
# Note: tight_packing option => molecule COMs will be in unit cell, which is nice for displays.
crystal_structure = crystal.CrystalStructure(pdb_id, tight_packing=True)
# Make the FiniteCrystal tool, which stores lattice coordinates + occupancies, helps create facets and more.
finite_crystal = crystal.FiniteCrystal(crystal_structure, max_size=20)
# Make the CrystalDensityMap tool, which chooses sensible grid points along with symmetry transform operators.
density_map = crystal.CrystalDensityMap(cryst=crystal_structure, resolution=resolution, oversampling=oversampling)
# Generate the actual density map of the asymmetric unit (a numpy array).  Trilinear insertion is probably ok for now...
f = np.real(crystal_structure.molecule.get_scattering_factors(photon_energy=photon_energy))
au_map = density_map.place_atoms_in_map(crystal_structure.fractional_coordinates, f, mode='trilinear', fixed_atom_sigma=10e-10)
# Generate (and save) molecular transforms of each symmetry partner by (1) re-mapping asymmetric unit then (2) FFT
mol_amps = []
for k in range(crystal_structure.spacegroup.n_operations):
    rho = density_map.au_to_k(k, au_map)
    mol_amps.append(clcore.to_device(np.fft.fftshift(np.fft.fftn(rho)), dtype=clcore.complex_t))
# Use the FiniteCrystal tool to form a particular crystal.  We can of course generate lots of them and merge together.
# For Lysozyme, we make a simple parallelepiped.  There is a disorder option if desired, and also we could randomly
# set some occupancies to zero to make a holey crystal.  A remaining task is to make some stacking-fault crystals, which
# can probably be done by manually inserting more symmetry partners into the CrystalStructure instance, and then
# tweaking the FiniteLattice occupancies to create ABC bands without overlapping molecules.  Note that if we add
# symmetry partners to the CrystalStructure, then you will not need to change *anything* in the IPA.  You simply have
# more transform operators.
# finite_crystal.make_hexagonal_prism(width=2, length=2)
finite_crystal.make_parallelepiped((2, 2, 2))
# Now that FiniteCrystal has occupancies set up as desired, we will compute lattice transforms and form the amplitudes
# from the whole crystal.  If you want more crystals, you can "reset" the FiniteCrystal occupancies and repeat.
lattice_amps = clcore.to_device(shape=density_map.shape, dtype=clcore.complex_t)
intensity = clcore.to_device(shape=density_map.shape, dtype=clcore.real_t) * 0
crystal_amps = 0
lims = density_map.h_limits * 2 * np.pi
for k in range(crystal_structure.spacegroup.n_molecules):
    x = finite_crystal.lattices[k].occupied_x_coordinates
    clcore.phase_factor_mesh(x, N=density_map.shape, q_min=lims[:, 0], q_max=lims[:, 1], a=lattice_amps, add=False)
    crystal_amps += lattice_amps * mol_amps[k]  # <--- Easy formula to write down on paper, but tons of bookkeeping!
# Now the ** BIG TEST **: if we inverse FFT, will we see a sensible crystal?  We've mixed the FFT method for the
# molecular transform along with the direct GPU computation for the lattice transforms, with attention to how
# translation vectors of symmetry partners and lattice points are handled... not a simple task!!!  The benefit of this
# complexity is that we can have arbitrarily large crystals than the oversampled real-space grid would allow.
crystal_density = np.real(np.fft.ifftn(np.fft.ifftshift(crystal_amps.get().reshape(density_map.shape))))
plt.subplot(1, 3, 1)
plt.imshow(np.sum(np.fft.fftshift(crystal_density), axis=0))
plt.subplot(1, 3, 2)
plt.imshow(np.sum(np.fft.fftshift(crystal_density), axis=1))
plt.subplot(1, 3, 3)
plt.imshow(np.sum(np.fft.fftshift(crystal_density), axis=2))
plt.show()
# Looks good to me!  For the reconstruction algorithm, you probably just need the density_map.au_to_k method.
