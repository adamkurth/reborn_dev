import numpy as np
from bornagain.target import crystal
from bornagain.simulate import clcore, atoms
import pyqtgraph as pg
from scipy import constants

# The CrystalStructure object has a UnitCell, SpaceGroup, and other information.  The input can be any path to a PDB
# file or it can be the name of a PDB entry.  The PDB will be fetched from the web if necessary and possible.  The
# PDB entry 2LYZ comes with bornagain.
cryst = crystal.CrystalStructure('2LYZ')

# The oversampling ratio:
osr = 2
# The desired map resolution, which will be adjusted according to crystal lattice and sampling constraints:
res = 2e-10
# The CrystalDensityMap is a helper class that ensures sampling in the crystal basis is configured such that
# the crystal spacegroup symmetry operations of a density map can be performed strictly through permutation operations.
# Thus, no interpolations are needed for spacegroup symmetry operations.
cdmap = crystal.CrystalDensityMap(cryst, res, osr)

# The ClCore instance manages the GPU for simulations.
simcore = clcore.ClCore()

# Create two atom position vectors, both at the origin.
x_vecs = np.zeros([2, 3])
# Now shift one of them along the "z" coordinate (in crystal basis) by n steps.  The step size comes from the
# CrystalDensityMap, which, again, considers how to intelligently sample crystallographic density maps.
n = 2 #np.round(1/cdmap.dx[2]).astype(int) - 1
x_vecs[1, 2] = n*cdmap.dx[2]

# Get some scattering factors
f = np.ones((2,)).astype(np.complex)  #atoms.get_scattering_factors(atomic_numbers=[6, 8], photon_energy=1e4*constants.eV)
f[0] = 2
f[1] = 1j
# METHOD 1:
# Simulate amplitudes using atomistic coordinates, structure factors, and a direct summation over
#                              F(h) =  sum_n f_n*exp(-i 2*pi*h.x_n)
# Recipcorcal-space coordinates are chosen such that they will correspond to a numpy FFT operation.  The limits of that
# sample grid are provided by the CrystalDensityMap class:
g_min = cdmap.h_min * 2 * np.pi
g_max = cdmap.h_max * 2 * np.pi
# Simulation tool for regular 3D grid of reciprocal-space samples.
amps1 = simcore.phase_factor_mesh(x_vecs, f=f, q_min=g_min, q_max=g_max, N=cdmap.shape)
# Because the phase_factor_mesh function above computes on a grid, the direct 000 voxel is centered.  We must shift
# the array such that the h=000 is located at the first voxel as per the standard FFT arrangement in numpy.
amps1 = np.fft.ifftshift(amps1.reshape(cdmap.shape))
# Transforming from amplitudes to density is now a simple inverse FFT.
dmap1 = np.fft.ifftn(amps1.astype(np.float32))

# METHOD 2:
# First make the scattering density map, and then FFT the map to create amplitudes.
dmap2 = np.zeros(cdmap.shape).astype(np.complex64)
# Instead of defining a list of atomic coordinates, we directly set the scattering densities to the scattering factors
# used in METHOD 1.  Note that we've chosen atomic coordinates so that they will lie exactly on grid points in our 3D
# maps.
dmap2[0, 0, 0] = f[0]
dmap2[0, 0, n] = f[1]
amps2 = np.fft.fftn(dmap2)
print(amps2.dtype)

def compare(a, b):
    print('max(a-b) = %.2g,   mean((a+b)/2) = %.2g,   max(a-b) / mean((a+b)/2) = %.2g' %
          (np.max(a-b), np.mean((a+b)/2), np.max(a-b)/np.mean((a+b)/2)))



print(osr, cdmap.shape, cdmap.h_min, cdmap.h_max)
print(f)
print(dmap2.ravel()[0:4])
print(dmap1.ravel()[0:4])

print('\n\n\n')


# In principle, Methods 1 and 2 should yield identical results, aside from different float operations on the GPU and CPU
print('Compare a = |amps1|, b = |amps2| :')
compare(np.abs(amps1), np.abs(amps2))
print('Compare a = Re(amps1), b = Re{amps2} :')
compare(np.abs(np.real(amps1)), np.abs(np.real(amps2)))
print('Compare a = Im(amps1), b = Im{amps2} :')
compare(np.abs(np.imag(amps1)), np.abs(np.imag(amps2)))



# pg.image(np.abs(amps1)/np.abs(amps2))
pg.image(np.abs(dmap1), title='dmap1')
pg.image(np.abs(dmap2), title='dmap2')
# dif = np.abs(dmap1-dmap2)
# a = np.abs(dmap1)
# b = np.abs(dmap2)
#
# div = np.abs(dmap1)/np.abs(dmap2)
# print(np.max(div))
# pg.image(div, title='dmap1-dmap2')



# pg.image(np.abs(np.real(amps1)), title='1 (clcore)')
# pg.image(np.abs(np.real(amps2)), title='2 (fft)')
# pg.show()
pg.mkQApp().exec_()