import numpy as np
from bornagain.target.crystal import CrystalStructure, CrystalDensityMap
from bornagain.simulate.clcore import ClCore
import pyqtgraph as pg

# The CrystalStructure object has a UnitCell, SpaceGroup, and other information
cryst = CrystalStructure('lysozyme')
# The oversampling ratio:
osr = 2
# The desired map resolution:
res = 2e-10
# The CrystalDensityMap is a helper class.  On initialization, sampling in the crystal basis is configured such that
# the crystallographic symmetry operations of a density map are strictly permutation operations (no interpolations).
cdmap = CrystalDensityMap(cryst, res, osr)
# The ClCore instance manages the GPU for simulations.
clcore = ClCore()

# Create two atom position vectors, both at the origin.
x_vecs = np.zeros([2, 3])
# Now shift one of them along the "z" coordinate by n steps.  The step comes from the CrystalDensityMap.
n = np.round(1/cdmap.dx[2]).astype(int) - 1
x_vecs[0, 2] = n*cdmap.dx[2]

# Create some pnony scattering factors
f = np.zeros(2)
f[0] = 1
f[1] = 2

# METHOD 1:
# Simulate amplitudes using atomistic coordinates.  Recipcorcal-space coordinates are chosen such that they will
# correspond to a numpy FFT operation.  The CrystalDensityMap class conveniently provides these limits for this purpose.
amps1 = clcore.phase_factor_mesh(x_vecs, f=f, q_min=cdmap.h_min*2*np.pi, q_max=cdmap.h_max*2*np.pi, N=cdmap.shape)
# Because the phase_factor_mesh function above computes on a grid, the direct 000 voxel is centered.  We must shift
# the array to put the amplitudes in the standard FFT arrangement, with the 000 as the first voxel.
amps1 = np.fft.ifftshift(amps1.reshape(cdmap.shape))
dmap1 = np.fft.ifftn(amps1)

# METHOD 2:
# First make the scattering density map, and then FFT the map to create amplitudes.
dmap2 = np.zeros(cdmap.shape)
# Instead of defining a list of atomic coordinates, we directly set the scattering densities to the scattering factors
# used for METHOD 1:
dmap2[0, 0, 0] = f[0]
dmap2[0, 0, n] = f[1]
amps2 = np.fft.fftn(dmap2)


def compare(a, b):
    print(np.max(a-b), np.mean((a+b)/2), np.max(a-b)/np.mean((a+b)/2))


compare(np.abs(amps1), np.abs(amps2))
compare(np.abs(np.real(amps1)), np.abs(np.real(amps2)))
compare(np.abs(np.imag(amps1)), np.abs(np.imag(amps2)))

pg.image(np.abs(np.real(amps1)), title='1 (clcore)')
pg.image(np.abs(np.real(amps2)), title='2 (fft)')
pg.show()
pg.mkQApp().exec_()