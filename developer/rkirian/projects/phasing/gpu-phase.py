from time import time
from scipy import constants

if False:
    import afnumpy as np
    from afnumpy.fft import fftn, ifftn, fftshift
else:
    import numpy as np
    from numpy.fft import fftn, ifftn, fftshift

import matplotlib.pyplot as plt

# from reborn import units
from reborn.viewers import qtviews
from reborn.target import crystal

hc = constants.h*constants.c

Niter = 500  # Number of phase-retrieval iterations

load_pdb = False

if not load_pdb:  # Make a phony water molecule

    # Settings
    N = 64  # Size of the 3D density array

    # Create the 3D density array
    def sphere(R, T):
        x = np.arange(0, N) - N / 2.0 + 0.5 + N * T[0]
        y = np.arange(0, N) - N / 2.0 + 0.5 + N * T[1]
        z = np.arange(0, N) - N / 2.0 + 0.5 + N * T[2]
        xx, yy, zz = np.meshgrid(x, y, z, indexing='xy')
        rr = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
        rho = np.zeros([N, N, N])
        R *= N
        rho.flat[rr.flat < R] = 1  # rr.flat[rr.flat < R]
        return rho

    rho0 = sphere(0.2, [0, 0, 0])
    rho0 += sphere(0.1, [0.2, 0, 0])
    rho0 += sphere(0.1, [0, 0, 0.2])

else:  # Alternatively, load a pdb file

    cryst = crystal.CrystalStructure('lysozyme')

    print('Getting scattering factors')
    wavelength = 1.5e-10
    f = cryst.molecule.get_scattering_factors(hc / wavelength)

    print('Setting up 3D mesh')
    d = 0.1e-9  # Minimum resolution in SI units (as always!)
    s = 1  # Oversampling factor.  s = 1 means Bragg sampling
    mt = crystal.CrystalDensityMap(cryst, d, s)
    print('Grid size:', mt.shape)
    h = mt.h_vecs  # Miller indices (fractional)

    print('Creating density map directly from atoms')
    x = cryst.x
    rho0 = mt.place_atoms_in_map(cryst.x % mt.oversampling, np.abs(f))
    # Make a full unit cell
    rho_cell = 0
    for i in range(0, len(mt.get_sym_luts())):
        rho_cell += mt.symmetry_transform(0, i, rho0)
    rho0 = rho_cell

rho0 = np.abs(rho0)


if 0:
    print("Showing intial density (volumetric)")
    view = qtviews.Volumetric3D()
    view.add_density(np.abs(rho0))
    view.show(hold=True, kill=True)

if 0:
    print("Showing initial density (slices)")
    qtviews.MapSlices(np.abs(rho0))

if 0:
    print("Showing initial density (projections)")
    qtviews.MapProjection(np.abs(rho0))

# Create the initial support
S0 = np.zeros(rho0.shape)
S0.flat[np.abs(rho0.flat) < 0.01*np.max(np.abs(rho0.flat))] = 1

if 0:
    print("Showing initial support (volumetric)")
    view = qtviews.Volumetric3D()
    view.add_density(S0)
    view.show()

if 0:
    print("Showing initial support (slices)")
    qtviews.MapSlices(S0)

# Create the "measured" intensities
I0 = np.abs(fftn(rho0)) ** 2
sqrtI0 = np.sqrt(I0)

if 0:
    print("Showing intensities (slices)")
    qtviews.MapSlices(np.log(fftshift(I0) + 1))

# Create the initial phases
phi = np.random.random(rho0.shape) * 2 * np.pi

# Create the initial iterate
rho = ifftn(np.exp(phi) * sqrtI0)

# Do phase retrieval
t = time()
for i in np.arange(0, Niter):
    print("Iteration #%d" % (i))
    rho = np.real(rho * S0)
    rho = ifftn(sqrtI0 * np.exp(1j * np.angle(fftn(rho))))
delT = time() - t
print('Total time (s): %g ; Time per iteration (s): %g' % (delT, delT / float(Niter)))

if 0:
    print("Showing reconstruction (volumetric)")
    view = qtviews.Volumetric3D()
    view.add_density(np.abs(rho))
    view.show()
    # plt.imshow(-np.sum(np.abs(rho), axis=0), cmap='gray', interpolation='none')
    # plt.show()

if 0:
    print("Showing reconstruction (projections)")
    view = qtviews.MapProjection(np.abs(rho), axis=0)
    # view.show(hold=True, kill=True)
