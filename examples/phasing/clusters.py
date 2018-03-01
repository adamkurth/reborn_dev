import sys
from time import time

if 1:
    import afnumpy as np
    from afnumpy.fft import fftn, ifftn, fftshift
else:
    import numpy as np
    from numpy.fft import fftn, ifftn, fftshift

import matplotlib.pyplot as plt
import pyqtgraph as pg

sys.path.append("../..")
import bornagain as ba
from bornagain.viewers import qtviews
from bornagain.target import crystal, map

Niter = 100  # Number of phase-retrieval iterations

pdbFile = '../data/pdb/1JB0.pdb'
print('Loading pdb file (%s)' % pdbFile)
cryst = crystal.structure(pdbFile)
print(cryst.cryst1.strip())

print('Getting scattering factors')
wavelength = 1.5e-10
f = ba.simulate.atoms.get_scattering_factors(cryst.Z, ba.units.hc / wavelength)

print('Setting up 3D mesh')
d = 0.15e-9  # Minimum resolution in SI units (as always!)
s = 1  # Oversampling factor.  s = 1 means Bragg sampling
mt = map.CrystalMeshTool(cryst, d, s)
print('Grid size: (%d, %d, %d)' % (mt.N, mt.N, mt.N))
# h = mt.get_h_vecs()  # Miller indices (fractional)

print('Creating density map directly from atoms')
x = cryst.x
rho0 = mt.place_atoms_in_map(cryst.x % mt.s, np.abs(f))

J = 1
K = []
w = np.array([1])
L = []
Linv = []
for j in range(0, J):

    L.append([])
    Linv.append([])
    K.append(len(mt.get_sym_luts()))

    for i in range(0, K[j]):

        L[j].append(lambda a: mt.symmetry_transform(0, i, a))
        Linv[j].append(lambda a: mt.symmetry_transform(i, 0, a))


rho_cell = 0
for i in range(0, K[0]):
    rho_cell += L[0][i](rho0)
rho0 = rho_cell


if 0:
    print("Showing intial density (volumetric)")
    view = qtviews.Volumetric3D()
    view.add_density(rho0)
    view.show()

if 0:
    print("Showing initial density (slices)")
    qtviews.MapSlices(rho0)

if 0:
    print("Showing initial density (projections)")
    qtviews.MapProjection(rho0, axis=0)

# Create the initial support
S0 = mt.zeros()
S0.flat[rho0.flat != 0] = 1

if 0:
    print("Showing initial support (volumetric)")
    view = qtviews.Volumetric3D()
    view.add_density(S0)
    view.show()

if 0:
    print("Showing initial support (slices)")
    qtviews.MapSlices(S0)

# Create the "measured" intensities
rho0 = np.array(rho0)
I0 = np.abs(fftn(rho0)) ** 2
sqrtI0 = np.sqrt(I0)

if 0:
    print("Showing intensities (slices)")
    qtviews.MapSlices(np.log(fftshift(I0) + 1))


beta = 0.9
gamma_s = -1/beta
gamma_m = 1/beta

def symmetrize(rho):

    n_molecules = len(mt.get_sym_luts())

    rhosym = 0
    for i in range(0, n_molecules):
        rhosym += mt.symmetry_transform(0, i, rho)

    return rhosym/n_molecules

def PS(rho, S):

    return rho*S

def PM(rho, sqrtI0):

    return ifftn(sqrtI0 * np.exp(1j * np.angle(fftn(rho))))

def RS(rho, S):

    psrho = PS(rho, S)
    return (1+gamma_s)*psrho - gamma_s*rho

def RM(rho, sqrtI0):

    pmrho = PM(rho, sqrtI0)
    return (1 + gamma_m) * pmrho - gamma_m * rho

def DM(rho, S, sqrtI0):

    rmrho = RM(rho, sqrtI0)
    rsrho = RS(rho, S)

    psrho = PS(rmrho, S)
    pmrho = PM(rsrho, sqrtI0)

    return rho + beta*(psrho - pmrho)

def ER(rho, S, sqrtI0):

    return PM(PS(rho, S), sqrtI0)

# Create the initial phases
phi = np.random.random([mt.N]*3) * 2 * np.pi

# Create the initial iterate
rho = ifftn(np.exp(phi) * sqrtI0)

# Do phase retrieval
errors = np.zeros([Niter])
t = time()
for i in np.arange(0, Niter):

    if (i % 20) > 5:
        rho = DM(rho, S0, sqrtI0)
        alg = 'DM'
    else:
        rho = ER(rho, S0, sqrtI0)
        alg = 'ER'

    rho = symmetrize(rho)

    R = np.sum((rho-rho0)**2)/np.sum(rho0**2)
    errors[i] = R

    print("Iteration #%d (%s; R: %.2g)" % (i, alg, R))

delT = time() - t
print('Total time (s): %g ; Time per iteration (s): %g' % (delT, delT / float(Niter)))

if 1:
    print("Showing reconstruction (image viewer)")
    pg.image(np.abs(rho))
    pg.show()

if 0:
    print("Showing reconstruction (volumetric)")
    view = qtviews.Volumetric3D()
    view.add_density(np.abs(rho))
    view.show()
    # plt.imshow(-np.sum(np.abs(rho), axis=0), cmap='gray', interpolation='none')
    # plt.show()

if 0:
    print("Showing reconstruction (projections)")
    qtviews.MapProjection(np.abs(rho), axis=[0,1])

if 1:
    plt.plot(np.log10(errors))
    plt.xlabel('Iteration number')
    plt.ylabel(r'$R = \frac{\sum (\rho_c-\rho_m)^2}{ \sum \rho_m^2 }$')
    plt.show()

