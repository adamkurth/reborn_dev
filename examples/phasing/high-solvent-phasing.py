import sys
from time import time

if 0:
    import afnumpy as np
    from afnumpy.fft import fftn, ifftn
else:
    import numpy as np
    from numpy.fft import fftn, ifftn

import pyqtgraph as pg
import matplotlib.pyplot as plt

sys.path.append("../..")
from bornagain.viewers.qtviews import qtviews
from bornagain.target import crystal, density

Niter = 200  # Number of phase-retrieval iterations

pdbFile = crystal.get_pdb_file('1jb0')
print('Loading pdb file (%s)' % pdbFile)
cryst = crystal.CrystalStructure(pdbFile)

print('Getting scattering factors')
f = cryst.molecule.get_scattering_factors(wavelength=1.5e-10)

print('Setting up 3D mesh')
d = 0.8e-9  # Minimum resolution in SI units (as always!)
s = 1  # Oversampling factor.  s = 1 means Bragg sampling
mt = density.CrystalMeshTool(cryst, d, s)
n_molecules = len(mt.get_sym_luts())
print('Grid size: (%d, %d, %d)' % (mt.N, mt.N, mt.N))
# h = mt.get_h_vecs()  # Miller indices (fractional)

print('Creating density map directly from atoms')
x = cryst.x
rho0 = mt.place_atoms_in_map(cryst.x % mt.s, np.abs(f))

# hkl_file = '../data/crystfel/stream-13k-allruns.hkl'
# hkl = np.genfromtxt(hkl_file, skip_header=3, skip_footer=1, usecols=[0, 1, 2, 3])
# Idata = hkl[:, 3]
# hkl = hkl[:, 0:3]
# print(Idata.shape)
# print(hkl.shape)
# Idata = mt.place_intensities_in_map(hkl, Idata)
# if 1:
#     pg.image(Idata)

rho_cell = 0
for i in range(0, n_molecules):
    rho_cell += mt.symmetry_transform(0, i, rho0)
rho0 = rho_cell

# im = pg.image(np.abs(rho0))
# pg.QtGui.QApplication.instance().exec_()

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
S0.flat[rho0.flat >= np.percentile(rho0.flat, 70)] = 1

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

# Create initial support
S = S0.copy()

# Do phase retrieval
errors = np.zeros([Niter])
t = time()
for i in np.arange(0, Niter):

    if (i % 20) > 10:
        rho = DM(rho, S, sqrtI0)
        alg = 'DM'
    else:
        rho = ER(rho, S, sqrtI0)
        alg = 'ER'

    rho = symmetrize(rho)

    if ((i+1) % 1e6 == 0):
        S *= 0
        rhomag = np.abs(rho)
        S.flat[rhomag.flat >= np.percentile(rhomag.flat, 60)] = 1

    R = np.sum(np.abs(rho-rho0)**2)/np.sum(np.abs(rho0)**2)
    errors[i] = R

    print("Iteration #%d (%s; R: %.2g)" % (i, alg, R))

delT = time() - t
print('Total time (s): %g ; Time per iteration (s): %g' % (delT, delT / float(Niter)))


if 0:
    pg.image(fftshift(I0))

if 0:
    print("Showing reconstruction (image viewer)")
    pg.image(np.abs(rho))

if 0:
    print("Showing reconstruction (volumetric)")
    view = qtviews.Volumetric3D()
    view.add_density(np.abs(rho))
    view.show()
    # plt.imshow(-np.sum(np.abs(rho), axis=0), cmap='gray', interpolation='none')
    # plt.show()

if 1:
    print("Showing reconstruction (projections)")
    qtviews.MapProjection(np.abs(rho), axis=[0, 1, 2])

if 1:
    plt.plot(np.log10(errors))
    plt.xlabel('Iteration number')
    plt.ylabel(r'$R = \frac{\sum (\rho_c-\rho_m)^2}{ \sum \rho_m^2 }$')
    plt.show()

