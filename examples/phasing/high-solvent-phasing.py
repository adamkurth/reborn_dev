import sys
from time import time
import numpy as np
from numpy.fft import fftn, ifftn
import pyqtgraph as pg
from bornagain.target import crystal, density

beta = 0.90        # Difference Map parameter
gamma_s = -1/beta  # Difference Map parameter
gamma_m = 1/beta   # Difference Map parameter
n_iter = 200       # Number of phase-retrieval iterations
solv_frac = 0.7    # Solvent fraction

try:
    a = pdb_file
except NameError:

    d = 0.5e-9         # Minimum resolution in SI units
    s = 1              # Oversampling factor.  s = 1 means Bragg sampling

    pdb_file = crystal.get_pdb_file('1jb0')
    cryst = crystal.CrystalStructure(pdb_file)

    f = cryst.molecule.get_scattering_factors(wavelength=1.5e-10)

    dens = crystal.CrystalDensityMap(cryst, d, s)
    n_molecules = len(dens.get_sym_luts())
    print('Grid size: (%d, %d, %d)' % tuple(dens.shape))

    x = cryst.x
    print('Placing atoms in density map (this is slow... will speed up later...)')
    rho0 = dens.place_atoms_in_map(cryst.x % dens.oversampling, np.abs(f))

    rho_cell = np.zeros_like(rho0)
    for i in range(0, n_molecules):
        rho_cell += dens.symmetry_transform(0, i, rho0)
    rho0 = rho_cell

    rho0.flags.writeable = False


print('Loading pdb file (%s)' % pdb_file)
print('Getting scattering factors')
print('Setting up 3D mesh')
print('Creating density map directly from atoms')

# Create the initial support
S0 = np.zeros_like(rho0)
S0.flat[rho0.flat > np.percentile(rho0.flat, solv_frac*100)] = 1

# Create the "measured" intensities
I0 = np.abs(fftn(rho0)) ** 2
sqrtI0 = np.sqrt(I0)


def symmetrize(rho):

    n_molecules = len(dens.get_sym_luts())

    rhosym = 0
    for i in range(0, n_molecules):
        rhosym += dens.symmetry_transform(0, i, rho)

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

# Initial phases
phi = np.random.random(dens.shape) * 2 * np.pi
# Iterate
rho = ifftn(np.exp(phi) * sqrtI0)
# Initial support
S = S0.copy()

# Do phase retrieval
print('Start phase retrieval...')
errorsI = np.zeros([n_iter])
errorsrho = np.zeros([n_iter])
t = time()
msg = ''
for i in np.arange(0, n_iter):

    if i < 50 or (i % 50) > 40:
        rho = ER(rho, S, sqrtI0)
        alg = 'ER'
    else:
        rho = DM(rho, S, sqrtI0)
        alg = 'DM'

    rho = symmetrize(rho)

    if ((i+1) % 1e6 == 0):
        S *= 0
        rhomag = np.abs(rho)
        S.flat[rhomag.flat >= np.percentile(rhomag.flat, solv_frac*100)] = 1

    Is = np.abs(fftn(PS(rho, S)))**2
    RI = np.sqrt(np.sum((Is.flat[1:]-I0.flat[1:])**2) / np.sum(I0.flat[1:]**2))
    errorsI[i] = RI

    # rhoe = PM(rho, S)
    # Rrho = np.sqrt(np.sum((np.abs(rhoe)-np.abs(rho0))**2)/np.sum(np.abs(rho0)**2))
    # errorsrho[i] = Rrho

    # pmsg = msg; msg = "Iteration %4d (%s; RI=%7.2g, Rrho=%7.2g)" % (i, alg, RI, Rrho)
    pmsg = msg; msg = "Iteration %4d (%s; RI=%7.2g)" % (i, alg, RI)
    sys.stdout.write('\b'*len(pmsg) + msg); sys.stdout.flush()
    pmsg = msg

print('')
delT = time() - t
print('Total time (s): %g ; Time per iteration (s): %g' % (delT, delT / float(n_iter)))

pg.plot(np.log10(errorsI), title='Log10(errorsI)')
# pg.plot(np.log10(errorsrho), title='Log10(errorsrho)')
imwin = pg.image(np.hstack([np.abs(rho0), np.zeros(rho.shape)[:, 0:int(rho.shape[1]/4), :], np.abs(rho)]), title='rho0')
imwin.setPredefinedGradient('flame')
# pg.image(, title='rho')

print('Note the horizontal slider on the bottom of the window that shows the density maps.')

pg.QtGui.QApplication.instance().exec_()