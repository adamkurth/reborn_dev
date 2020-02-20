import xraylib
import numpy as np
from bornagain.simulate import atoms
import matplotlib.pyplot as plt
import scipy.constants as const
h = const.h
c = const.c
eV = const.value('electron volt')

# Checking that xraylib and the Henke tables give (nearly) the same results

# These are in SI units
Z = 79
E = np.arange(1000, 15000, 10)*eV
max_theta = np.pi/2
theta = np.arange(0, max_theta, max_theta / 1000)

f_xraylib = np.zeros((len(E), len(theta)), dtype=np.complex)
f_henke = atoms.get_scattering_factors_fixed_z(Z, E)
for i in range(len(E)):
    this_E = E[i]
    for j in range(len(theta)):
        # xraylib uses units of keV and Angstrom
        this_theta = theta[j]
        lam = h * c / this_E
        this_q = 4 * np.pi * np.sin(this_theta / 2) / lam
        FF = xraylib.FF_Rayl(Z, 1e-10*this_q/4/np.pi)
        Fi = xraylib.Fi(Z, this_E/eV/1000)
        Fii = xraylib.Fii(Z, this_E/eV/1000)
        f_xraylib[i, j] = FF + Fi - 1j*Fii

f = plt.figure()
plt.subplot(121)
plt.semilogy(E/eV, np.real(f_henke), 'r-', label='Henke Real Part')
plt.semilogy(E/eV, np.imag(f_henke), 'r:', label='Henke Imag Part')
plt.semilogy(E/eV, np.real(f_xraylib[:, 0]), 'b-', label='xraylib Real Part')
plt.semilogy(E/eV, np.imag(f_xraylib[:, 0]), 'b:', label='xraylib Real Part')
plt.xlabel('Photon Energy (eV)')
plt.ylabel('Scattering factor at Q=0')
plt.title('Atomic Number %d (%s)' % (Z, atoms.atomic_numbers_to_symbols([Z])))
plt.legend()
# plt.show()
# f.savefig("anamalous_%d.pdf" % (Z,), bbox_inches='tight')

# Check that we are doing the lookup of Henke tables correctly
f_correct = 16.5705 + 1j*2.98532
f_lookup = atoms.get_scattering_factors([29], 798.570*eV)[0]
assert np.abs(f_lookup - f_correct) == 0
f_lookup = atoms.get_scattering_factors_fixed_z(29, np.array([798.570*eV]))[0]
assert np.abs(f_lookup - f_correct) == 0
# Check one of the values from the Hubbel et al. 1975 paper:
assert np.abs(24.461 - xraylib.FF_Rayl(29, 0.175)) == 0

nq = 1000
qmax = 3
qrange = np.arange(nq)*qmax/nq
E = 8
fray = np.array([xraylib.FF_Rayl(Z, q) for q in qrange])
plt.subplot(122)
plt.semilogy(qrange, fray, label='Hubbel')
plt.semilogy(qrange, np.abs(fray + xraylib.Fi(Z, E) - 1j*xraylib.Fii(Z, E)), label='Hubbel w/ xraylib anamalous')
plt.semilogy(qrange, qrange*0+np.abs(atoms.get_scattering_factors_fixed_z(Z, E*1000*eV)), label='Henke')
# plt.semilogy(qrange, np.abs(atoms.xraylib_scattering_factors(qrange*4*np.pi, atomic_number=Z, photon_energy=E*1000*eV)),
#              label='bornagain abs')
plt.semilogy(qrange, np.abs(atoms.hubbel_henke_atomic_scattering_factors(qrange * 4 * np.pi / 1e10, Z, E * 1000 * eV)),
             label='Hubbel form factor, Henke dispersion')
plt.title('Atomic Number %d (%s)' % (Z, atoms.atomic_numbers_to_symbols([Z])))
plt.xlabel(r'$Q = \sin(\theta/2)/\lambda$ [angstrom]')
plt.ylabel(r'|$f(Q)$|: $Q$-dependent scattering factor at 8 keV')
plt.legend()
plt.show()
# f.savefig("formfactor_%d.pdf" % (Z,), bbox_inches='tight')


# nr = 1000
# rmax = 100
# dq = qrange[1] - qrange[0]
# rho = np.zeros(nr)
# for i in range(nr):
#     rho[i] =


