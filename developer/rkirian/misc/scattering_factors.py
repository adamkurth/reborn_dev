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
f_henke = atoms.henke_scattering_factors(Z, E)
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


q_mags = 1e10*4*np.pi*np.arange(1000)*3/1000
E = 8000*eV
f_henke = np.zeros_like(q_mags)+np.abs(atoms.henke_scattering_factors(Z, E))
f_hubbel = atoms.hubbel_form_factors(q_mags, Z)  # Hubbel atomic form factors from xraylib
f_xraylib = atoms.xraylib_scattering_factors(q_mags, Z, E)  # Hubbel form factors with xraylib dispersion
f_hubbel_henke = atoms.hubbel_henke_scattering_factors(q_mags, Z, E)

plt.subplot(122)
plt.semilogy(q_mags/1e10, np.abs(f_hubbel), label='Hubbel')
plt.semilogy(q_mags/1e10, np.abs(f_xraylib), label='Hubbel w/ xraylib anamalous')
plt.semilogy(q_mags/1e10, np.abs(f_henke), label='Henke')
plt.semilogy(q_mags/1e10, np.abs(f_hubbel_henke), label='Hubbel form factor, Henke dispersion')
plt.title('Atomic Number %d (%s)' % (Z, atoms.atomic_numbers_to_symbols([Z])))
plt.xlabel(r'$Q = 4\pi \sin(\theta/2)/\lambda$ [\AA{} angstrom]')
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


