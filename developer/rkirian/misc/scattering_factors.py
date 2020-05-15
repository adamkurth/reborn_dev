import xraylib
import numpy as np
from reborn.simulate import atoms
import matplotlib.pyplot as plt
import scipy.constants as const
h = const.h
c = const.c
eV = const.value('electron volt')

save_figures = False

# # Look up electron densities for a few atoms
# f = plt.figure(figsize=plt.figaspect(0.4))
# for z in np.arange(1)+1:
#     rho, r = atoms.hubbel_density_lut(z)
#     plt.plot(r*1e10, rho*1e-30, label='Z=%d'%z)
# plt.xlim([0, 1])
# plt.xlabel(r'Radius [${\rm \AA{}}$]')
# plt.ylabel(r'Electron density [${\rm \AA{}}^{-3}$]')
# plt.legend()
# plt.show()
# brok

# Check that Fourier transform yields the hydrogen atom wavefunction density
z = 1
dr = 0.05e-10
rmax = 10e-10
r = np.arange(0, rmax, dr)
n = r.size
dq = 2 * np.pi / dr / n
q = np.arange(n) * dq
f = atoms.hubbel_form_factors(q, z)
rho = np.imag(np.fft.ifft(q*f))/np.pi/r/dr
rho[0] = np.sum(q**2*f)*dq/2/np.pi**2
a0 = 5.2917721e-11
rho_h = np.exp(-2*r/a0)/np.pi/a0**3  # hydrogen S1

plt.figure()
plt.loglog(q*1e-10, 16/(a0**2*q**2+4)**2, lw=4, label='True')
plt.loglog(q*1e-10, f, label='Calc')
plt.xlabel(r'$q = 4\pi \sin(\theta/2)/\lambda \; [{\rm \AA{}}^{-1}]$')
plt.ylabel(r'f(q)')
plt.title(r'Hydrogen Hubbel form factor')
plt.legend()
f = plt.figure(figsize=plt.figaspect(0.4))
plt.subplot(121)
plt.plot(r*1e10, rho_h*1e-30, lw=4, label='True')
plt.plot(r*1e10, rho*1e-30, label='Calc')
plt.xlim((0, 2))
plt.xlabel(r'$r$ [${\rm \AA{}}$]')
plt.ylabel(r'$\rho(r)$ [${\rm \AA{}}^{-3}$]')
plt.title(r'Hydrogen S1 electron density')
plt.legend()
plt.subplot(122)
# plt.plot(r*1e10, rho_h*1e-30, lw=4, label='True')
# plt.plot(r*1e10, rho*1e-30, label='Calc')
# plt.xlabel(r'$r$ [${\rm \AA{}}$]')
# plt.ylabel(r'$\rho(r)$ [${\rm \AA{}}^{-3}$]')
# plt.title(r'Hydrogen S1 electron density')
# plt.legend()
# plt.figure()
plt.loglog(r*1e10, rho_h*1e-30, lw=4, label='True')
plt.loglog(r*1e10, rho*1e-30, label='Calc')
plt.xlabel(r'$r$ [${\rm \AA{}}$]')
plt.ylabel(r'$\rho(r)$ [${\rm \AA{}}^{-3}$]')
plt.title(r'Hydrogen S1 electron density')
plt.legend()
if save_figures:
    f.savefig("../notes/scatter/figures/hydrogen_density_1.pdf", bbox_inches='tight')

# Checking that xraylib and the Henke tables give (nearly) the same results

# These are in SI units
z = 79
E = np.arange(1000, 15000, 10)*eV
max_theta = np.pi/2
theta = np.arange(0, max_theta, max_theta / 1000)

f_xraylib = np.zeros((len(E), len(theta)), dtype=np.complex)
f_henke = atoms.henke_scattering_factors(z, E)
for i in range(len(E)):
    this_E = E[i]
    for j in range(len(theta)):
        # xraylib uses units of keV and Angstrom
        this_theta = theta[j]
        lam = h * c / this_E
        this_q = 4 * np.pi * np.sin(this_theta / 2) / lam
        FF = xraylib.FF_Rayl(z, 1e-10 * this_q / 4 / np.pi)
        Fi = xraylib.Fi(z, this_E / eV / 1000)
        Fii = xraylib.Fii(z, this_E / eV / 1000)
        f_xraylib[i, j] = FF + Fi - 1j*Fii

f = plt.figure(figsize=plt.figaspect(0.4))
plt.subplot(121)
plt.semilogy(E/eV, np.real(f_henke), 'r-', label='Henke Real Part')
plt.semilogy(E/eV, np.imag(f_henke), 'r:', label='Henke Imag Part')
plt.semilogy(E/eV, np.real(f_xraylib[:, 0]), 'b-', label='xraylib Real Part')
plt.semilogy(E/eV, np.imag(f_xraylib[:, 0]), 'b:', label='xraylib Real Part')
plt.xlabel('Photon Energy [eV]')
plt.ylabel('Scattering factor at q=0')
plt.title('Atomic Number %d (%s)' % (z, atoms.atomic_numbers_to_symbols([z])))
plt.legend()

q_mags = 1e10*4*np.pi*np.arange(1000)*3/1000
E = 8000*eV
f_henke = np.zeros_like(q_mags)+np.abs(atoms.henke_scattering_factors(z, E))
f_hubbel = atoms.hubbel_form_factors(q_mags, z)  # Hubbel atomic form factors from xraylib
f_xraylib = atoms.xraylib_scattering_factors(q_mags, z, E)  # Hubbel form factors with xraylib dispersion
f_hubbel_henke = atoms.hubbel_henke_scattering_factors(q_mags, z, E)

plt.subplot(122)
plt.semilogy(q_mags/1e10, np.abs(f_hubbel), label='Hubbel')
plt.semilogy(q_mags/1e10, np.abs(f_xraylib), label='xraylib')
plt.semilogy(q_mags/1e10, np.abs(f_henke), label='Henke')
plt.semilogy(q_mags/1e10, np.abs(f_hubbel_henke), label='Hubbel/Henke')
plt.title('Atomic Number %d (%s)' % (z, atoms.atomic_numbers_to_symbols([z])))
plt.xlabel(r'$q = 4\pi \sin(\theta/2)/\lambda$ [${\rm \AA{}}^{-1}$]')
plt.ylabel(r'|$f(q)$|: $q$-dependent scattering factor at 8 keV')
plt.legend()
if save_figures:
    f.savefig("../notes/scatter/figures/formfactor_%d.pdf" % (z,), bbox_inches='tight')
plt.show()
