r"""
Atomic scattering factors
=========================

Notes on how to work with atomic scattering factors.

Contributed by Richard Kirian.

Imports:
"""

import xraylib
import numpy as np
from reborn.simulate import atoms
import matplotlib.pyplot as plt
import scipy.constants as const

# %%
# Constants and configurations

a0 = 5.2917721e-11  # Bohr radius
h = const.h
c = const.c
eV = const.value('electron volt')
save_figures = False

# %%
# There are several numerical approaches to simulating the scattering amplitudes from objects.  At high resolutions,
# atomistic features become important.  A common formula is the following:
#
# .. math::
#
#     I(\vec{q}) = \left| \sum_{n=1}^N f_n(q) \exp(i \vec{q}\cdot \vec{r}_n) \right|^2
#
# If you take the above approach, you'll need the scattering factors :math:`f_n(q)`.  The above expresses the usual
# approximation where all atoms are spherically symmetric, we do not consider the details of bonds or other forms
# of asymmetries since those corrections are hard to detect even if you try.  To first approximation, the scattering
# factors are identical to the electron densities of the atoms, but when we are near resonant conditions, we need to
# include dispersion corrections.  We write the energy- and :math:`q`-dependent scattering factors as
#
# .. math::
#
#    f(q, E) = f_0(q) + f'(E) + i f''(E)
#
# where :math:`f_0(q)` is the electron density, and the other two terms are the dispersion corrections.
#
# At very low resolutions, we only need to know :math:`f(0, E)`, in which case you can use the tables of |Henke1993|.
# A very simple approximation is :math:`f \approx Z`, where :math:`Z` is the atomic number.

# %%
# In many cases, we prefer not to use the direct summation in equation
#
#

# %%
# Let's start by getting the atomic form factor of hydrogen.  We will then Fourier transform the form factor to get the
# electron density.  We can get form factors from the |xraylib| library, which in turn gets the values from
# |Hubbel1975|.  For convenience, reborn provides a few |xraylib| wrappers in order to maintain consistent units and
# interface.  Some of the wrappers for the |xraylib| library have not been vectorized in the ways that are expected
# of numpy-like functions.

z = 1
dr = 0.05e-10  # Sampling in real space
rmax = 10e-10
r = np.arange(0, rmax, dr)
n = r.size
dq = 2 * np.pi / dr / n  # Sampling in reciprocal space
q = np.arange(n) * dq
f = atoms.hubbel_form_factors(q, z)  # Here's the reborn wrapper that gets the xraylib Hubbel form factors
f_h = 16/(a0**2*q**2+4)**2  # Exact FT of hydrogen S1 orbital

# %%
# Let's compare the scattering factor in the Hubbel tables to the exact answer.  There should be no errors, because,
# in fact, the Hubbel tables don't even include hydrogen!  Why would they make the effort to numerically solve for the
# electron density if the exact solution is included in every undergraduate QM textbook?

plt.figure()
plt.loglog(q*1e-10, f_h, lw=4, label='True')
plt.loglog(q*1e-10, f, label='Calc')
plt.xlabel(r'$q = 4\pi \sin(\theta/2)/\lambda \; [{\rm \AA{}}^{-1}]$')
plt.ylabel(r'f(q)')
plt.title(r'Hydrogen Hubbel form factor')
plt.legend()

# %%
# Now let's have a look in real space electron density of the hydrogen atom.  We take the Fourier transform of the
# scattering factor.  The problem here is that the electron density has a cusp at :math:`r = 0`, and this cannot be
# faithfully reproduced by an FT without going to huge values of :math:`q` in the scattering factor :math:`f(q)`.
# The following plot shows what happened when we tried to go from scattering factors to electron densities:

rho = np.imag(np.fft.ifft(q*f))  # This is a 3D Fourier transform of f(q), reduced to 1D due to symmetry
rho[1:] /= np.pi*r[1:]*dr  # Be careful with the divide-by-zero at r=0...
rho[0] = np.sum(q**2*f)*dq/2/np.pi**2  # The correct value at r=0
rho_h = np.exp(-2*r/a0)/np.pi/a0**3  # Exact hydrogen S1 electron density

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

# %%
# As you can see, there are numerical issues at both large and small radii.  In order to access electron densities,
# we will need to find a better way forward, probably by looking up tabulated real-space densities.

#
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
