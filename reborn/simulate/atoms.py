from __future__ import (absolute_import, division, print_function, unicode_literals)

import pkg_resources
import numpy as np
from .. import utils
from scipy import constants as const
import xraylib


eV = const.value('electron volt')
NA = const.value('Avogadro constant')

henke_data_path = pkg_resources.resource_filename('reborn', 'data/scatter/henke')

atomic_symbols = np.array(['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
                           'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
                           'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La',
                           'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re',
                           'Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np',
                           'Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
                           'Cn','Uut','Uuq','Uup','Uuh','Uus','Uuo'])


def atomic_symbols_to_numbers(symbols):
    r"""
    Convert atomic symbol strings (such as C, N, He, Be) to atomic numbers.

    Arguments:
        symbols (string/list-like): Atomic symbol strings. Case insensitive.

    Returns:
        numbers (numpy array): Atomic numbers.
    """
    symbols = np.array(symbols)
    if symbols.ndim > 0:
        Z = np.zeros([len(symbols)])
        U = np.unique(symbols)
        for u in U:
            w = np.nonzero(atomic_symbols == u.capitalize())
            Z[symbols == u] = w[0] + 1
        Z = Z.astype(np.int)
    else:
        w = np.nonzero(atomic_symbols == np.str_(symbols).capitalize())
        Z = int(w[0] + 1)

    return Z


def atomic_numbers_to_symbols(numbers):
    r"""
    Convert atomic numbers to atomic symbols (such as C, N, He, Ca)

    Arguments:
        numbers (int/list-like): Atomic numbers

    Returns:
        symbols (numpy string array): Atomic element symbols (such as H, He, Be)
    """

    numbers = np.array(numbers, dtype=np.int)
    symbols = atomic_symbols[numbers - 1]
    return symbols


@utils.memoize
def _get_henke_data(atomic_number):
    r"""
    Load the `Henke scattering factor data <http://henke.lbl.gov/optical_constants/asf.html>`_ from disk.  The
    Henke data are complex numbers, and the data are available from the Center for X-Ray Optics web page at LBNL.

    Note:
        The data are cached on the first call to this function so that subsequent requests load faster.

    Arguments:
        atomic_number (int): Atomic number

    Returns:
        dict : Dictionary with populated fields 'Atomic Number', 'Element Symbol', 'Photon Energy', 'Scatter Factor'
    """

    sym = atomic_numbers_to_symbols(atomic_number)
    table = np.loadtxt(henke_data_path + '/' + sym.lower() + '.nff', skiprows=1)
    data = dict()
    data['Atomic Number'] = atomic_number
    data['Element Symbol'] = sym
    data['Photon Energy'] = table[:, 0] * eV
    data['Scatter Factor'] = table[:, 1] + 1j * table[:, 2]
    return data


def henke_scattering_factors(atomic_numbers, photon_energies):
    r"""
    Get complex atomic scattering factors from Henke tables.

    Note: The inputs are converted to numpy arrays, and the output is a squeezed 2D array of shape
    (atomic_numbers.size, photon_energies.size).

    Arguments:
        atomic_numbers (int/list-like): Atomic numbers.
        photon_energies (float/list-like): Photon energy in SI units.

    Returns:
        numpy array: Complex scattering factors.
    """
    Z = np.array(atomic_numbers).ravel()
    E = np.array(photon_energies).ravel()
    f = np.zeros([Z.size, E.size], dtype=np.complex).reshape([Z.size, E.size])
    # Yes the reshape above shouldn't be necessary, obviously, but this is a quirky feature of numpy...
    for z in np.unique(Z):
        dat = _get_henke_data(z)
        w = np.where(Z == z)
        f[w, :] = np.interp(E, dat['Photon Energy'], dat['Scatter Factor'], left=-9999, right=-9999)
    return np.squeeze(f)


def get_scattering_factors(atomic_numbers, photon_energy):
    r"""
    Get complex atomic scattering factors (from Henke tables) for a single photon energy and a range of atomic numbers.
    The scattering factor data come from the function :func:`get_henke_data <reborn.simulate.atoms.get_henke_data>`.
    See the function :func:`get_scattering_factors_fixed_z` if you have a range of photon energies and one atomic number.

    Arguments:
        atomic_numbers (int/list-like): Atomic numbers.
        photon_energy (float): Photon energy in SI units.

    Returns:
        numpy array: Complex scattering factors.
    """

    Z = np.array(atomic_numbers).ravel()
    f = np.zeros([len(Z)], dtype=np.complex)
    for z in np.unique(Z):
        dat = _get_henke_data(z)
        w = np.where(Z == z)
        f[w] = np.interp(photon_energy, dat['Photon Energy'], dat['Scatter Factor'], left=-9999, right=-9999)
    return f


def get_scattering_factors_fixed_z(atomic_number, photon_energies):
    r"""
    Get complex atomic scattering factors (from Henke tables) for a single atomic number and a range of photon energies.

    Arguments:
        atomic_number (int): Atomic number.
        photon_energies (float/list-like): Photon energies in SI units.

    Returns:
        scattering_factors (complex/numpy array): Complex scattering factors.
    """

    utils.depreciate('get_scattering_factors_fixed_z is depreciated.  Use henke_scattering_factors instead.')
    dat = _get_henke_data(atomic_number)
    return np.interp(np.array(photon_energies), dat['Photon Energy'], dat['Scatter Factor'], left=-9999, right=-9999)


def xraylib_scattering_factors(qmags, atomic_number, photon_energy):
    r"""
    Get the q-dependent atomic scattering factors from the xraylib package.  The scattering factor is equal to

    :math:`\text{FF_Rayl}(Z, q) + \text{Fi}(Z, E) - i \text{Fii}(Z, E)`

    where FF_Rayl, Fi, and Fii are functions in the xraylib package.  Note the strange minus sign in this formula, which
    is not a typo -- it has been difficult to find documentation on these functions.  Note that the dispersion
    relations in the Henke tables appear to be better than those in xraylib. See
    :func:`hubbel_henke_atomic_scattering_factors` for a better alternative.  Note that this function can be slow
    because the loops over q magnitudes are on the Python level because of the Python wrapper included with xraylib.

    Args:
        qmags (numpy array):  q-vector magnitudes, where :math:`\vec{q} = \frac{2\pi}{\lambda}(\vec{s}-\vec{s}_0)`
        atomic_number (int):  The atomic number.
        photon_energy (float):  The photon energy in SI units.

    Returns:
        numpy array: complex scattering factors as a function of q
    """

    Z = atomic_number
    E = photon_energy/eV/1000
    qq = qmags/4.0/np.pi/1e10
    from xraylib import FF_Rayl, Fi, Fii
    return np.array([FF_Rayl(Z, q) for q in qq]) + Fi(Z, E) - 1j*Fii(Z, E)


def hubbel_form_factors(qmags, atomic_number):
    r"""
    Get the q-dependent atomic form factors.  This allows for an arbitrary list of q magnitudes and returns an array.
    The scattering factors come from Hubbel et al 1975, and are accessed through the xraylib package.

    Args:
        qmags (numpy array):  q vector magnitudes.
        atomic_number (int):  Atomic number.

    Returns:
        Numpy array : Atomic form factor :math:`f(q)`
    """

    qq = np.array(qmags).ravel()  # In case input is just  scalar
    qq = qq*1e-10/4.0/np.pi  # xraylib is in inv. angstrom units without the 4 pi
    f = np.array([xraylib.FF_Rayl(atomic_number, q) for q in qq])
    return f


# #@utils.memoize
# def hubbel_density_lut(atomic_number, dr=0.01e-10, rmax=20e-10):
#     r"""
#     This is experimental.  Do not use it.
#
#     Return the radial profile of the atomic electron density as derived from the inverse transform of the Hubbel
#     form factors.  This function should not exist because the Hubbel form factors were generated from electron
#     densities in the first place, so we've gone full circle... but this is what we have for the time being.
#
#     Args:
#         atomic_number (int): Atomic number
#         dr (float): Step size in the radius grid
#         rmax (float): Maximum radius
#
#     Returns:
#
#         2-element tuple with the following entries
#
#         - **rho** (*numpy array*) : Numpy array of electron densities (in SI units of course)
#         - **r** (*numpy array*) : Numpy array of radii corresponding to densities
#     """
#     z = int(atomic_number)
#
#     r = np.linspace(0, rmax, int(np.ceil(rmax/dr)))
#
#     # if z == 1:
#     #     a0 = 5.2917721e-11
#     #     rho = np.exp(-2 * r / a0) / np.pi / a0 ** 3  # hydrogen S1
#     #     return rho, r
#
#     n = r.size
#     dq = 2 * np.pi / dr / n
#     q = np.arange(n) * dq
#     f = hubbel_form_factors(q, z)
#     rho = np.imag(np.fft.ifft(q * f))
#     rho[1:] /= np.pi * r[1:] * dr
#     rho[0] = np.sum(q ** 2 * f) * dq / 2 / np.pi ** 2
#
#     return rho, r
#
#
# def hubbel_density(atomic_number, radii):
#     r"""
#     This is experimental.  Do not use it.
#
#     Return the radial profile of the atomic electron density as derived from the inverse transform of the Hubbel
#     form factors.  This function should not exist because the Hubbel form factors were generated from electron
#     densities in the first place, so we've gone full circle... but this is what we have for the time being.
#
#     Note that this function uses hubbel_density_lut to generate the atomic densities once on a regular grid, and
#     thereafter this function will interpolate from that standard grid.
#
#     Args:
#         - atomic_number (int) : Atomic number
#         - radii (numpy array) : Radii at which to calculate electron density
#     Returns:
#         - **rho** (*numpy array*) : Electron densities corresponding to input radii
#     """
#     rho, r = hubbel_density_lut(atomic_number)
#     return np.interp(radii, r, rho)


def hubbel_henke_scattering_factors(qmags, atomic_number, photon_energy):
    r"""
    Get the q-dependent atomic form factors for a single atomic number and single photon energy, using the Hubbel atomic
    form factors and the Henke dispersion corrections.  This is what most people should use if they want q-dependent
    scattering factors with dispersion included.

    Args:
        qmags (numpy array):  q vector magnitudes.
        atomic_number (int):  Atomic number.
        photon_energy (float):  Photon energy.

    Returns:
        Numpy array: Atomic form factor :math:`f(q)` with dispersion corrections
    """
    f0 = hubbel_form_factors(qmags, atomic_number)
    df = henke_scattering_factors(atomic_number, photon_energy) - atomic_number
    return f0 + df
