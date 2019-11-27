from __future__ import (absolute_import, division, print_function, unicode_literals)

import pkg_resources
import numpy as np
from bornagain import utils
from scipy import constants as const

try:
    import xraylib
except ImportError:
    pass

eV = const.value('electron volt')
NA = const.value('Avogadro constant')


henke_data_path = pkg_resources.resource_filename('bornagain', 'data/scatter/henke')


atomic_symbols = np.array(['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
                           'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
                           'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La',
                           'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re',
                           'Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np',
                           'Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg',
                           'Cn','Uut','Uuq','Uup','Uuh','Uus','Uuo'])


def atomic_symbols_to_numbers(symbols):
    r"""
    Convert atomic symbols (such as C, N, He, Be) to atomic numbers.

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
        Z = w[0] + 1

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
    if len(symbols) == 1:
        symbols = symbols[0]
    return symbols


@utils.memoize
def get_henke_data(atomic_number):
    r"""
    Load the `Henke scattering factor data <http://henke.lbl.gov/optical_constants/asf.html>`_ from disk.  The
    Henke data was gathered from the Center for X-Ray Optics web page at the Lawrence Berkeley National Laboratory.

    The data are cached on the first call to this function so that subsequent requests load faster.

    Arguments:
        atomic_number (int): Atomic number

    Returns:
        dict : A dictionary with the following fields
            'Atomic Number' -- Atomic number of the element
            'Element Symbol' -- Standard element symbol (e.g. He)
            'Photon Energy' -- A numpy array of photon energies (SI units)
            'Scatter Factor' -- A numpy array of complex atomic scattering factors that correspond to the photon
            energies above.
    """

    sym = atomic_numbers_to_symbols(atomic_number)
    table = np.loadtxt(henke_data_path + '/' + sym.lower() + '.nff', skiprows=1)
    data = dict()
    data['Atomic Number'] = atomic_number
    data['Element Symbol'] = sym
    data['Photon Energy'] = table[:, 0] * eV
    data['Scatter Factor'] = table[:, 1] + 1j * table[:, 2]
    return data


def get_scattering_factors(atomic_numbers, photon_energy):
    r"""
    Get complex atomic scattering factors for a given photon energy and a range of atomic numbers.  The scattering
    factor data come from the function :func:`bornagain.simulate.atoms.get_henke_data()`.

    Arguments:
        atomic_numbers (int/list-like): Atomic numbers.
        photon_energy (float): Photon energy in SI units.

    Returns:
        scattering_factors (complex/numpy array): Complex scattering factors.
    """

    Z = np.array(atomic_numbers)
    f = np.zeros([len(Z)], dtype=np.complex)
    for z in np.unique(Z):
        dat = get_henke_data(z)
        w = np.where(Z == z)
        f[w] = np.interp(photon_energy, dat['Photon Energy'], dat['Scatter Factor'], left=-9999, right=-9999)
    return f


def get_scattering_factors_fixed_z(atomic_number, photon_energies):
    r"""
    Get complex atomic scattering factors for one atomic number and a range of energies.

    Arguments:
        atomic_number (int): Atomic number.
        photon_energies (float/list-like): Photon energies in SI units.

    Returns:
        scattering_factors (complex/numpy array): Complex scattering factors.
    """

    dat = get_henke_data(atomic_number)
    return np.interp(np.array(photon_energies), dat['Photon Energy'], dat['Scatter Factor'], left=-9999, right=-9999)


def xraylib_scattering_factors(qmags, atomic_number, photon_energy):
    r"""
    Get the q-dependent atomic scattering factors from the xraylib package.

    Args:
        qmags (numpy array):
        atomic_number:
        photon_energy:

    Returns:
    """

    Z = atomic_number
    E = photon_energy/eV/1000
    qq = qmags/4.0/np.pi/1e10
    from xraylib import FF_Rayl, Fi, Fii
    return np.array([FF_Rayl(Z, q) for q in qq]) + Fi(Z, E) - 1j*Fii(Z, E)


def hubbel_atomic_form_factor(qmags, atomic_number, photon_energy):
    r"""
    Fetch the q-dependent atomic form factors.  This allows for an arbitrary list of q magnitudes and returns an array.
    The scattering factors come from Hubbel et al 1975, and are accessed through the xraylib package.

    Args:
        qmags (numpy array):
        atomic_number:
        photon_energy:

    Returns:
        Numpy array
    """

    Z = atomic_number
    E = photon_energy/eV/1000
    qq = qmags/4.0/np.pi/1e10
    f = np.array([xraylib.FF_Rayl(Z, q) for q in qq])
    return f


@utils.memoize
def hubbel_atomic_form_factor_table(atomic_number, photon_energy):
    r"""
    This calls hubbel_atomic_form_factor() and gets a 1D uniformly-spaced sampling for pre-defined q magnitude spacing.

    Args:
        atomic_number:
        photon_energy:

    Returns:
        Numpy array
    """

    Z = atomic_number
    E = photon_energy/eV/1000
    dq = 10e10/1000
    qq = np.arange(0, 10e10, dq)/4.0/np.pi/1e10
    f = np.array([xraylib.FF_Rayl(Z, q) for q in qq])
    return f
