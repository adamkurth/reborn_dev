r"""
Resources for working with atoms: atomic scattering factors, electron densities.
"""

import pkg_resources
import numpy as np
from reborn import utils
from scipy import constants as const

r_e = const.value('classical electron radius')
eV = const.value('electron volt')
NA = const.value('Avogadro constant')
h = const.h
c = const.c

henke_data_path = pkg_resources.resource_filename('reborn', 'data/scatter/henke')


atomic_symbols = np.array(['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',    # noqa
    'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',       # noqa
    'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',     # noqa
    'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl',     # noqa
    'Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No',     # noqa
    'Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Uut','Uuq','Uup','Uuh','Uus','Uuo'])                      # noqa

atomic_weights = np.array([1.68e-27,6.64e-27,1.15e-26,1.5e-26,1.8e-26,1.99e-26,2.33e-26,2.66e-26,3.16e-26,3.35e-26,
                           3.82e-26,4.04e-26,4.48e-26,4.66e-26,5.14e-26,5.33e-26,5.89e-26,6.63e-26,6.49e-26,6.66e-26,
                           7.47e-26,7.95e-26,8.46e-26,8.64e-26,9.12e-26,9.27e-26,9.79e-26,9.75e-26,1.06e-25,1.09e-25,
                           1.16e-25,1.21e-25,1.24e-25,1.31e-25,1.33e-25,1.39e-25,1.42e-25,1.45e-25,1.48e-25,1.51e-25,
                           1.54e-25,1.59e-25,1.64e-25,1.68e-25,1.71e-25,1.77e-25,1.79e-25,1.87e-25,1.91e-25,1.97e-25,
                           2.02e-25,2.12e-25,2.11e-25,2.18e-25,2.21e-25,2.28e-25,2.31e-25,2.33e-25,2.34e-25,2.4e-25,
                           2.44e-25,2.5e-25,2.52e-25,2.61e-25,2.64e-25,2.7e-25,2.74e-25,2.78e-25,2.78e-25,2.87e-25,
                           2.91e-25,2.96e-25,3e-25,3.05e-25,3.09e-25,3.16e-25,3.19e-25,3.24e-25,3.27e-25,3.33e-25,
                           3.39e-25,3.44e-25,3.47e-25,3.47e-25,3.49e-25,3.69e-25,3.7e-25,3.75e-25,3.77e-25,3.85e-25,
                           3.84e-25,3.95e-25,3.94e-25,3.97e-25,4.04e-25,4.1e-25,4.13e-25,4.17e-25,4.22e-25])


def atomic_symbols_to_numbers(symbols):
    r"""
    Convert atomic symbol strings (such as C, N, He, Be) to atomic numbers.

    Arguments:
        symbols (string/list-like): Atomic symbol strings. Case insensitive.

    Returns:
        numbers (|ndarray|): Atomic numbers.
    """
    symbols = np.array(symbols)
    if symbols.ndim > 0:
        z = np.zeros([len(symbols)])
        z_u = np.unique(symbols)
        for u in z_u:
            w = np.nonzero(atomic_symbols == u.capitalize())
            z[symbols == u] = w[0] + 1
        z = z.astype(np.int64)
    else:
        w = np.nonzero(atomic_symbols == np.str_(symbols).capitalize())
        z = int(w[0] + 1)

    return z


def atomic_numbers_to_symbols(numbers):
    r"""
    Convert atomic numbers to atomic symbols (such as C, N, He, Ca)

    Arguments:
        numbers (|ndarray|): Atomic numbers

    Returns:
        |ndarray|: Atomic element symbols (such as H, He, Be)
    """

    numbers = np.array(numbers, dtype=np.int64)
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
        |ndarray|: Complex scattering factors.
    """
    atomic_numbers = np.array(atomic_numbers).ravel()
    photon_energies = np.array(photon_energies).ravel()
    f = np.zeros([atomic_numbers.size, photon_energies.size], dtype=complex).reshape([atomic_numbers.size,
                                                                                         photon_energies.size])
    # Yes the reshape above shouldn't be necessary, obviously, but this is a quirky feature of numpy...
    for z in np.unique(atomic_numbers):
        dat = _get_henke_data(z)
        w = np.where(atomic_numbers == z)
        f[w, :] = np.interp(photon_energies, dat['Photon Energy'], dat['Scatter Factor'], left=-9999, right=-9999)
    return np.squeeze(f)


def henke_dispersion_corrections(atomic_numbers, photon_energies):
    r"""
    Same as henke_scattering_factors but subtracts the atomic number
    Args:
        atomic_numbers:
        photon_energies:

    Returns:

    """
    f = henke_scattering_factors(atomic_numbers, photon_energies)
    f =  f.T
    f -= atomic_numbers
    return f.T.copy()


def get_scattering_factors(atomic_numbers, photon_energy):
    r"""
    Get complex atomic scattering factors (from Henke tables) for a single photon energy and a range of atomic numbers.
    The scattering factor data come from the function :func:`get_henke_data <reborn.target.atoms.get_henke_data>`.
    See the function :func:`get_scattering_factors_fixed_z` if you have a range of photon energies and one atomic
    number.

    Arguments:
        atomic_numbers (int/list-like): Atomic numbers.
        photon_energy (float): Photon energy in SI units.

    Returns:
        |ndarray|: Complex scattering factors.
    """

    atomic_numbers = np.array(atomic_numbers).ravel()
    f = np.zeros([len(atomic_numbers)], dtype=complex)
    for z in np.unique(atomic_numbers):
        dat = _get_henke_data(z)
        w = np.where(atomic_numbers == z)
        f[w] = np.interp(photon_energy, dat['Photon Energy'], dat['Scatter Factor'], left=-9999, right=-9999)
    return f


def get_scattering_factors_fixed_z(atomic_number, photon_energies):
    r"""
    Get complex atomic scattering factors (from Henke tables) for a single atomic number and a range of photon energies.

    Arguments:
        atomic_number (int): Atomic number.
        photon_energies (float/list-like): Photon energies in SI units.

    Returns:
        |ndarray|: Complex scattering factors.
    """

    utils.depreciate('get_scattering_factors_fixed_z is depreciated.  Use henke_scattering_factors instead.')
    dat = _get_henke_data(atomic_number)
    return np.interp(np.array(photon_energies), dat['Photon Energy'], dat['Scatter Factor'], left=-9999, right=-9999)


def xraylib_scattering_factors(q_mags, atomic_number, photon_energy):
    r"""
    Get the q-dependent atomic scattering factors from the xraylib package.  The scattering factor is equal to

    :math:`\text{FF_Rayl}(Z, q) + \text{Fi}(Z, E) - i \text{Fii}(Z, E)`

    where FF_Rayl, Fi, and Fii are functions in the xraylib package.  Note the strange minus sign in this formula, which
    is not a typo -- it has been difficult to find documentation on these functions.  Note that the dispersion
    relations in the Henke tables appear to be better than those in xraylib. See
    :func:`hubbel_henke_atomic_scattering_factors` for a better alternative.  Note that this function can be slow
    because the loops over q magnitudes are on the Python level because of the Python wrapper included with xraylib.

    Args:
        q_mags (|ndarray|):  q-vector magnitudes, where :math:`\vec{q} = \frac{2\pi}{\lambda}(\vec{s}-\vec{s}_0)`
        atomic_number (int):  The atomic number.
        photon_energy (float):  The photon energy in SI units.

    Returns:
        |ndarray|: Complex scattering factors :math:`f(q)`
    """
    photon_energy = photon_energy/eV/1000
    qq = q_mags / 4.0 / np.pi / 1e10
    from xraylib import FF_Rayl, Fi, Fii
    f0 = np.array([FF_Rayl(atomic_number, q) for q in qq])
    fi = Fi(atomic_number, photon_energy)
    fii = Fii(atomic_number, photon_energy)
    return f0 + fi - 1j*fii


def xraylib_refractive_index(compound='H2O', density=1000, photon_energy=10000*eV, beam=None, approximate=False):
    r"""
    Get the x-ray refractive index given a chemical compound, density, and photon energy.

    Arguments:
        compound (str): Chemical compound formula (e.g. H20)
        density (float): Mass density in SI
        photon_energy (float): Photon energy in SI
        beam (|Beam|): For convenience, an alternative to photon_energy
        approximate (bool): Approximate with the non-resonant, high-frequency limit (Equation 1.1 of |Guinier|)
                            (Default: False)

    Returns:
        float: Refractive index
    """
    import xraylib
    if beam is not None:
        photon_energy = beam.photon_energy
    cmp = xraylib.CompoundParser(compound)
    MM = cmp['molarMass']  # g/mol
    N = NA * density / (MM / 1000)  # Number density of molecules (SI)
    n = 0  # Refractive index
    E_keV = photon_energy / (1000 * eV)  # This is energy in **keV**, the default for xraylib
    ne = 0  # Number of electrons per molecule
    for i in range(cmp['nElements']):
        Z = cmp['Elements'][i]
        nZ = cmp['nAtoms'][i]
        ne += Z*nZ
        # mf = cmp['massFractions'][i]
        f = xraylib.FF_Rayl(Z, 0) + xraylib.Fi(Z, E_keV) - 1j * xraylib.Fii(Z, E_keV)
        n += N * f * nZ
    n = 1 - (n * (h*c/photon_energy) ** 2 * r_e / (2 * np.pi))
    if approximate:
        print(N*ne)
        n = 1 - r_e * N * ne * (h*c/photon_energy) ** 2 / (2 * np.pi)
    return n


def xraylib_scattering_density(compound='H2O', density=1000, photon_energy=10000*eV, beam=None, approximate=False):
    r"""
    Calculate scattering density of compound given density and photon energy.  This is approximately equal to the
    electron density far from resonance.

    Args:
        compound (str): Chemical compound formula (e.g. H20)
        density (float): Mass density in SI
        photon_energy (float): Photon energy in SI
        approximate (bool): Approximate with the non-resonant, high-frequency limit (Equation 1.1 of |Guinier|)
                            (Default: False)

    Returns:
        float: Scattering density
    """
    if beam is not None:
        photon_energy = beam.photon_energy
    n = xraylib_refractive_index(compound=compound, density=density, photon_energy=photon_energy, beam=beam,
                                 approximate=approximate)
    rho = (1 - n)*2*np.pi/(h*c/photon_energy)**2/r_e
    return rho


def hubbel_form_factors(q_mags, atomic_number):
    r"""
    Get the q-dependent atomic form factors.  This allows for an arbitrary list of q magnitudes and returns an array.
    The scattering factors come from Hubbel et al 1975, and are accessed through the xraylib package.

    Args:
        q_mags (|ndarray|):  q vector magnitudes.
        atomic_number (int):  Atomic number.

    Returns:
        |ndarray| : Atomic form factor :math:`f(q)`
    """
    import xraylib
    qq = np.array(q_mags).ravel()  # In case input is just  scalar
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


def hubbel_henke_scattering_factors(q_mags, atomic_number, photon_energy):
    r"""
    Get the q-dependent atomic form factors for a single atomic number and single photon energy, using the Hubbel atomic
    form factors and the Henke dispersion corrections.  This is what most people should use if they want q-dependent
    scattering factors with dispersion included.

    Args:
        q_mags (|ndarray|):  q vector magnitudes.
        atomic_number (int):  Atomic number.
        photon_energy (float):  Photon energy.

    Returns:
        |ndarray|: Atomic form factor :math:`f(q)` with dispersion corrections
    """
    f0 = hubbel_form_factors(q_mags, atomic_number)
    df = henke_scattering_factors(atomic_number, photon_energy) - atomic_number
    return f0 + df


def cromer_mann_scattering_factors(q_mags, atomic_numbers):
    r"""
    Get Cromer-Mann scattering factors (|Cromer1968|) for a range of atomic numbers and q magnitudes.  The Cromer-Mann
    formula is

    .. math::

        f(q) = c + \sum_{i=1}^4 a_i \exp[-b_i q^2 / 16\pi^2] \\

    Note: The Cromer-Mann tables contain the :math:`a_i` and :math:`b_i` coefficients for ions, but this function only
    works with neutral atoms for simplicity.  You may use :func:`get_cmann_form_factors` if you want to work with
    ions.

    Args:
        atomic_numbers (|ndarray|): Array of atomic numbers.  Will be converted to 1D |ndarray| of int type.
        q_mags (|ndarray|): Array of q magnitudes.  Will be converted to 1D |ndarray| or floats.

    Returns:
        |ndarray|: Atomic scattering factor array with shape (len(atomic_numbers), len(q_mags))
    """
    atomic_numbers = np.atleast_1d(atomic_numbers).ravel().astype(int)
    q2 = np.atleast_1d(q_mags).ravel()**2/16/np.pi**2/1e20
    out = np.zeros((atomic_numbers.size, q_mags.size))
    for i in range(len(atomic_numbers)):
        z = atomic_numbers[i]
        if (z > 92) or (z < 1):
            raise ValueError('Atomic number is out of range.')
        a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c = _cmann_coeffs_neutral[z-1, :]
        out[i, :] = c + a_1*np.exp(-b_1*q2) + a_2*np.exp(-b_2*q2) + a_3*np.exp(-b_3*q2) + a_4*np.exp(-b_4*q2)
    out = np.squeeze(out)
    return out


def get_cmann_form_factors(cman, q):
    r"""
    Generates the form factors :math:`f_0(q)` from the model of |Cromer1968|.  Needs documentation - ask Derek Mendez.
    If you want to work with neutral atoms, use :func:`cromer_mann_scattering_factors` instead.

    Arguments:
        cman (dict): Dictionary containing {atomic_number : array of cman parameters}
        q (|ndarray|): Either an array of vectors :math:`\vec{q}` with shape (N,v3), or an array of :math:`q`
                       magnitudes.

    Returns:
        Ask Derek Mendez.
    """

    form_facts = {}

    q = np.atleast_1d(np.array(q))

    if len(q.shape) == 1:
        q_mag_squared = q**2
    elif len(q.shape) == 2:
        q_mag_squared = np.sum(q**2, 1)
    else:
        raise ValueError('get_cmann_form_factors: q must be a 1D array or 2D array of shape Nx3')

    expo_terms = np.exp(-q_mag_squared / (16 * np.pi*np.pi))

    for z, coefs in cman.iteritems():  # Does this actually work?
        a_vals = coefs[:4]
        b_vals = coefs[4:8]
        c = coefs[-1]
#       make the form factors for each q for atomic number z
        z_ff = np.sum([a * (expo_terms**b) for a, b in zip(a_vals, b_vals)], 0)
        z_ff += c
        form_facts[z] = z_ff
    return form_facts


@utils.memoize
def get_cromermann_parameters(atomic_numbers=None):
    r"""
    Get Cromer-Mann parameters for each atom type.

    Modified from tjlane/thor.git by Derek Mendez.

    Arguments:
        atomic_numbers (|ndarray|, int) : A numpy array of the atomic numbers of each atom in the system.

    Returns:
        dict : The Cromer-Mann parameters for the system. The dictionary key corresponds to the atomic number, and the
               parameters are listed in order as a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c
    """
    if atomic_numbers is None:
        atomic_numbers = np.arange(89) + 1
    atom_types = np.unique(atomic_numbers)
    cromermann = {}
    for i, a in enumerate(atom_types):
        try:
            cromermann[a] = np.array(cromer_mann_params[(a, 0)], dtype=np.float32)
        except KeyError:
            print('Element number %d not in Cromer-Mann form factor parameter database' % a)
            raise RuntimeError('Could not get critical parameters for computation')
    return cromermann


def get_cromer_mann_densities(Z, r):

    Z = utils.atleast_1d(Z)
    r = utils.atleast_1d(r)
    out = np.zeros((Z.size, r.size))
    cmann = get_cromermann_parameters()
    for (zi, z) in enumerate(Z):
        if z == 1:
            rho = (1/(np.pi*5.291e-30))*np.exp(-2*r/5.291e-10)
            out[zi, :] = rho
            continue
        a = cmann[z]
        b = a[4:]
        rho = np.zeros(r.size)
        for i in range(4):
            rho += a[i]*8*(np.pi/b[i])**(3/2)*np.exp(-12*np.pi**2*(r*1e10)**2/b[i])
        out[zi, :] = rho
    return out


# def get_cromermann_parameters_legacy(atomic_numbers, max_num_atom_types=None):
#     r"""
#     Get cromer-mann parameters for each atom type and renumber the atom
#     types to 0, 1, 2, ... to point to their CM params.
#
#     Arguments:
#
#         atomic_numbers (|ndarray|, int) :
#             A numpy array of the atomic numbers of each atom in the system.
#
#         max_num_atom_types (int) :
#             The maximium number of atom types allowable
#
#     Returns:
#
#         cromermann (c-array, float) :
#             The Cromer-Mann parameters for the system. Positions [(0-8) * aid] are
#             reserved for each atom type in the system (see `aid` below).
#
#         aid (c-array, int) :
#             The indicies of the atomic ids of each atom in the system. This is an
#             arbitrary compressed index for use within the scattering code. Really
#             this is just a renumbering so that each atom type recieves an index
#             0, 1, 2, ... corresponding to the position of that atom's type in
#             the `cromermann` array.
#     """
#
#     atom_types = np.unique(atomic_numbers)
#     num_atom_types = len(atom_types)
#
#     if max_num_atom_types:
#         if num_atom_types > max_num_atom_types:
#             raise Exception('Fatal Error. Your molecule has too many unique atom  '
#                             'types -- the scattering code cannot handle more due'
#                             ' to code requirements. You can recompile the kernel'
#                             ' to fix this -- see file odin/src/scatter. Email '
#                             'tjlane@stanford.edu complaining about shitty code'
#                             'if you get confused.')
#
#     cromermann = np.zeros(9*num_atom_types, dtype=np.float32)
#     aid = np.zeros(len(atomic_numbers), dtype=np.int32)
#
#     for i, a in enumerate(atom_types):
#         ind = i * 9
#         try:
#             cromermann[ind:ind+9] = np.array(cromer_mann_params[(a, 0)], dtype=np.float32)
#         except KeyError:
#             print('Element number %d not in Cromer-Mann form factor parameter database' % a)
#             raise RuntimeError('Could not get critical parameters for computation')
#         aid[atomic_numbers == a] = np.int32(i)
#
#     return cromermann, aid


# ------------------------------------------------------------------------------
# REFERENCE TABLES (Taken from tjlane/thor.git)
# ------------------------------------------------------------------------------

# CROMER-MANN XRAY CROSS SECTION PARAMETERS
#
# These are the parameters used to calculate the non-anomolous part of the x-ray
# scattering atomic form factor, f0.
#
# ORIGIN: DABAX
# GENERATION OF THIS TABLE: odin/data/cromer-mann.py
#
# Recall the Cromer-Mann formula:
#
#     f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]
#                 i=1,4
#
# where k = sin(theta) / lambda = |q| / (4*pi)
# and c, a_i and b_i are the so called Cromer-Mann coefficients
# (International Tables vol. 4 or vol C; in vol. C refer to pg 500-502)
#
# cromer_mann_params : dict
#     A dictionary such that
#
#     cromer_mann_params[(atomic_number, ionization_state)] =
#         [a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c]
#
#     where `ionization_state` is a
#       -- positive int for a cation
#       -- negative int for an anion
#       -- '.' for a radical


# These are simplified Cromer-Mann coefficients -- only the neutral atoms, and stored as a 95x9 array.
# This is almost always what we want.  The order is [a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c].
_cmann_coeffs_neutral = np.array([
    [0.493002, 0.322912, 0.140191, 0.04081, 10.5109, 26.1257, 3.14236, 57.7997, 0.0030380001],
    [0.8734, 0.6309, 0.3112, 0.178, 9.1037, 3.3568, 22.9276, 0.9821, 0.0063999998],
    [1.1282, 0.7508, 0.6175, 0.4653, 3.9546, 1.0524, 85.3905, 168.261, 0.037700001],
    [1.5919, 1.1278, 0.5391, 0.7029, 43.6427, 1.8623, 103.483, 0.542, 0.0385],
    [2.0545, 1.3326, 1.0979, 0.7068, 23.2185, 1.021, 60.3498, 0.1403, -0.1932],
    [2.31, 1.02, 1.5886, 0.865, 20.8439, 10.2075, 0.5687, 51.6512, 0.2156],
    [12.2126, 3.1322, 2.0125, 1.1663, 0.0057000001, 9.8933, 28.9975, 0.5826, -11.529],
    [3.0485, 2.2868, 1.5463, 0.867, 13.2771, 5.7011, 0.3239, 32.9089, 0.2508],
    [3.5392, 2.6412, 1.517, 1.0243, 10.2825, 4.2944, 0.2615, 26.1476, 0.2776],
    [3.9553, 3.1125, 1.4546, 1.1251, 8.4042, 3.4262, 0.2306, 21.7184, 0.3515],
    [4.7626, 3.1736, 1.2674, 1.1128, 3.285, 8.8422, 0.3136, 129.424, 0.676],
    [5.4204, 2.1735, 1.2269, 2.3073, 2.8275, 79.2611, 0.3808, 7.1937, 0.8584],
    [6.4202, 1.9002, 1.5936, 1.9646, 3.0387, 0.7426, 31.5472, 85.0886, 1.1151],
    [6.2915, 3.0353, 1.9891, 1.541, 2.4386, 32.3337, 0.6785, 81.6937, 1.1407],
    [6.4345, 4.1791, 1.78, 1.4908, 1.9067, 27.157, 0.526, 68.1645, 1.1149],
    [6.9053, 5.2034, 1.4379, 1.5863, 1.4679, 22.2151, 0.2536, 56.172, 0.8669],
    [11.4604, 7.1964, 6.2556, 1.6455, 0.0104, 1.1662, 18.5194, 47.7784, -9.5574],
    [7.4845, 6.7723, 0.6539, 1.6442, 0.9072, 14.8407, 43.8983, 33.3929, 1.4445],
    [8.2186, 7.4398, 1.0519, 0.8659, 12.7949, 0.7748, 213.187, 41.6841, 1.4228],
    [8.6266, 7.3873, 1.5899, 1.0211, 10.4421, 0.6599, 85.7484, 178.437, 1.3751],
    [9.189, 7.3679, 1.6409, 1.468, 9.0213, 0.5729, 136.108, 51.3531, 1.3329],
    [9.7595, 7.3558, 1.6991, 1.9021, 7.8508, 0.5, 35.6338, 116.105, 1.2807],
    [10.2971, 7.3511, 2.0703, 2.0571, 6.8657, 0.4385, 26.8938, 102.478, 1.2199],
    [10.6406, 7.3537, 3.324, 1.4922, 6.1038, 0.392, 20.2626, 98.7399, 1.1832],
    [11.2819, 7.3573, 3.0193, 2.2441, 5.3409, 0.3432, 17.8674, 83.7543, 1.0896],
    [11.0424, 7.374, 4.1346, 0.4399, 4.6538, 0.3053, 12.0546, 31.2809, 1.0097],
    [12.2841, 7.3409, 4.0034, 2.3488, 4.2791, 0.2784, 13.5359, 71.1692, 1.0118],
    [12.8376, 7.292, 4.4438, 2.38, 3.8785, 0.2565, 12.1763, 66.3421, 1.0341],
    [13.338, 7.1676, 5.6158, 1.6735, 3.5828, 0.247, 11.3966, 64.8126, 1.191],
    [14.0743, 7.0318, 5.1652, 2.41, 3.2655, 0.2333, 10.3163, 58.7097, 1.3041],
    [15.2354, 6.7006, 4.3591, 2.9623, 3.0669, 0.2412, 10.7805, 61.4135, 1.7189],
    [16.0816, 6.3747, 3.7068, 3.683, 2.8509, 0.2516, 11.4468, 54.7625, 2.1313],
    [16.6723, 6.0701, 3.4313, 4.2779, 2.6345, 0.2647, 12.9479, 47.7972, 2.531],
    [17.0006, 5.8196, 3.9731, 4.3543, 2.4098, 0.2726, 15.2372, 43.8163, 2.8409],
    [17.1789, 5.2358, 5.6377, 3.9851, 2.1723, 16.5796, 0.2609, 41.4328, 2.9557],
    [17.3555, 6.7286, 5.5493, 3.5375, 1.9384, 16.5623, 0.2261, 39.3972, 2.825],
    [17.1784, 9.6435, 5.1399, 1.5292, 1.7888, 17.3151, 0.2748, 164.934, 3.4873],
    [17.5663, 9.8184, 5.422, 2.6694, 1.5564, 14.0988, 0.1664, 132.376, 2.5064],
    [17.776, 10.2946, 5.72629, 3.26588, 1.4029, 12.8006, 0.125599, 104.354, 1.91213],
    [17.8765, 10.948, 5.41732, 3.65721, 1.27618, 11.916, 0.117622, 87.6627, 2.06929],
    [17.6142, 12.0144, 4.04183, 3.53346, 1.18865, 11.766, 0.204785, 69.7957, 3.75591],
    [3.7025, 17.2356, 12.8876, 3.7429, 0.2772, 1.0958, 11.004, 61.6584, 4.3875],
    [19.1301, 11.0948, 4.64901, 2.71263, 0.864132, 8.14487, 21.5707, 86.8472, 5.40428],
    [18.5003, 13.1787, 4.71304, 2.18535, 0.844582, 8.12534, 0.036495, 20.8504, 1.42357],
    [19.2957, 14.3501, 4.73425, 1.28918, 0.751536, 8.21758, 25.8749, 98.6062, 5.328],
    [19.3319, 15.5017, 5.29537, 0.605844, 0.698655, 7.98929, 25.2052, 76.8986, 5.26593],
    [19.2808, 16.6885, 4.8045, 1.0463, 0.6446, 7.4726, 24.6605, 99.8156, 5.179],
    [19.2214, 17.6444, 4.461, 1.6029, 0.5946, 6.9089, 24.7008, 87.4825, 5.0694],
    [19.1624, 18.5596, 4.2948, 2.0396, 0.5476, 6.3776, 25.8499, 92.8029, 4.9391],
    [19.1889, 19.1005, 4.4585, 2.4663, 5.8303, 0.5031, 26.8909, 83.9571, 4.7821],
    [19.6418, 19.0455, 5.0371, 2.6827, 5.3034, 0.4607, 27.9074, 75.2825, 4.5909],
    [19.9644, 19.0138, 6.14487, 2.5239, 4.81742, 0.420885, 28.5284, 70.8403, 4.352],
    [20.1472, 18.9949, 7.5138, 2.2735, 4.347, 0.3814, 27.766, 66.8776, 4.0712],
    [20.2933, 19.0298, 8.9767, 1.99, 3.9282, 0.344, 26.4659, 64.2658, 3.7118],
    [20.3892, 19.1062, 10.662, 1.4953, 3.569, 0.3107, 24.3879, 213.904, 3.3352],
    [20.3361, 19.297, 10.888, 2.6959, 3.216, 0.2756, 20.2073, 167.202, 2.7731],
    [20.578, 19.599, 11.3727, 3.28719, 2.94817, 0.244475, 18.7726, 133.124, 2.14678],
    [21.1671, 19.7695, 11.8513, 3.33049, 2.81219, 0.226836, 17.6083, 127.113, 1.86264],
    [22.044, 19.6697, 12.3856, 2.82428, 2.77393, 0.222087, 16.7669, 143.644, 2.0583],
    [22.6845, 19.6847, 12.774, 2.85137, 2.66248, 0.210628, 15.885, 137.903, 1.98486],
    [23.3405, 19.6095, 13.1235, 2.87516, 2.5627, 0.202088, 15.1009, 132.721, 2.02876],
    [24.0042, 19.4258, 13.4396, 2.89604, 2.47274, 0.196451, 14.3996, 128.007, 2.20963],
    [24.6274, 19.0886, 13.7603, 2.9227, 2.3879, 0.1942, 13.7546, 123.174, 2.5745],
    [25.0709, 19.0798, 13.8518, 3.54545, 2.25341, 0.181951, 12.9331, 101.398, 2.4196],
    [25.8976, 18.2185, 14.3167, 2.95354, 2.24256, 0.196143, 12.6648, 115.362, 3.58324],
    [26.507, 17.6383, 14.5596, 2.96577, 2.1802, 0.202172, 12.1899, 111.874, 4.29728],
    [26.9049, 17.294, 14.5583, 3.63837, 2.07051, 0.19794, 11.4407, 92.6566, 4.56796],
    [27.6563, 16.4285, 14.9779, 2.98233, 2.07356, 0.223545, 11.3604, 105.703, 5.92046],
    [28.1819, 15.8851, 15.1542, 2.98706, 2.02859, 0.238849, 10.9975, 102.961, 6.75621],
    [28.6641, 15.4345, 15.3087, 2.98963, 1.9889, 0.257119, 10.6647, 100.417, 7.56672],
    [28.9476, 15.2208, 15.1, 3.71601, 1.90182, 9.98519, 0.261033, 84.3298, 7.97628],
    [29.144, 15.1726, 14.7586, 4.30013, 1.83262, 9.5999, 0.275116, 72.029, 8.58154],
    [29.2024, 15.2293, 14.5135, 4.76492, 1.77333, 9.37046, 0.295977, 63.3644, 9.24354],
    [29.0818, 15.43, 14.4327, 5.11982, 1.72029, 9.2259, 0.321703, 57.056, 9.8875],
    [28.7621, 15.7189, 14.5564, 5.44174, 1.67191, 9.09227, 0.3505, 52.0861, 10.472],
    [28.1894, 16.155, 14.9305, 5.67589, 1.62903, 8.97948, 0.382661, 48.1647, 11.0005],
    [27.3049, 16.7296, 15.6115, 5.83377, 1.59279, 8.86553, 0.417916, 45.0011, 11.4722],
    [27.0059, 17.7639, 15.7131, 5.7837, 1.51293, 8.81174, 0.424593, 38.6103, 11.6883],
    [16.8819, 18.5913, 25.5582, 5.86, 0.4611, 8.6216, 1.4826, 36.3956, 12.0658],
    [20.6809, 19.0417, 21.6575, 5.9676, 0.545, 8.4484, 1.5729, 38.3246, 12.6089],
    [27.5446, 19.1584, 15.538, 5.52593, 0.65515, 8.70751, 1.96347, 45.8149, 13.1746],
    [31.0617, 13.0637, 18.442, 5.9696, 0.6902, 2.3576, 8.618, 47.2579, 13.4118],
    [33.3689, 12.951, 16.5877, 6.4692, 0.704, 2.9238, 8.7937, 48.0093, 13.5782],
    [34.6726, 15.4733, 13.1138, 7.02588, 0.700999, 3.55078, 9.55642, 47.0045, 13.677],
    [35.3163, 19.0211, 9.49887, 7.42518, 0.68587, 3.97458, 11.3824, 45.4715, 13.7108],
    [35.5631, 21.2816, 8.0037, 7.4433, 0.6631, 4.0691, 14.0422, 44.2473, 13.6905],
    [35.9299, 23.0547, 12.1439, 2.11253, 0.646453, 4.17619, 23.1052, 150.645, 13.7247],
    [35.763, 22.9064, 12.4739, 3.21097, 0.616341, 3.87135, 19.9887, 142.325, 13.6211],
    [35.6597, 23.1032, 12.5977, 4.08655, 0.589092, 3.65155, 18.599, 117.02, 13.5266],
    [35.5645, 23.4219, 12.7473, 4.80703, 0.563359, 3.46204, 17.8309, 99.1722, 13.4314],
    [35.8847, 23.2948, 14.1891, 4.17287, 0.547751, 3.41519, 16.9235, 105.251, 13.4287],
    [36.0228, 23.4128, 14.9491, 4.188, 0.5293, 3.3253, 16.0927, 100.613, 13.3966],
    [36.1874, 23.5964, 15.6402, 4.1855, 0.511929, 3.25396, 15.3622, 97.4908, 13.3573],
    [35.5103, 22.5787, 12.7766, 4.92159, 0.498626, 2.96627, 11.9484, 22.7502, 13.2116],
    [36.6706, 24.0992, 17.3415, 3.49331, 0.483629, 3.20647, 14.3136, 102.273, 13.3592]
], dtype=np.float32)

cromer_mann_params = {
    'Ac' : [35.6597, 23.1032, 12.5977, 4.08655, 0.589092, 3.65155, 18.599, 117.02, 13.5266],                   # noqa
    'Ac3+' : [35.1736, 22.1112, 8.19216, 7.05545, 0.579689, 3.41437, 12.9187, 25.9443, 13.4637],               # noqa
    'Ag' : [19.2808, 16.6885, 4.8045, 1.0463, 0.6446, 7.4726, 24.6605, 99.8156, 5.179],                        # noqa
    'Ag1+' : [19.1812, 15.9719, 5.27475, 0.357534, 0.646179, 7.19123, 21.7326, 66.1147, 5.21572],              # noqa
    'Ag2+' : [19.1643, 16.2456, 4.3709, 0.0, 0.645643, 7.18544, 21.4072, 0.0, 5.21404],                        # noqa
    'Al' : [6.4202, 1.9002, 1.5936, 1.9646, 3.0387, 0.7426, 31.5472, 85.0886, 1.1151],                         # noqa
    'Al3+' : [4.17448, 3.3876, 1.20296, 0.528137, 1.93816, 4.14553, 0.228753, 8.28524, 0.706786],              # noqa
    'Am' : [36.6706, 24.0992, 17.3415, 3.49331, 0.483629, 3.20647, 14.3136, 102.273, 13.3592],                 # noqa
    'Ar' : [7.4845, 6.7723, 0.6539, 1.6442, 0.9072, 14.8407, 43.8983, 33.3929, 1.4445],                        # noqa
    'As' : [16.6723, 6.0701, 3.4313, 4.2779, 2.6345, 0.2647, 12.9479, 47.7972, 2.531],                         # noqa
    'At' : [35.3163, 19.0211, 9.49887, 7.42518, 0.68587, 3.97458, 11.3824, 45.4715, 13.7108],                  # noqa
    'Au' : [16.8819, 18.5913, 25.5582, 5.86, 0.4611, 8.6216, 1.4826, 36.3956, 12.0658],                        # noqa
    'Au1+' : [28.0109, 17.8204, 14.3359, 6.58077, 1.35321, 7.7395, 0.356752, 26.4043, 11.2299],                # noqa
    'Au3+' : [30.6886, 16.9029, 12.7801, 6.52354, 1.2199, 6.82872, 0.212867, 18.659, 9.0968],                  # noqa
    'B' : [2.0545, 1.3326, 1.0979, 0.7068, 23.2185, 1.021, 60.3498, 0.1403, -0.1932],                          # noqa
    'Ba' : [20.3361, 19.297, 10.888, 2.6959, 3.216, 0.2756, 20.2073, 167.202, 2.7731],                         # noqa
    'Ba2+' : [20.1807, 19.1136, 10.9054, 0.773634, 3.21367, 0.28331, 20.0558, 51.746, 3.02902],                # noqa
    'Be' : [1.5919, 1.1278, 0.5391, 0.7029, 43.6427, 1.8623, 103.483, 0.542, 0.0385],                          # noqa
    'Be2+' : [6.2603, 0.8849, 0.7993, 0.1647, 0.0027000001, 0.8313, 2.2758, 5.1146, -6.1092],                  # noqa
    'Bi' : [33.3689, 12.951, 16.5877, 6.4692, 0.704, 2.9238, 8.7937, 48.0093, 13.5782],                        # noqa
    'Bi3+' : [21.8053, 19.5026, 19.1053, 7.10295, 1.2356, 6.24149, 0.469999, 20.3185, 12.4711],                # noqa
    'Bi5+' : [33.5364, 25.0946, 19.2497, 6.91555, 0.91654, 0.039042, 5.71414, 12.8285, -6.7994],               # noqa
    'Bk' : [36.7881, 24.7736, 17.8919, 4.23284, 0.451018, 3.04619, 12.8946, 86.003, 13.2754],                  # noqa
    'Br' : [17.1789, 5.2358, 5.6377, 3.9851, 2.1723, 16.5796, 0.2609, 41.4328, 2.9557],                        # noqa
    'Br1-' : [17.1718, 6.3338, 5.5754, 3.7272, 2.2059, 19.3345, 0.2871, 58.1535, 3.1776],                      # noqa
    'C' : [2.31, 1.02, 1.5886, 0.865, 20.8439, 10.2075, 0.5687, 51.6512, 0.2156],                              # noqa
    'C.' : [2.26069, 1.56165, 1.05075, 0.839259, 22.6907, 0.656665, 9.75618, 55.5949, 0.286977],               # noqa
    'Ca' : [8.6266, 7.3873, 1.5899, 1.0211, 10.4421, 0.6599, 85.7484, 178.437, 1.3751],                        # noqa
    'Ca2+' : [15.6348, 7.9518, 8.4372, 0.8537, -0.0074, 0.6089, 10.3116, 25.9905, -14.875],                    # noqa
    'Cd' : [19.2214, 17.6444, 4.461, 1.6029, 0.5946, 6.9089, 24.7008, 87.4825, 5.0694],                        # noqa
    'Cd2+' : [19.1514, 17.2535, 4.47128, 0.0, 0.597922, 6.80639, 20.2521, 0.0, 5.11937],                       # noqa
    'Ce' : [21.1671, 19.7695, 11.8513, 3.33049, 2.81219, 0.226836, 17.6083, 127.113, 1.86264],                 # noqa
    'Ce3+' : [20.8036, 19.559, 11.9369, 0.612376, 2.77691, 0.23154, 16.5408, 43.1692, 2.09013],                # noqa
    'Ce4+' : [20.3235, 19.8186, 12.1233, 0.144583, 2.65941, 0.21885, 15.7992, 62.2355, 1.5918],                # noqa
    'Cf' : [36.9185, 25.1995, 18.3317, 4.24391, 0.437533, 3.00775, 12.4044, 83.7881, 13.2674],                 # noqa
    'Cl' : [11.4604, 7.1964, 6.2556, 1.6455, 0.0104, 1.1662, 18.5194, 47.7784, -9.5574],                       # noqa
    'Cl1-' : [18.2915, 7.2084, 6.5337, 2.3386, 0.0066, 1.1717, 19.5424, 60.4486, -16.378],                     # noqa
    'Cm' : [36.6488, 24.4096, 17.399, 4.21665, 0.465154, 3.08997, 13.4346, 88.4834, 13.2887],                  # noqa
    'Co' : [12.2841, 7.3409, 4.0034, 2.3488, 4.2791, 0.2784, 13.5359, 71.1692, 1.0118],                        # noqa
    'Co2+' : [11.2296, 7.3883, 4.7393, 0.7108, 4.1231, 0.2726, 10.2443, 25.6466, 0.9324],                      # noqa
    'Co3+' : [10.338, 7.88173, 4.76795, 0.725591, 3.90969, 0.238668, 8.35583, 18.3491, 0.286667],              # noqa
    'Cr' : [10.6406, 7.3537, 3.324, 1.4922, 6.1038, 0.392, 20.2626, 98.7399, 1.1832],                          # noqa
    'Cr2+' : [9.54034, 7.7509, 3.58274, 0.509107, 5.66078, 0.344261, 13.3075, 32.4224, 0.616898],              # noqa
    'Cr3+' : [9.6809, 7.81136, 2.87603, 0.113575, 5.59463, 0.334393, 12.8288, 32.8761, 0.518275],              # noqa
    'Cs' : [20.3892, 19.1062, 10.662, 1.4953, 3.569, 0.3107, 24.3879, 213.904, 3.3352],                        # noqa
    'Cs1+' : [20.3524, 19.1278, 10.2821, 0.9615, 3.552, 0.3086, 23.7128, 59.4565, 3.2791],                     # noqa
    'Cu' : [13.338, 7.1676, 5.6158, 1.6735, 3.5828, 0.247, 11.3966, 64.8126, 1.191],                           # noqa
    'Cu1+' : [11.9475, 7.3573, 6.2455, 1.5578, 3.3669, 0.2274, 8.6625, 25.8487, 0.89],                         # noqa
    'Cu2+' : [11.8168, 7.11181, 5.78135, 1.14523, 3.37484, 0.244078, 7.9876, 19.897, 1.14431],                 # noqa
    'Dy' : [26.507, 17.6383, 14.5596, 2.96577, 2.1802, 0.202172, 12.1899, 111.874, 4.29728],                   # noqa
    'Dy3+' : [25.5395, 20.2861, 11.9812, 4.50073, 1.9804, 0.143384, 9.34972, 19.581, 0.68969],                 # noqa
    'Er' : [27.6563, 16.4285, 14.9779, 2.98233, 2.07356, 0.223545, 11.3604, 105.703, 5.92046],                 # noqa
    'Er3+' : [26.722, 19.7748, 12.1506, 5.17379, 1.84659, 0.13729, 8.36225, 17.8974, 1.17613],                 # noqa
    'Eu' : [24.6274, 19.0886, 13.7603, 2.9227, 2.3879, 0.1942, 13.7546, 123.174, 2.5745],                      # noqa
    'Eu2+' : [24.0063, 19.9504, 11.8034, 3.87243, 2.27783, 0.17353, 11.6096, 26.5156, 1.36389],                # noqa
    'Eu3+' : [23.7497, 20.3745, 11.8509, 3.26503, 2.22258, 0.16394, 11.311, 22.9966, 0.759344],                # noqa
    'F' : [3.5392, 2.6412, 1.517, 1.0243, 10.2825, 4.2944, 0.2615, 26.1476, 0.2776],                           # noqa
    'F1-' : [3.6322, 3.51057, 1.26064, 0.940706, 5.27756, 14.7353, 0.442258, 47.3437, 0.653396],               # noqa
    'Fe' : [11.7695, 7.3573, 3.5222, 2.3045, 4.7611, 0.3072, 15.3535, 76.8805, 1.0369],                        # noqa
    'Fe+2' : [11.0424, 7.374, 4.1346, 0.4399, 4.6538, 0.3053, 12.0546, 31.2809, 1.0097],                       # noqa
    'Fe3+' : [11.1764, 7.3863, 3.3948, 0.072400004, 4.6147, 0.3005, 11.6729, 38.5566, 0.9707],                 # noqa
    'Fr' : [35.9299, 23.0547, 12.1439, 2.11253, 0.646453, 4.17619, 23.1052, 150.645, 13.7247],                 # noqa
    'Ga' : [15.2354, 6.7006, 4.3591, 2.9623, 3.0669, 0.2412, 10.7805, 61.4135, 1.7189],                        # noqa
    'Ga3+' : [12.692, 6.69883, 6.06692, 1.0066, 2.81262, 0.22789, 6.36441, 14.4122, 1.53545],                  # noqa
    'Gd' : [25.0709, 19.0798, 13.8518, 3.54545, 2.25341, 0.181951, 12.9331, 101.398, 2.4196],                  # noqa
    'Gd3+' : [24.3466, 20.4208, 11.8708, 3.7149, 2.13553, 0.155525, 10.5782, 21.7029, 0.645089],               # noqa
    'Ge' : [16.0816, 6.3747, 3.7068, 3.683, 2.8509, 0.2516, 11.4468, 54.7625, 2.1313],                         # noqa
    'Ge4+' : [12.9172, 6.70003, 6.06791, 0.859041, 2.53718, 0.205855, 5.47913, 11.603, 1.45572],               # noqa
    'H' : [0.493002, 0.322912, 0.140191, 0.04081, 10.5109, 26.1257, 3.14236, 57.7997, 0.0030380001],           # noqa
    'H.' : [0.489918, 0.262003, 0.196767, 0.049879, 20.6593, 7.74039, 49.5519, 2.20159, 0.001305],             # noqa
    'H1-' : [0.897661, 0.565616, 0.415815, 0.116973, 53.1368, 15.187, 186.576, 3.56709, 0.002389],             # noqa
    'He' : [0.8734, 0.6309, 0.3112, 0.178, 9.1037, 3.3568, 22.9276, 0.9821, 0.0063999998],                     # noqa
    'Hf' : [29.144, 15.1726, 14.7586, 4.30013, 1.83262, 9.5999, 0.275116, 72.029, 8.58154],                    # noqa
    'Hf4+' : [28.8131, 18.4601, 12.7285, 5.59927, 1.59136, 0.128903, 6.76232, 14.0366, 2.39699],               # noqa
    'Hg' : [20.6809, 19.0417, 21.6575, 5.9676, 0.545, 8.4484, 1.5729, 38.3246, 12.6089],                       # noqa
    'Hg1+' : [25.0853, 18.4973, 16.8883, 6.48216, 1.39507, 7.65105, 0.443378, 28.2262, 12.0205],               # noqa
    'Hg2+' : [29.5641, 18.06, 12.8374, 6.89912, 1.21152, 7.05639, 0.284738, 20.7482, 10.6268],                 # noqa
    'Ho' : [26.9049, 17.294, 14.5583, 3.63837, 2.07051, 0.19794, 11.4407, 92.6566, 4.56796],                   # noqa
    'Ho3+' : [26.1296, 20.0994, 11.9788, 4.93676, 1.91072, 0.139358, 8.80018, 18.5908, 0.852795],              # noqa
    'I' : [20.1472, 18.9949, 7.5138, 2.2735, 4.347, 0.3814, 27.766, 66.8776, 4.0712],                          # noqa
    'I1-' : [20.2332, 18.997, 7.8069, 2.8868, 4.3579, 0.3815, 29.5259, 84.9304, 4.0714],                       # noqa
    'In' : [19.1624, 18.5596, 4.2948, 2.0396, 0.5476, 6.3776, 25.8499, 92.8029, 4.9391],                       # noqa
    'In3+' : [19.1045, 18.1108, 3.78897, 0.0, 0.551522, 6.3247, 17.3595, 0.0, 4.99635],                        # noqa
    'Ir' : [27.3049, 16.7296, 15.6115, 5.83377, 1.59279, 8.86553, 0.417916, 45.0011, 11.4722],                 # noqa
    'Ir3+' : [30.4156, 15.862, 13.6145, 5.82008, 1.34323, 7.10909, 0.204633, 20.3254, 8.27903],                # noqa
    'Ir4+' : [30.7058, 15.5512, 14.2326, 5.53672, 1.30923, 6.71983, 0.167252, 17.4911, 6.96824],               # noqa
    'K' : [8.2186, 7.4398, 1.0519, 0.8659, 12.7949, 0.7748, 213.187, 41.6841, 1.4228],                         # noqa
    'K1+' : [7.9578, 7.4917, 6.359, 1.1915, 12.6331, 0.7674, -0.0020000001, 31.9128, -4.9978],                 # noqa
    'Kr' : [17.3555, 6.7286, 5.5493, 3.5375, 1.9384, 16.5623, 0.2261, 39.3972, 2.825],                         # noqa
    'La' : [20.578, 19.599, 11.3727, 3.28719, 2.94817, 0.244475, 18.7726, 133.124, 2.14678],                   # noqa
    'La3+' : [20.2489, 19.3763, 11.6323, 0.336048, 2.9207, 0.250698, 17.8211, 54.9453, 2.4086],                # noqa
    'Li' : [1.1282, 0.7508, 0.6175, 0.4653, 3.9546, 1.0524, 85.3905, 168.261, 0.037700001],                    # noqa
    'Li1+' : [0.6968, 0.7888, 0.3414, 0.1563, 4.6237, 1.9557, 0.6316, 10.0953, 0.0167],                        # noqa
    'Lu' : [28.9476, 15.2208, 15.1, 3.71601, 1.90182, 9.98519, 0.261033, 84.3298, 7.97628],                    # noqa
    'Lu3+' : [28.4628, 18.121, 12.8429, 5.59415, 1.68216, 0.142292, 7.33727, 16.3535, 2.97573],                # noqa
    'Mg' : [5.4204, 2.1735, 1.2269, 2.3073, 2.8275, 79.2611, 0.3808, 7.1937, 0.8584],                          # noqa
    'Mg2+' : [3.4988, 3.8378, 1.3284, 0.8497, 2.1676, 4.7542, 0.185, 10.1411, 0.4853],                         # noqa
    'Mn' : [11.2819, 7.3573, 3.0193, 2.2441, 5.3409, 0.3432, 17.8674, 83.7543, 1.0896],                        # noqa
    'Mn2+' : [10.8061, 7.362, 3.5268, 0.2184, 5.2796, 0.3435, 14.343, 41.3235, 1.0874],                        # noqa
    'Mn3+' : [9.84521, 7.87194, 3.56531, 0.323613, 4.91797, 0.294393, 10.8171, 24.1281, 0.393974],             # noqa
    'Mn4+' : [9.96253, 7.97057, 2.76067, 0.054446999, 4.8485, 0.283303, 10.4852, 27.573, 0.251877],            # noqa
    'Mo' : [3.7025, 17.2356, 12.8876, 3.7429, 0.2772, 1.0958, 11.004, 61.6584, 4.3875],                        # noqa
    'Mo3+' : [21.1664, 18.2017, 11.7423, 2.30951, 0.014734, 1.03031, 9.53659, 26.6307, -14.421],               # noqa
    'Mo5+' : [21.0149, 18.0992, 11.4632, 0.740625, 0.014345, 1.02238, 8.78809, 23.3452, -14.316],              # noqa
    'Mo6+' : [17.8871, 11.175, 6.57891, 0.0, 1.03649, 8.48061, 0.058881, 0.0, 0.344941],                       # noqa
    'N' : [12.2126, 3.1322, 2.0125, 1.1663, 0.0057000001, 9.8933, 28.9975, 0.5826, -11.529],                   # noqa
    'Na' : [4.7626, 3.1736, 1.2674, 1.1128, 3.285, 8.8422, 0.3136, 129.424, 0.676],                            # noqa
    'Na1+' : [3.2565, 3.9362, 1.3998, 1.0032, 2.6671, 6.1153, 0.2001, 14.039, 0.404],                          # noqa
    'Nb' : [17.6142, 12.0144, 4.04183, 3.53346, 1.18865, 11.766, 0.204785, 69.7957, 3.75591],                  # noqa
    'Nb3+' : [19.8812, 18.0653, 11.0177, 1.94715, 0.019175, 1.13305, 10.1621, 28.3389, -12.912],               # noqa
    'Nb5+' : [17.9163, 13.3417, 10.799, 0.337905, 1.12446, 0.028781001, 9.28206, 25.7228, -6.3934],            # noqa
    'Nd' : [22.6845, 19.6847, 12.774, 2.85137, 2.66248, 0.210628, 15.885, 137.903, 1.98486],                   # noqa
    'Nd3+' : [21.961, 19.9339, 12.12, 1.51031, 2.52722, 0.199237, 14.1783, 30.8717, 1.47588],                  # noqa
    'Ne' : [3.9553, 3.1125, 1.4546, 1.1251, 8.4042, 3.4262, 0.2306, 21.7184, 0.3515],                          # noqa
    'Ni' : [12.8376, 7.292, 4.4438, 2.38, 3.8785, 0.2565, 12.1763, 66.3421, 1.0341],                           # noqa
    'Ni2+' : [11.4166, 7.4005, 5.3442, 0.9773, 3.6766, 0.2449, 8.873, 22.1626, 0.8614],                        # noqa
    'Ni3+' : [10.7806, 7.75868, 5.22746, 0.847114, 3.5477, 0.22314, 7.64468, 16.9673, 0.386044],               # noqa
    'Np' : [36.1874, 23.5964, 15.6402, 4.1855, 0.511929, 3.25396, 15.3622, 97.4908, 13.3573],                  # noqa
    'Np3+' : [35.0136, 22.7286, 14.3884, 1.75669, 0.48981, 2.81099, 12.33, 22.6581, 13.113],                   # noqa
    'Np4+' : [36.5254, 23.8083, 16.7707, 3.47947, 0.499384, 3.26371, 14.9455, 105.98, 13.3812],                # noqa
    'Np6+' : [35.7074, 22.613, 12.9898, 5.43227, 0.502322, 3.03807, 12.1449, 25.4928, 13.2544],                # noqa
    'O' : [3.0485, 2.2868, 1.5463, 0.867, 13.2771, 5.7011, 0.3239, 32.9089, 0.2508],                           # noqa
    'O1-' : [4.1916, 1.63969, 1.52673, -20.307, 12.8573, 4.17236, 47.0179, -0.01404, 21.9412],                 # noqa
    'O2-.' : [4.758, 3.637, 0.0, 0.0, 7.831, 30.05, 0.0, 0.0, 1.594],                                          # noqa
    'Os' : [28.1894, 16.155, 14.9305, 5.67589, 1.62903, 8.97948, 0.382661, 48.1647, 11.0005],                  # noqa
    'Os4+' : [30.419, 15.2637, 14.7458, 5.06795, 1.37113, 6.84706, 0.165191, 18.003, 6.49804],                 # noqa
    'P' : [6.4345, 4.1791, 1.78, 1.4908, 1.9067, 27.157, 0.526, 68.1645, 1.1149],                              # noqa
    'Pa' : [35.8847, 23.2948, 14.1891, 4.17287, 0.547751, 3.41519, 16.9235, 105.251, 13.4287],                 # noqa
    'Pb' : [31.0617, 13.0637, 18.442, 5.9696, 0.6902, 2.3576, 8.618, 47.2579, 13.4118],                        # noqa
    'Pb2+' : [21.7886, 19.5682, 19.1406, 7.01107, 1.3366, 0.488383, 6.7727, 23.8132, 12.4734],                 # noqa
    'Pb4+' : [32.1244, 18.8003, 12.0175, 6.96886, 1.00566, 6.10926, 0.147041, 14.714, 8.08428],                # noqa
    'Pd' : [19.3319, 15.5017, 5.29537, 0.605844, 0.698655, 7.98929, 25.2052, 76.8986, 5.26593],                # noqa
    'Pd2+' : [19.1701, 15.2096, 4.32234, 0.0, 0.696219, 7.55573, 22.5057, 0.0, 5.2916],                        # noqa
    'Pd4+' : [19.2493, 14.79, 2.89289, -7.9492, 0.683839, 7.14833, 17.9144, 0.0051270002, 13.0174],            # noqa
    'Pm' : [23.3405, 19.6095, 13.1235, 2.87516, 2.5627, 0.202088, 15.1009, 132.721, 2.02876],                  # noqa
    'Pm3+' : [22.5527, 20.1108, 12.0671, 2.07492, 2.4174, 0.185769, 13.1275, 27.4491, 1.19499],                # noqa
    'Po' : [34.6726, 15.4733, 13.1138, 7.02588, 0.700999, 3.55078, 9.55642, 47.0045, 13.677],                  # noqa
    'Pr' : [22.044, 19.6697, 12.3856, 2.82428, 2.77393, 0.222087, 16.7669, 143.644, 2.0583],                   # noqa
    'Pr3+' : [21.3727, 19.7491, 12.1329, 0.97518, 2.6452, 0.214299, 15.323, 36.4065, 1.77132],                 # noqa
    'Pr4+' : [20.9413, 20.0539, 12.4668, 0.296689, 2.54467, 0.202481, 14.8137, 45.4643, 1.24285],              # noqa
    'Pt' : [27.0059, 17.7639, 15.7131, 5.7837, 1.51293, 8.81174, 0.424593, 38.6103, 11.6883],                  # noqa
    'Pt2+' : [29.8429, 16.7224, 13.2153, 6.35234, 1.32927, 7.38979, 0.263297, 22.9426, 9.85329],               # noqa
    'Pt4+' : [30.9612, 15.9829, 13.7348, 5.92034, 1.24813, 6.60834, 0.16864, 16.9392, 7.39534],                # noqa
    'Pu' : [35.5103, 22.5787, 12.7766, 4.92159, 0.498626, 2.96627, 11.9484, 22.7502, 13.2116],                 # noqa
    'Pu3+' : [35.84, 22.7169, 13.5807, 5.66016, 0.484938, 2.96118, 11.5331, 24.3992, 13.1991],                 # noqa
    'Pu4+' : [35.6493, 22.646, 13.3595, 5.18831, 0.481422, 2.8902, 11.316, 21.8301, 13.1555],                  # noqa
    'Pu6+' : [35.1736, 22.7181, 14.7635, 2.28678, 0.473204, 2.73848, 11.553, 20.9303, 13.0582],                # noqa
    'Ra' : [35.763, 22.9064, 12.4739, 3.21097, 0.616341, 3.87135, 19.9887, 142.325, 13.6211],                  # noqa
    'Ra2+' : [35.215, 21.67, 7.91342, 7.65078, 0.604909, 3.5767, 12.601, 29.8436, 13.5431],                    # noqa
    'Rb' : [17.1784, 9.6435, 5.1399, 1.5292, 1.7888, 17.3151, 0.2748, 164.934, 3.4873],                        # noqa
    'Rb1+' : [17.5816, 7.6598, 5.8981, 2.7817, 1.7139, 14.7957, 0.1603, 31.2087, 2.0782],                      # noqa
    'Re' : [28.7621, 15.7189, 14.5564, 5.44174, 1.67191, 9.09227, 0.3505, 52.0861, 10.472],                    # noqa
    'Rh' : [19.2957, 14.3501, 4.73425, 1.28918, 0.751536, 8.21758, 25.8749, 98.6062, 5.328],                   # noqa
    'Rh3+' : [18.8785, 14.1259, 3.32515, -6.1989, 0.764252, 7.84438, 21.2487, -0.01036, 11.8678],              # noqa
    'Rh4+' : [18.8545, 13.9806, 2.53464, -5.6526, 0.760825, 7.62436, 19.3317, -0.0102, 11.2835],               # noqa
    'Rn' : [35.5631, 21.2816, 8.0037, 7.4433, 0.6631, 4.0691, 14.0422, 44.2473, 13.6905],                      # noqa
    'Ru' : [19.2674, 12.9182, 4.86337, 1.56756, 0.80852, 8.43467, 24.7997, 94.2928, 5.37874],                  # noqa
    'Ru+4' : [18.5003, 13.1787, 4.71304, 2.18535, 0.844582, 8.12534, 0.036495, 20.8504, 1.42357],              # noqa
    'Ru3+' : [18.5638, 13.2885, 9.32602, 3.00964, 0.847329, 8.37164, 0.017662, 22.887, -3.1892],               # noqa
    'S' : [6.9053, 5.2034, 1.4379, 1.5863, 1.4679, 22.2151, 0.2536, 56.172, 0.8669],                           # noqa
    'Sb' : [19.6418, 19.0455, 5.0371, 2.6827, 5.3034, 0.4607, 27.9074, 75.2825, 4.5909],                       # noqa
    'Sb3+' : [18.9755, 18.933, 5.10789, 0.288753, 0.467196, 5.22126, 19.5902, 55.5113, 4.69626],               # noqa
    'Sb5+' : [19.8685, 19.0302, 2.41253, 0.0, 5.44853, 0.467973, 14.1259, 0.0, 4.69263],                       # noqa
    'Sc' : [9.189, 7.3679, 1.6409, 1.468, 9.0213, 0.5729, 136.108, 51.3531, 1.3329],                           # noqa
    'Sc3+' : [13.4008, 8.0273, 1.65943, 1.57936, 0.29854, 7.9629, -0.28604, 16.0662, -6.6667],                 # noqa
    'Se' : [17.0006, 5.8196, 3.9731, 4.3543, 2.4098, 0.2726, 15.2372, 43.8163, 2.8409],                        # noqa
    'Si' : [6.2915, 3.0353, 1.9891, 1.541, 2.4386, 32.3337, 0.6785, 81.6937, 1.1407],                          # noqa
    'Si.' : [5.66269, 3.07164, 2.62446, 1.3932, 2.6652, 38.6634, 0.916946, 93.5458, 1.24707],                  # noqa
    'Si4+' : [4.43918, 3.20345, 1.19453, 0.41653, 1.64167, 3.43757, 0.2149, 6.65365, 0.746297],                # noqa
    'Sm' : [24.0042, 19.4258, 13.4396, 2.89604, 2.47274, 0.196451, 14.3996, 128.007, 2.20963],                 # noqa
    'Sm3+' : [23.1504, 20.2599, 11.9202, 2.71488, 2.31641, 0.174081, 12.1571, 24.8242, 0.954586],              # noqa
    'Sn' : [19.1889, 19.1005, 4.4585, 2.4663, 5.8303, 0.5031, 26.8909, 83.9571, 4.7821],                       # noqa
    'Sn2+' : [19.1094, 19.0548, 4.5648, 0.487, 0.5036, 5.8378, 23.3752, 62.2061, 4.7861],                      # noqa
    'Sn4+' : [18.9333, 19.7131, 3.4182, 0.019300001, 5.764, 0.4655, 14.0049, -0.7583, 3.9182],                 # noqa
    'Sr' : [17.5663, 9.8184, 5.422, 2.6694, 1.5564, 14.0988, 0.1664, 132.376, 2.5064],                         # noqa
    'Sr2+' : [18.0874, 8.1373, 2.5654, -34.193, 1.4907, 12.6963, 24.5651, -0.0138, 41.4025],                   # noqa
    'Ta' : [29.2024, 15.2293, 14.5135, 4.76492, 1.77333, 9.37046, 0.295977, 63.3644, 9.24354],                 # noqa
    'Ta5+' : [29.1587, 18.8407, 12.8268, 5.38695, 1.50711, 0.116741, 6.31524, 12.4244, 1.78555],               # noqa
    'Tb' : [25.8976, 18.2185, 14.3167, 2.95354, 2.24256, 0.196143, 12.6648, 115.362, 3.58324],                 # noqa
    'Tb3+' : [24.9559, 20.3271, 12.2471, 3.773, 2.05601, 0.149525, 10.0499, 21.2773, 0.691967],                # noqa
    'Tc' : [19.1301, 11.0948, 4.64901, 2.71263, 0.864132, 8.14487, 21.5707, 86.8472, 5.40428],                 # noqa
    'Te' : [19.9644, 19.0138, 6.14487, 2.5239, 4.81742, 0.420885, 28.5284, 70.8403, 4.352],                    # noqa
    'Th' : [35.5645, 23.4219, 12.7473, 4.80703, 0.563359, 3.46204, 17.8309, 99.1722, 13.4314],                 # noqa
    'Th4+' : [35.1007, 22.4418, 9.78554, 5.29444, 0.555054, 3.24498, 13.4661, 23.9533, 13.376],                # noqa
    'Ti' : [9.7595, 7.3558, 1.6991, 1.9021, 7.8508, 0.5, 35.6338, 116.105, 1.2807],                            # noqa
    'Ti2+' : [9.11423, 7.62174, 2.2793, 0.087898999, 7.5243, 0.457585, 19.5361, 61.6558, 0.897155],            # noqa
    'Ti3+' : [17.7344, 8.73816, 5.25691, 1.92134, 0.22061, 7.04716, -0.15762, 15.9768, -14.652],               # noqa
    'Ti4+' : [19.5114, 8.23473, 2.01341, 1.5208, 0.178847, 6.67018, -0.29263, 12.9464, -13.28],                # noqa
    'Tl' : [27.5446, 19.1584, 15.538, 5.52593, 0.65515, 8.70751, 1.96347, 45.8149, 13.1746],                   # noqa
    'Tl1+' : [21.3985, 20.4723, 18.7478, 6.82847, 1.4711, 0.517394, 7.43463, 28.8482, 12.5258],                # noqa
    'Tl3+' : [30.8695, 18.3841, 11.9328, 7.00574, 1.1008, 6.53852, 0.219074, 17.2114, 9.8027],                 # noqa
    'Tm' : [28.1819, 15.8851, 15.1542, 2.98706, 2.02859, 0.238849, 10.9975, 102.961, 6.75621],                 # noqa
    'Tm3+' : [27.3083, 19.332, 12.3339, 5.38348, 1.78711, 0.136974, 7.96778, 17.2922, 1.63929],                # noqa
    'U' : [36.0228, 23.4128, 14.9491, 4.188, 0.5293, 3.3253, 16.0927, 100.613, 13.3966],                       # noqa
    'U3+' : [35.5747, 22.5259, 12.2165, 5.37073, 0.52048, 3.12293, 12.7148, 26.3394, 13.3092],                 # noqa
    'U4+' : [35.3715, 22.5326, 12.0291, 4.7984, 0.516598, 3.05053, 12.5723, 23.4582, 13.2671],                 # noqa
    'U6+' : [34.8509, 22.7584, 14.0099, 1.21457, 0.507079, 2.8903, 13.1767, 25.2017, 13.1665],                 # noqa
    'V' : [10.2971, 7.3511, 2.0703, 2.0571, 6.8657, 0.4385, 26.8938, 102.478, 1.2199],                         # noqa
    'V2+' : [10.106, 7.3541, 2.2884, 0.022299999, 6.8818, 0.4409, 20.3004, 115.122, 1.2298],                   # noqa
    'V3+' : [9.43141, 7.7419, 2.15343, 0.016865, 6.39535, 0.383349, 15.1908, 63.969, 0.656565],                # noqa
    'V5+' : [15.6887, 8.14208, 2.03081, -9.576, 0.679003, 5.40135, 9.97278, 0.940464, 1.7143],                 # noqa
    'W' : [29.0818, 15.43, 14.4327, 5.11982, 1.72029, 9.2259, 0.321703, 57.056, 9.8875],                       # noqa
    'W6+' : [29.4936, 19.3763, 13.0544, 5.06412, 1.42755, 0.104621, 5.93667, 11.1972, 1.01074],                # noqa
    'Xe' : [20.2933, 19.0298, 8.9767, 1.99, 3.9282, 0.344, 26.4659, 64.2658, 3.7118],                          # noqa
    'Y' : [17.776, 10.2946, 5.72629, 3.26588, 1.4029, 12.8006, 0.125599, 104.354, 1.91213],                    # noqa
    'Y3+' : [17.9268, 9.1531, 1.76795, -33.108, 1.35417, 11.2145, 22.6599, -0.01319, 40.2602],                 # noqa
    'Yb' : [28.6641, 15.4345, 15.3087, 2.98963, 1.9889, 0.257119, 10.6647, 100.417, 7.56672],                  # noqa
    'Yb2+' : [28.1209, 17.6817, 13.3335, 5.14657, 1.78503, 0.15997, 8.18304, 20.39, 3.70983],                  # noqa
    'Yb3+' : [27.8917, 18.7614, 12.6072, 5.47647, 1.73272, 0.13879, 7.64412, 16.8153, 2.26001],                # noqa
    'Zn' : [14.0743, 7.0318, 5.1652, 2.41, 3.2655, 0.2333, 10.3163, 58.7097, 1.3041],                          # noqa
    'Zn2+' : [11.9719, 7.3862, 6.4668, 1.394, 2.9946, 0.2031, 7.0826, 18.0995, 0.7807],                        # noqa
    'Zr' : [17.8765, 10.948, 5.41732, 3.65721, 1.27618, 11.916, 0.117622, 87.6627, 2.06929],                   # noqa
    'Zr4+' : [18.1668, 10.0562, 1.01118, -2.6479, 1.2148, 10.1483, 21.6054, -0.10276, 9.41454],                # noqa
    (1, 0) : [0.493002, 0.322912, 0.140191, 0.04081, 10.5109, 26.1257, 3.14236, 57.7997, 0.0030380001],        # noqa
    (1, 1) : [0.897661, 0.565616, 0.415815, 0.116973, 53.1368, 15.187, 186.576, 3.56709, 0.002389],            # noqa
    (1, '.') : [0.489918, 0.262003, 0.196767, 0.049879, 20.6593, 7.74039, 49.5519, 2.20159, 0.001305],         # noqa
    (2, 0) : [0.8734, 0.6309, 0.3112, 0.178, 9.1037, 3.3568, 22.9276, 0.9821, 0.0063999998],                   # noqa
    (3, 0) : [1.1282, 0.7508, 0.6175, 0.4653, 3.9546, 1.0524, 85.3905, 168.261, 0.037700001],                  # noqa
    (3, 1) : [0.6968, 0.7888, 0.3414, 0.1563, 4.6237, 1.9557, 0.6316, 10.0953, 0.0167],                        # noqa
    (4, 0) : [1.5919, 1.1278, 0.5391, 0.7029, 43.6427, 1.8623, 103.483, 0.542, 0.0385],                        # noqa
    (4, 2) : [6.2603, 0.8849, 0.7993, 0.1647, 0.0027000001, 0.8313, 2.2758, 5.1146, -6.1092],                  # noqa
    (5, 0) : [2.0545, 1.3326, 1.0979, 0.7068, 23.2185, 1.021, 60.3498, 0.1403, -0.1932],                       # noqa
    (6, 0) : [2.31, 1.02, 1.5886, 0.865, 20.8439, 10.2075, 0.5687, 51.6512, 0.2156],                           # noqa
    (6, '.') : [2.26069, 1.56165, 1.05075, 0.839259, 22.6907, 0.656665, 9.75618, 55.5949, 0.286977],           # noqa
    (7, 0) : [12.2126, 3.1322, 2.0125, 1.1663, 0.0057000001, 9.8933, 28.9975, 0.5826, -11.529],                # noqa
    (8, 0) : [3.0485, 2.2868, 1.5463, 0.867, 13.2771, 5.7011, 0.3239, 32.9089, 0.2508],                        # noqa
    (8, 1) : [4.1916, 1.63969, 1.52673, -20.307, 12.8573, 4.17236, 47.0179, -0.01404, 21.9412],                # noqa
    (8, '.') : [4.758, 3.637, 0.0, 0.0, 7.831, 30.05, 0.0, 0.0, 1.594],                                        # noqa
    (9, 0) : [3.5392, 2.6412, 1.517, 1.0243, 10.2825, 4.2944, 0.2615, 26.1476, 0.2776],                        # noqa
    (9, 1) : [3.6322, 3.51057, 1.26064, 0.940706, 5.27756, 14.7353, 0.442258, 47.3437, 0.653396],              # noqa
    (10, 0) : [3.9553, 3.1125, 1.4546, 1.1251, 8.4042, 3.4262, 0.2306, 21.7184, 0.3515],                       # noqa
    (11, 0) : [4.7626, 3.1736, 1.2674, 1.1128, 3.285, 8.8422, 0.3136, 129.424, 0.676],                         # noqa
    (11, 1) : [3.2565, 3.9362, 1.3998, 1.0032, 2.6671, 6.1153, 0.2001, 14.039, 0.404],                         # noqa
    (12, 0) : [5.4204, 2.1735, 1.2269, 2.3073, 2.8275, 79.2611, 0.3808, 7.1937, 0.8584],                       # noqa
    (12, 2) : [3.4988, 3.8378, 1.3284, 0.8497, 2.1676, 4.7542, 0.185, 10.1411, 0.4853],                        # noqa
    (13, 0) : [6.4202, 1.9002, 1.5936, 1.9646, 3.0387, 0.7426, 31.5472, 85.0886, 1.1151],                      # noqa
    (13, 3) : [4.17448, 3.3876, 1.20296, 0.528137, 1.93816, 4.14553, 0.228753, 8.28524, 0.706786],             # noqa
    (14, 0) : [6.2915, 3.0353, 1.9891, 1.541, 2.4386, 32.3337, 0.6785, 81.6937, 1.1407],                       # noqa
    (14, 4) : [4.43918, 3.20345, 1.19453, 0.41653, 1.64167, 3.43757, 0.2149, 6.65365, 0.746297],               # noqa
    (14, '.') : [5.66269, 3.07164, 2.62446, 1.3932, 2.6652, 38.6634, 0.916946, 93.5458, 1.24707],              # noqa
    (15, 0) : [6.4345, 4.1791, 1.78, 1.4908, 1.9067, 27.157, 0.526, 68.1645, 1.1149],                          # noqa
    (16, 0) : [6.9053, 5.2034, 1.4379, 1.5863, 1.4679, 22.2151, 0.2536, 56.172, 0.8669],                       # noqa
    (17, 0) : [11.4604, 7.1964, 6.2556, 1.6455, 0.0104, 1.1662, 18.5194, 47.7784, -9.5574],                    # noqa
    (17, 1) : [18.2915, 7.2084, 6.5337, 2.3386, 0.0066, 1.1717, 19.5424, 60.4486, -16.378],                    # noqa
    (18, 0) : [7.4845, 6.7723, 0.6539, 1.6442, 0.9072, 14.8407, 43.8983, 33.3929, 1.4445],                     # noqa
    (19, 0) : [8.2186, 7.4398, 1.0519, 0.8659, 12.7949, 0.7748, 213.187, 41.6841, 1.4228],                     # noqa
    (19, 1) : [7.9578, 7.4917, 6.359, 1.1915, 12.6331, 0.7674, -0.0020000001, 31.9128, -4.9978],               # noqa
    (20, 0) : [8.6266, 7.3873, 1.5899, 1.0211, 10.4421, 0.6599, 85.7484, 178.437, 1.3751],                     # noqa
    (20, 2) : [15.6348, 7.9518, 8.4372, 0.8537, -0.0074, 0.6089, 10.3116, 25.9905, -14.875],                   # noqa
    (21, 0) : [9.189, 7.3679, 1.6409, 1.468, 9.0213, 0.5729, 136.108, 51.3531, 1.3329],                        # noqa
    (21, 3) : [13.4008, 8.0273, 1.65943, 1.57936, 0.29854, 7.9629, -0.28604, 16.0662, -6.6667],                # noqa
    (22, 0) : [9.7595, 7.3558, 1.6991, 1.9021, 7.8508, 0.5, 35.6338, 116.105, 1.2807],                         # noqa
    (22, 2) : [9.11423, 7.62174, 2.2793, 0.087898999, 7.5243, 0.457585, 19.5361, 61.6558, 0.897155],           # noqa
    (22, 3) : [17.7344, 8.73816, 5.25691, 1.92134, 0.22061, 7.04716, -0.15762, 15.9768, -14.652],              # noqa
    (22, 4) : [19.5114, 8.23473, 2.01341, 1.5208, 0.178847, 6.67018, -0.29263, 12.9464, -13.28],               # noqa
    (23, 0) : [10.2971, 7.3511, 2.0703, 2.0571, 6.8657, 0.4385, 26.8938, 102.478, 1.2199],                     # noqa
    (23, 2) : [10.106, 7.3541, 2.2884, 0.022299999, 6.8818, 0.4409, 20.3004, 115.122, 1.2298],                 # noqa
    (23, 3) : [9.43141, 7.7419, 2.15343, 0.016865, 6.39535, 0.383349, 15.1908, 63.969, 0.656565],              # noqa
    (23, 5) : [15.6887, 8.14208, 2.03081, -9.576, 0.679003, 5.40135, 9.97278, 0.940464, 1.7143],               # noqa
    (24, 0) : [10.6406, 7.3537, 3.324, 1.4922, 6.1038, 0.392, 20.2626, 98.7399, 1.1832],                       # noqa
    (24, 2) : [9.54034, 7.7509, 3.58274, 0.509107, 5.66078, 0.344261, 13.3075, 32.4224, 0.616898],             # noqa
    (24, 3) : [9.6809, 7.81136, 2.87603, 0.113575, 5.59463, 0.334393, 12.8288, 32.8761, 0.518275],             # noqa
    (25, 0) : [11.2819, 7.3573, 3.0193, 2.2441, 5.3409, 0.3432, 17.8674, 83.7543, 1.0896],                     # noqa
    (25, 2) : [10.8061, 7.362, 3.5268, 0.2184, 5.2796, 0.3435, 14.343, 41.3235, 1.0874],                       # noqa
    (25, 3) : [9.84521, 7.87194, 3.56531, 0.323613, 4.91797, 0.294393, 10.8171, 24.1281, 0.393974],            # noqa
    (25, 4) : [9.96253, 7.97057, 2.76067, 0.054446999, 4.8485, 0.283303, 10.4852, 27.573, 0.251877],           # noqa
    (26, 0) : [11.0424, 7.374, 4.1346, 0.4399, 4.6538, 0.3053, 12.0546, 31.2809, 1.0097],                      # noqa
    (26, 3) : [11.1764, 7.3863, 3.3948, 0.072400004, 4.6147, 0.3005, 11.6729, 38.5566, 0.9707],                # noqa
    (27, 0) : [12.2841, 7.3409, 4.0034, 2.3488, 4.2791, 0.2784, 13.5359, 71.1692, 1.0118],                     # noqa
    (27, 2) : [11.2296, 7.3883, 4.7393, 0.7108, 4.1231, 0.2726, 10.2443, 25.6466, 0.9324],                     # noqa
    (27, 3) : [10.338, 7.88173, 4.76795, 0.725591, 3.90969, 0.238668, 8.35583, 18.3491, 0.286667],             # noqa
    (28, 0) : [12.8376, 7.292, 4.4438, 2.38, 3.8785, 0.2565, 12.1763, 66.3421, 1.0341],                        # noqa
    (28, 2) : [11.4166, 7.4005, 5.3442, 0.9773, 3.6766, 0.2449, 8.873, 22.1626, 0.8614],                       # noqa
    (28, 3) : [10.7806, 7.75868, 5.22746, 0.847114, 3.5477, 0.22314, 7.64468, 16.9673, 0.386044],              # noqa
    (29, 0) : [13.338, 7.1676, 5.6158, 1.6735, 3.5828, 0.247, 11.3966, 64.8126, 1.191],                        # noqa
    (29, 1) : [11.9475, 7.3573, 6.2455, 1.5578, 3.3669, 0.2274, 8.6625, 25.8487, 0.89],                        # noqa
    (29, 2) : [11.8168, 7.11181, 5.78135, 1.14523, 3.37484, 0.244078, 7.9876, 19.897, 1.14431],                # noqa
    (30, 0) : [14.0743, 7.0318, 5.1652, 2.41, 3.2655, 0.2333, 10.3163, 58.7097, 1.3041],                       # noqa
    (30, 2) : [11.9719, 7.3862, 6.4668, 1.394, 2.9946, 0.2031, 7.0826, 18.0995, 0.7807],                       # noqa
    (31, 0) : [15.2354, 6.7006, 4.3591, 2.9623, 3.0669, 0.2412, 10.7805, 61.4135, 1.7189],                     # noqa
    (31, 3) : [12.692, 6.69883, 6.06692, 1.0066, 2.81262, 0.22789, 6.36441, 14.4122, 1.53545],                 # noqa
    (32, 0) : [16.0816, 6.3747, 3.7068, 3.683, 2.8509, 0.2516, 11.4468, 54.7625, 2.1313],                      # noqa
    (32, 4) : [12.9172, 6.70003, 6.06791, 0.859041, 2.53718, 0.205855, 5.47913, 11.603, 1.45572],              # noqa
    (33, 0) : [16.6723, 6.0701, 3.4313, 4.2779, 2.6345, 0.2647, 12.9479, 47.7972, 2.531],                      # noqa
    (34, 0) : [17.0006, 5.8196, 3.9731, 4.3543, 2.4098, 0.2726, 15.2372, 43.8163, 2.8409],                     # noqa
    (35, 0) : [17.1789, 5.2358, 5.6377, 3.9851, 2.1723, 16.5796, 0.2609, 41.4328, 2.9557],                     # noqa
    (35, 1) : [17.1718, 6.3338, 5.5754, 3.7272, 2.2059, 19.3345, 0.2871, 58.1535, 3.1776],                     # noqa
    (36, 0) : [17.3555, 6.7286, 5.5493, 3.5375, 1.9384, 16.5623, 0.2261, 39.3972, 2.825],                      # noqa
    (37, 0) : [17.1784, 9.6435, 5.1399, 1.5292, 1.7888, 17.3151, 0.2748, 164.934, 3.4873],                     # noqa
    (37, 1) : [17.5816, 7.6598, 5.8981, 2.7817, 1.7139, 14.7957, 0.1603, 31.2087, 2.0782],                     # noqa
    (38, 0) : [17.5663, 9.8184, 5.422, 2.6694, 1.5564, 14.0988, 0.1664, 132.376, 2.5064],                      # noqa
    (38, 2) : [18.0874, 8.1373, 2.5654, -34.193, 1.4907, 12.6963, 24.5651, -0.0138, 41.4025],                  # noqa
    (39, 0) : [17.776, 10.2946, 5.72629, 3.26588, 1.4029, 12.8006, 0.125599, 104.354, 1.91213],                # noqa
    (39, 3) : [17.9268, 9.1531, 1.76795, -33.108, 1.35417, 11.2145, 22.6599, -0.01319, 40.2602],               # noqa
    (40, 0) : [17.8765, 10.948, 5.41732, 3.65721, 1.27618, 11.916, 0.117622, 87.6627, 2.06929],                # noqa
    (40, 4) : [18.1668, 10.0562, 1.01118, -2.6479, 1.2148, 10.1483, 21.6054, -0.10276, 9.41454],               # noqa
    (41, 0) : [17.6142, 12.0144, 4.04183, 3.53346, 1.18865, 11.766, 0.204785, 69.7957, 3.75591],               # noqa
    (41, 3) : [19.8812, 18.0653, 11.0177, 1.94715, 0.019175, 1.13305, 10.1621, 28.3389, -12.912],              # noqa
    (41, 5) : [17.9163, 13.3417, 10.799, 0.337905, 1.12446, 0.028781001, 9.28206, 25.7228, -6.3934],           # noqa
    (42, 0) : [3.7025, 17.2356, 12.8876, 3.7429, 0.2772, 1.0958, 11.004, 61.6584, 4.3875],                     # noqa
    (42, 3) : [21.1664, 18.2017, 11.7423, 2.30951, 0.014734, 1.03031, 9.53659, 26.6307, -14.421],              # noqa
    (42, 5) : [21.0149, 18.0992, 11.4632, 0.740625, 0.014345, 1.02238, 8.78809, 23.3452, -14.316],             # noqa
    (42, 6) : [17.8871, 11.175, 6.57891, 0.0, 1.03649, 8.48061, 0.058881, 0.0, 0.344941],                      # noqa
    (43, 0) : [19.1301, 11.0948, 4.64901, 2.71263, 0.864132, 8.14487, 21.5707, 86.8472, 5.40428],              # noqa
    (44, 0) : [18.5003, 13.1787, 4.71304, 2.18535, 0.844582, 8.12534, 0.036495, 20.8504, 1.42357],             # noqa
    (44, 3) : [18.5638, 13.2885, 9.32602, 3.00964, 0.847329, 8.37164, 0.017662, 22.887, -3.1892],              # noqa
    (45, 0) : [19.2957, 14.3501, 4.73425, 1.28918, 0.751536, 8.21758, 25.8749, 98.6062, 5.328],                # noqa
    (45, 3) : [18.8785, 14.1259, 3.32515, -6.1989, 0.764252, 7.84438, 21.2487, -0.01036, 11.8678],             # noqa
    (45, 4) : [18.8545, 13.9806, 2.53464, -5.6526, 0.760825, 7.62436, 19.3317, -0.0102, 11.2835],              # noqa
    (46, 0) : [19.3319, 15.5017, 5.29537, 0.605844, 0.698655, 7.98929, 25.2052, 76.8986, 5.26593],             # noqa
    (46, 2) : [19.1701, 15.2096, 4.32234, 0.0, 0.696219, 7.55573, 22.5057, 0.0, 5.2916],                       # noqa
    (46, 4) : [19.2493, 14.79, 2.89289, -7.9492, 0.683839, 7.14833, 17.9144, 0.0051270002, 13.0174],           # noqa
    (47, 0) : [19.2808, 16.6885, 4.8045, 1.0463, 0.6446, 7.4726, 24.6605, 99.8156, 5.179],                     # noqa
    (47, 1) : [19.1812, 15.9719, 5.27475, 0.357534, 0.646179, 7.19123, 21.7326, 66.1147, 5.21572],             # noqa
    (47, 2) : [19.1643, 16.2456, 4.3709, 0.0, 0.645643, 7.18544, 21.4072, 0.0, 5.21404],                       # noqa
    (48, 0) : [19.2214, 17.6444, 4.461, 1.6029, 0.5946, 6.9089, 24.7008, 87.4825, 5.0694],                     # noqa
    (48, 2) : [19.1514, 17.2535, 4.47128, 0.0, 0.597922, 6.80639, 20.2521, 0.0, 5.11937],                      # noqa
    (49, 0) : [19.1624, 18.5596, 4.2948, 2.0396, 0.5476, 6.3776, 25.8499, 92.8029, 4.9391],                    # noqa
    (49, 3) : [19.1045, 18.1108, 3.78897, 0.0, 0.551522, 6.3247, 17.3595, 0.0, 4.99635],                       # noqa
    (50, 0) : [19.1889, 19.1005, 4.4585, 2.4663, 5.8303, 0.5031, 26.8909, 83.9571, 4.7821],                    # noqa
    (50, 2) : [19.1094, 19.0548, 4.5648, 0.487, 0.5036, 5.8378, 23.3752, 62.2061, 4.7861],                     # noqa
    (50, 4) : [18.9333, 19.7131, 3.4182, 0.019300001, 5.764, 0.4655, 14.0049, -0.7583, 3.9182],                # noqa
    (51, 0) : [19.6418, 19.0455, 5.0371, 2.6827, 5.3034, 0.4607, 27.9074, 75.2825, 4.5909],                    # noqa
    (51, 3) : [18.9755, 18.933, 5.10789, 0.288753, 0.467196, 5.22126, 19.5902, 55.5113, 4.69626],              # noqa
    (51, 5) : [19.8685, 19.0302, 2.41253, 0.0, 5.44853, 0.467973, 14.1259, 0.0, 4.69263],                      # noqa
    (52, 0) : [19.9644, 19.0138, 6.14487, 2.5239, 4.81742, 0.420885, 28.5284, 70.8403, 4.352],                 # noqa
    (53, 0) : [20.1472, 18.9949, 7.5138, 2.2735, 4.347, 0.3814, 27.766, 66.8776, 4.0712],                      # noqa
    (53, 1) : [20.2332, 18.997, 7.8069, 2.8868, 4.3579, 0.3815, 29.5259, 84.9304, 4.0714],                     # noqa
    (54, 0) : [20.2933, 19.0298, 8.9767, 1.99, 3.9282, 0.344, 26.4659, 64.2658, 3.7118],                       # noqa
    (55, 0) : [20.3892, 19.1062, 10.662, 1.4953, 3.569, 0.3107, 24.3879, 213.904, 3.3352],                     # noqa
    (55, 1) : [20.3524, 19.1278, 10.2821, 0.9615, 3.552, 0.3086, 23.7128, 59.4565, 3.2791],                    # noqa
    (56, 0) : [20.3361, 19.297, 10.888, 2.6959, 3.216, 0.2756, 20.2073, 167.202, 2.7731],                      # noqa
    (56, 2) : [20.1807, 19.1136, 10.9054, 0.773634, 3.21367, 0.28331, 20.0558, 51.746, 3.02902],               # noqa
    (57, 0) : [20.578, 19.599, 11.3727, 3.28719, 2.94817, 0.244475, 18.7726, 133.124, 2.14678],                # noqa
    (57, 3) : [20.2489, 19.3763, 11.6323, 0.336048, 2.9207, 0.250698, 17.8211, 54.9453, 2.4086],               # noqa
    (58, 0) : [21.1671, 19.7695, 11.8513, 3.33049, 2.81219, 0.226836, 17.6083, 127.113, 1.86264],              # noqa
    (58, 3) : [20.8036, 19.559, 11.9369, 0.612376, 2.77691, 0.23154, 16.5408, 43.1692, 2.09013],               # noqa
    (58, 4) : [20.3235, 19.8186, 12.1233, 0.144583, 2.65941, 0.21885, 15.7992, 62.2355, 1.5918],               # noqa
    (59, 0) : [22.044, 19.6697, 12.3856, 2.82428, 2.77393, 0.222087, 16.7669, 143.644, 2.0583],                # noqa
    (59, 3) : [21.3727, 19.7491, 12.1329, 0.97518, 2.6452, 0.214299, 15.323, 36.4065, 1.77132],                # noqa
    (59, 4) : [20.9413, 20.0539, 12.4668, 0.296689, 2.54467, 0.202481, 14.8137, 45.4643, 1.24285],             # noqa
    (60, 0) : [22.6845, 19.6847, 12.774, 2.85137, 2.66248, 0.210628, 15.885, 137.903, 1.98486],                # noqa
    (60, 3) : [21.961, 19.9339, 12.12, 1.51031, 2.52722, 0.199237, 14.1783, 30.8717, 1.47588],                 # noqa
    (61, 0) : [23.3405, 19.6095, 13.1235, 2.87516, 2.5627, 0.202088, 15.1009, 132.721, 2.02876],               # noqa
    (61, 3) : [22.5527, 20.1108, 12.0671, 2.07492, 2.4174, 0.185769, 13.1275, 27.4491, 1.19499],               # noqa
    (62, 0) : [24.0042, 19.4258, 13.4396, 2.89604, 2.47274, 0.196451, 14.3996, 128.007, 2.20963],              # noqa
    (62, 3) : [23.1504, 20.2599, 11.9202, 2.71488, 2.31641, 0.174081, 12.1571, 24.8242, 0.954586],             # noqa
    (63, 0) : [24.6274, 19.0886, 13.7603, 2.9227, 2.3879, 0.1942, 13.7546, 123.174, 2.5745],                   # noqa
    (63, 2) : [24.0063, 19.9504, 11.8034, 3.87243, 2.27783, 0.17353, 11.6096, 26.5156, 1.36389],               # noqa
    (63, 3) : [23.7497, 20.3745, 11.8509, 3.26503, 2.22258, 0.16394, 11.311, 22.9966, 0.759344],               # noqa
    (64, 0) : [25.0709, 19.0798, 13.8518, 3.54545, 2.25341, 0.181951, 12.9331, 101.398, 2.4196],               # noqa
    (64, 3) : [24.3466, 20.4208, 11.8708, 3.7149, 2.13553, 0.155525, 10.5782, 21.7029, 0.645089],              # noqa
    (65, 0) : [25.8976, 18.2185, 14.3167, 2.95354, 2.24256, 0.196143, 12.6648, 115.362, 3.58324],              # noqa
    (65, 3) : [24.9559, 20.3271, 12.2471, 3.773, 2.05601, 0.149525, 10.0499, 21.2773, 0.691967],               # noqa
    (66, 0) : [26.507, 17.6383, 14.5596, 2.96577, 2.1802, 0.202172, 12.1899, 111.874, 4.29728],                # noqa
    (66, 3) : [25.5395, 20.2861, 11.9812, 4.50073, 1.9804, 0.143384, 9.34972, 19.581, 0.68969],                # noqa
    (67, 0) : [26.9049, 17.294, 14.5583, 3.63837, 2.07051, 0.19794, 11.4407, 92.6566, 4.56796],                # noqa
    (67, 3) : [26.1296, 20.0994, 11.9788, 4.93676, 1.91072, 0.139358, 8.80018, 18.5908, 0.852795],             # noqa
    (68, 0) : [27.6563, 16.4285, 14.9779, 2.98233, 2.07356, 0.223545, 11.3604, 105.703, 5.92046],              # noqa
    (68, 3) : [26.722, 19.7748, 12.1506, 5.17379, 1.84659, 0.13729, 8.36225, 17.8974, 1.17613],                # noqa
    (69, 0) : [28.1819, 15.8851, 15.1542, 2.98706, 2.02859, 0.238849, 10.9975, 102.961, 6.75621],              # noqa
    (69, 3) : [27.3083, 19.332, 12.3339, 5.38348, 1.78711, 0.136974, 7.96778, 17.2922, 1.63929],               # noqa
    (70, 0) : [28.6641, 15.4345, 15.3087, 2.98963, 1.9889, 0.257119, 10.6647, 100.417, 7.56672],               # noqa
    (70, 2) : [28.1209, 17.6817, 13.3335, 5.14657, 1.78503, 0.15997, 8.18304, 20.39, 3.70983],                 # noqa
    (70, 3) : [27.8917, 18.7614, 12.6072, 5.47647, 1.73272, 0.13879, 7.64412, 16.8153, 2.26001],               # noqa
    (71, 0) : [28.9476, 15.2208, 15.1, 3.71601, 1.90182, 9.98519, 0.261033, 84.3298, 7.97628],                 # noqa
    (71, 3) : [28.4628, 18.121, 12.8429, 5.59415, 1.68216, 0.142292, 7.33727, 16.3535, 2.97573],               # noqa
    (72, 0) : [29.144, 15.1726, 14.7586, 4.30013, 1.83262, 9.5999, 0.275116, 72.029, 8.58154],                 # noqa
    (72, 4) : [28.8131, 18.4601, 12.7285, 5.59927, 1.59136, 0.128903, 6.76232, 14.0366, 2.39699],              # noqa
    (73, 0) : [29.2024, 15.2293, 14.5135, 4.76492, 1.77333, 9.37046, 0.295977, 63.3644, 9.24354],              # noqa
    (73, 5) : [29.1587, 18.8407, 12.8268, 5.38695, 1.50711, 0.116741, 6.31524, 12.4244, 1.78555],              # noqa
    (74, 0) : [29.0818, 15.43, 14.4327, 5.11982, 1.72029, 9.2259, 0.321703, 57.056, 9.8875],                   # noqa
    (74, 6) : [29.4936, 19.3763, 13.0544, 5.06412, 1.42755, 0.104621, 5.93667, 11.1972, 1.01074],              # noqa
    (75, 0) : [28.7621, 15.7189, 14.5564, 5.44174, 1.67191, 9.09227, 0.3505, 52.0861, 10.472],                 # noqa
    (76, 0) : [28.1894, 16.155, 14.9305, 5.67589, 1.62903, 8.97948, 0.382661, 48.1647, 11.0005],               # noqa
    (76, 4) : [30.419, 15.2637, 14.7458, 5.06795, 1.37113, 6.84706, 0.165191, 18.003, 6.49804],                # noqa
    (77, 0) : [27.3049, 16.7296, 15.6115, 5.83377, 1.59279, 8.86553, 0.417916, 45.0011, 11.4722],              # noqa
    (77, 3) : [30.4156, 15.862, 13.6145, 5.82008, 1.34323, 7.10909, 0.204633, 20.3254, 8.27903],               # noqa
    (77, 4) : [30.7058, 15.5512, 14.2326, 5.53672, 1.30923, 6.71983, 0.167252, 17.4911, 6.96824],              # noqa
    (78, 0) : [27.0059, 17.7639, 15.7131, 5.7837, 1.51293, 8.81174, 0.424593, 38.6103, 11.6883],               # noqa
    (78, 2) : [29.8429, 16.7224, 13.2153, 6.35234, 1.32927, 7.38979, 0.263297, 22.9426, 9.85329],              # noqa
    (78, 4) : [30.9612, 15.9829, 13.7348, 5.92034, 1.24813, 6.60834, 0.16864, 16.9392, 7.39534],               # noqa
    (79, 0) : [16.8819, 18.5913, 25.5582, 5.86, 0.4611, 8.6216, 1.4826, 36.3956, 12.0658],                     # noqa
    (79, 1) : [28.0109, 17.8204, 14.3359, 6.58077, 1.35321, 7.7395, 0.356752, 26.4043, 11.2299],               # noqa
    (79, 3) : [30.6886, 16.9029, 12.7801, 6.52354, 1.2199, 6.82872, 0.212867, 18.659, 9.0968],                 # noqa
    (80, 0) : [20.6809, 19.0417, 21.6575, 5.9676, 0.545, 8.4484, 1.5729, 38.3246, 12.6089],                    # noqa
    (80, 1) : [25.0853, 18.4973, 16.8883, 6.48216, 1.39507, 7.65105, 0.443378, 28.2262, 12.0205],              # noqa
    (80, 2) : [29.5641, 18.06, 12.8374, 6.89912, 1.21152, 7.05639, 0.284738, 20.7482, 10.6268],                # noqa
    (81, 0) : [27.5446, 19.1584, 15.538, 5.52593, 0.65515, 8.70751, 1.96347, 45.8149, 13.1746],                # noqa
    (81, 1) : [21.3985, 20.4723, 18.7478, 6.82847, 1.4711, 0.517394, 7.43463, 28.8482, 12.5258],               # noqa
    (81, 3) : [30.8695, 18.3841, 11.9328, 7.00574, 1.1008, 6.53852, 0.219074, 17.2114, 9.8027],                # noqa
    (82, 0) : [31.0617, 13.0637, 18.442, 5.9696, 0.6902, 2.3576, 8.618, 47.2579, 13.4118],                     # noqa
    (82, 2) : [21.7886, 19.5682, 19.1406, 7.01107, 1.3366, 0.488383, 6.7727, 23.8132, 12.4734],                # noqa
    (82, 4) : [32.1244, 18.8003, 12.0175, 6.96886, 1.00566, 6.10926, 0.147041, 14.714, 8.08428],               # noqa
    (83, 0) : [33.3689, 12.951, 16.5877, 6.4692, 0.704, 2.9238, 8.7937, 48.0093, 13.5782],                     # noqa
    (83, 3) : [21.8053, 19.5026, 19.1053, 7.10295, 1.2356, 6.24149, 0.469999, 20.3185, 12.4711],               # noqa
    (83, 5) : [33.5364, 25.0946, 19.2497, 6.91555, 0.91654, 0.039042, 5.71414, 12.8285, -6.7994],              # noqa
    (84, 0) : [34.6726, 15.4733, 13.1138, 7.02588, 0.700999, 3.55078, 9.55642, 47.0045, 13.677],               # noqa
    (85, 0) : [35.3163, 19.0211, 9.49887, 7.42518, 0.68587, 3.97458, 11.3824, 45.4715, 13.7108],               # noqa
    (86, 0) : [35.5631, 21.2816, 8.0037, 7.4433, 0.6631, 4.0691, 14.0422, 44.2473, 13.6905],                   # noqa
    (87, 0) : [35.9299, 23.0547, 12.1439, 2.11253, 0.646453, 4.17619, 23.1052, 150.645, 13.7247],              # noqa
    (88, 0) : [35.763, 22.9064, 12.4739, 3.21097, 0.616341, 3.87135, 19.9887, 142.325, 13.6211],               # noqa
    (88, 2) : [35.215, 21.67, 7.91342, 7.65078, 0.604909, 3.5767, 12.601, 29.8436, 13.5431],                   # noqa
    (89, 0) : [35.6597, 23.1032, 12.5977, 4.08655, 0.589092, 3.65155, 18.599, 117.02, 13.5266],                # noqa
    (89, 3) : [35.1736, 22.1112, 8.19216, 7.05545, 0.579689, 3.41437, 12.9187, 25.9443, 13.4637],              # noqa
    (90, 0) : [35.5645, 23.4219, 12.7473, 4.80703, 0.563359, 3.46204, 17.8309, 99.1722, 13.4314],              # noqa
    (90, 4) : [35.1007, 22.4418, 9.78554, 5.29444, 0.555054, 3.24498, 13.4661, 23.9533, 13.376],               # noqa
    (91, 0) : [35.8847, 23.2948, 14.1891, 4.17287, 0.547751, 3.41519, 16.9235, 105.251, 13.4287],              # noqa
    (92, 0) : [36.0228, 23.4128, 14.9491, 4.188, 0.5293, 3.3253, 16.0927, 100.613, 13.3966],                   # noqa
    (92, 3) : [35.5747, 22.5259, 12.2165, 5.37073, 0.52048, 3.12293, 12.7148, 26.3394, 13.3092],               # noqa
    (92, 4) : [35.3715, 22.5326, 12.0291, 4.7984, 0.516598, 3.05053, 12.5723, 23.4582, 13.2671],               # noqa
    (92, 6) : [34.8509, 22.7584, 14.0099, 1.21457, 0.507079, 2.8903, 13.1767, 25.2017, 13.1665],               # noqa
    (93, 0) : [36.1874, 23.5964, 15.6402, 4.1855, 0.511929, 3.25396, 15.3622, 97.4908, 13.3573],               # noqa
    (93, 3) : [35.0136, 22.7286, 14.3884, 1.75669, 0.48981, 2.81099, 12.33, 22.6581, 13.113],                  # noqa
    (93, 4) : [36.5254, 23.8083, 16.7707, 3.47947, 0.499384, 3.26371, 14.9455, 105.98, 13.3812],               # noqa
    (93, 6) : [35.7074, 22.613, 12.9898, 5.43227, 0.502322, 3.03807, 12.1449, 25.4928, 13.2544],               # noqa
    (94, 0) : [35.5103, 22.5787, 12.7766, 4.92159, 0.498626, 2.96627, 11.9484, 22.7502, 13.2116],              # noqa
    (94, 3) : [35.84, 22.7169, 13.5807, 5.66016, 0.484938, 2.96118, 11.5331, 24.3992, 13.1991],                # noqa
    (94, 4) : [35.6493, 22.646, 13.3595, 5.18831, 0.481422, 2.8902, 11.316, 21.8301, 13.1555],                 # noqa
    (94, 6) : [35.1736, 22.7181, 14.7635, 2.28678, 0.473204, 2.73848, 11.553, 20.9303, 13.0582],               # noqa
    (95, 0) : [36.6706, 24.0992, 17.3415, 3.49331, 0.483629, 3.20647, 14.3136, 102.273, 13.3592],              # noqa
    (96, 0) : [36.6488, 24.4096, 17.399, 4.21665, 0.465154, 3.08997, 13.4346, 88.4834, 13.2887],               # noqa
    (97, 0) : [36.7881, 24.7736, 17.8919, 4.23284, 0.451018, 3.04619, 12.8946, 86.003, 13.2754],               # noqa
    (98, 0) : [36.9185, 25.1995, 18.3317, 4.24391, 0.437533, 3.00775, 12.4044, 83.7881, 13.2674],              # noqa
    }