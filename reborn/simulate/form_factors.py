
import numpy as np
from xraylib import FF_Rayl, Fi, Fii

from scipy import constants as const

eV = const.value('electron volt')


# def atomic_form_factor(qmags, atomic_number, photon_energy, dispersion_correction=True):
#     r"""
#     Get the q-dependent atomic scattering factors from the xraylib package.
#
#     Args:
#         qmags (numpy array):
#         atomic_number:
#         photon_energy:
#
#     Returns:
#
#     """
#
#     Z = atomic_number
#     E = photon_energy/eV/1000
#     qq = qmags/4.0/np.pi/1e10
#     f = np.array([FF_Rayl(Z, q) for q in qq])
#     if dispersion_correction:
#         f += Fi(Z, E) - 1j*Fii(Z, E)
#     return f


def sphere_form_factor(radius, q_mags, check_divide_by_zero=True):

    r"""
    Form factor :math:`f(q)` for a sphere of radius :math:`r`, at given :math:`q` magnitudes.  The formula is

    .. math::

        f(q) = 4 \pi \frac{\sin(qr) - qr \cos(qr)}{q^3}

    When :math:`q = 0`, the following limit is used:

    .. math::

        f(0) = \frac{4}{3} \pi r^3

    Formula can be cound, for example, in Table A.1 in Guinier's "X-ray diffraction in crystals, imperfect crystals, and
    amorphous bodies".  There are no approximations in this formula beyond the 1st Born approximation; it is not a
    small-angle formula.

    Note that you need to multiply this by the electron density of the sphere if you want reasonable amplitudes.
    E.g., water molecules have 10 electrons, a molecular weight of 18 g/mol and a density of 1 g/ml, so you can google
    search the electron density of water, which is 10*(1 g/cm^3)/(18 g/6.022e23) = 3.346e29 per m^3 .

    Arguments:
        radius (float): In SI units of course.
        q_mags (numpy array): Also in SI units.
        check_divide_by_zero (bool): Check for divide by zero.  True by default.

    Returns: numpy array
    """

    qr = q_mags*radius
    if check_divide_by_zero is True:
        amp = np.zeros_like(qr)
        amp[qr == 0] = (4*np.pi*radius**3)/3
        w = qr != 0
        amp[w] = 4 * np.pi * radius ** 3 * (np.sin(qr[w]) - qr[w] * np.cos(qr[w])) / qr[w] ** 3
    else:
        amp = 4 * np.pi * radius ** 3 * (np.sin(qr) - qr * np.cos(qr)) / qr ** 3
    return amp
