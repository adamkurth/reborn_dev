
# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.
import pkg_resources
import numpy as np
from scipy import signal
from .. import utils, const, detector
r_e = const.r_e
eV = const.eV


hura_data = pkg_resources.resource_filename('reborn', 'data/scatter/water/hura.txt')
clark_data = pkg_resources.resource_filename('reborn', 'data/scatter/water/clark.txt')
water_data = pkg_resources.resource_filename('reborn', 'data/scatter/water/water.dat')

def water_number_density():
    """ Number density of water in SI units. """
    return 33.3679e27


# @utils.memoize
def _get_hura_water_data():
    with open(hura_data, 'r') as f:
        h = f.readline()
    h = h.split()[1:-1]
    temperatures = np.array([float(v) for v in h]) + 273.16
    d = np.loadtxt(hura_data, skiprows=1)
    Q = d[:, 0]
    # errors = d[:, -1]
    intensities = d[:, 1:-1]
    return 1e10 * Q.copy(), intensities.copy(), temperatures.copy()


def _get_clark_water_data():
    with open(clark_data, 'r') as f:
        h = f.readline()
    h = h.split()[1:]
    temperatures = np.array([float(v) for v in h])
    d = np.loadtxt(clark_data, skiprows=1)
    Q = d[:, 0]
    intensities = d[:, 1:]
    return 1e10 * Q.copy(), intensities.copy(), temperatures.copy()


# @utils.memoize
def _get_hura_water_data_smoothed():
    Q, intensities, temperatures = _get_hura_water_data()
    for i in range(intensities.shape[1]):
        p = intensities[:, i]
        intensities[:, i] = signal.savgol_filter(p, window_length=21, polyorder=2)
    return Q.copy(), intensities.copy(), temperatures.copy()


def _get_hura_interpolated(q, temperature=298, volume=None, smoothed=False):
    if smoothed:
        qmag, intensity, temp = _get_hura_water_data_smoothed()
    else:
        qmag, intensity, temp = _get_hura_water_data()
    temp_idx = np.interp(temperature, temp, np.arange(temp.size))
    if temp_idx <= 0:
        Iavg = intensity[:, 0]
    elif temp_idx >= (temp.size-1):
        Iavg = intensity[:, 0]
    else:
        a = temp_idx % 1
        Iavg = (1 - a) * intensity[:, int(np.floor(temp_idx))] + a * intensity[:, int(np.ceil(temp_idx))]
    F2 = np.interp(q, qmag, Iavg)
    if volume is not None:
        F2 *= volume * water_number_density()
    return F2


def _get_clark_interpolated(q, temperature=298, volume=None):
    qmag, intensity, temp = _get_clark_water_data()
    temp_idx = np.interp(temperature, temp, np.arange(temp.size))
    if temp_idx <= 0:
        Iavg = intensity[:, 0]
    elif temp_idx >= (temp.size-1):
        Iavg = intensity[:, 0]
    else:
        a = temp_idx % 1
        Iavg = (1 - a) * intensity[:, int(np.floor(temp_idx))] + a * intensity[:, int(np.ceil(temp_idx))]
    F2 = np.interp(q, qmag, Iavg)
    if volume is not None:
        F2 *= volume * water_number_density()
    return F2


def _get_water_interpolated(q, temperature=298, volume=None):
    with open(water_data, 'r') as f:
        h = f.readline()
    h = h.split()[1:]
    temp = np.array([float(v) for v in h]) + 273.16
    d = np.loadtxt(water_data, skiprows=1)
    qmag = d[:, 0]*1e10
    intensity = d[:, 1:]
    print(intensity.shape)
    temp_idx = np.interp(temperature, temp, np.arange(temp.size))
    if temp_idx <= 0:
        Iavg = intensity[:, 0]
    elif temp_idx >= (temp.size-1):
        Iavg = intensity[:, temp.size-1]
    else:
        a = temp_idx % 1
        Iavg = (1 - a) * intensity[:, int(np.floor(temp_idx))] + a * intensity[:, int(np.ceil(temp_idx))]
    F2 = np.interp(q, qmag, Iavg)
    if volume is not None:
        F2 *= volume * water_number_density()
    return F2


def get_water_profile(q, temperature=298):
    """
    Depreciated: Use :func:`water_scattering_factor_squared <reborn.simulate.solutions.water_scattering_factor_squared>`
    """
    utils.depreciate("Use the function water_scattering_factor_squared instead of get_water_profile.")
    return water_scattering_factor_squared(q=q, temperature=temperature, volume=None)


def water_scattering_factor_squared(q, temperature=298, volume=None):
    """
    Get water scattering profile :math:`|F(q)|^2` via interpolations/smoothing of the data from Hura et al.
    2003 (Phys Chem 5, 1981) and Clark et al. 2010 (PNAS 107, 14003–14007).  Temperatures from 1-77 C are
    included.  Note that the returned values are the scattering factors :math:`|F(q)|^2` per water molecule,
    so you should multiply this by the number of water molecules exposed to the x-ray beam.  The number density of
    water is approximately 33.3679e27 / m^3.  You may optionally provide the volume and the result will be multiplied by
    the number of water molecules.

    Arguments:
        q (|ndarray|) : The momentum transfer vector magnitudes.
        temperature (float): Desired water temperature in Kelvin.
        volume (float): If provided, the scattering factor will be multiplied by the number of molecules in this volume.

    Returns:
        |ndarray| : The scattering factor :math:`|F(q)|^2` , possibly multiplied by the number of water molecules if you
                    specified a volume.
    """
    # Interpolate profiles according to temperature (weighted average)
    # If out of range, choose nearest temperature
    return _get_water_interpolated(q, temperature=temperature, volume=volume)
    # qmag, intensity, temp = _get_hura_water_data_smoothed()
    # temp_idx = np.interp(temperature, temp, np.arange(temp.size))
    # if temp_idx <= 0:
    #     Iavg = intensity[:, 0]
    # elif temp_idx >= (temp.size-1):
    #     Iavg = intensity[:, 0]
    # else:
    #     a = temp_idx % 1
    #     Iavg = (1 - a) * intensity[:, int(np.floor(temp_idx))] + a * intensity[:, int(np.ceil(temp_idx))]
    # F2 = np.interp(q, qmag, Iavg)
    # if volume is not None:
    #     F2 *= volume * water_number_density()
    # return F2


def get_pad_solution_intensity(pad_geometry, beam, thickness=10.0e-6, liquid='water', temperature=298.0, poisson=True):
    r"""
    Given a list of |PADGeometry| instances along with a |Beam| instance, calculate the scattering intensity
    :math:`I(q)` of a liquid of given thickness.

    note::

        This function is only for convenience.  Consider using
        :func:`water_scattering_factor_squared <reborn.simulate.solutions.water_scattering_factor_squared>` if speed
        is important; this will avoid re-calculating q magnitudes, solid angles, and polarization factors.

    Args:
        pad_geometry (list of |PADGeometry| instances): PAD geometry info.
        beam (|Beam|): X-ray beam parameters.
        thickness (float): Thickness of the liquid (assumed to be a sheet geometry)
        liquid (str): We can only do "water" at this time.
        temperature (float): Temperature of the liquid.
        poisson (bool): If True, add Poisson noise (default=True)

    Returns:
        List of |ndarray| instances containing intensities.
    """
    if liquid != 'water':
        raise ValueError('Sorry, we can only simulate water at this time...')
    volume = thickness * np.pi * (beam.diameter_fwhm / 2) ** 2
    pads = detector.PADGeometryList(pad_geometry)
    q_mags = pads.q_mags(beam)
    F2 = water_scattering_factor_squared(q_mags, temperature=temperature, volume=volume)
    intensity = F2 * pads.f2phot(beam)
    if poisson:
        intensity = np.double(np.random.poisson(intensity))
    intensity = pads.split_data(intensity)
    return intensity
