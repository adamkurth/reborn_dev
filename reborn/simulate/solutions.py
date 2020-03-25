from __future__ import (absolute_import, division, print_function, unicode_literals)

import pkg_resources
import numpy as np
from reborn.utils import memoize
# import matplotlib.pyplot as plt

file_name = pkg_resources.resource_filename('reborn', 'data/scatter/water_scattering_data.txt')



def water_number_density():
    return 33.3679e27

@memoize
def load_data():

    with open(file_name, 'r') as f:
        h = f.readline()
    h = h.split()[1:-1]
    temperatures = np.array([float(v) for v in h]) + 273.16

    d = np.loadtxt(file_name, skiprows=1)
    Q = d[:, 0]
    errors = d[:, -1]
    intensities = d[:, 1:-1]

    return 1e10 * Q, intensities, temperatures


def get_water_profile(q, temperature=298):
    """
    Get water scattering profile from Greg Hura's PhD thesis data.  Interpolates the data for given
    q-vector magnitudes and temperature.  Temperatures range from 1, 11, 25, 44, 55, 66, 77 degrees
    Celcius.  Bornagain, of course, uses SI units, including for temperature.  To go from Celcius to Kelvin, add 273.16.
    Note that this is the scattering factor F for a single water molecule.  The number density of water is approximately
    n = 33.3679e27 / m^3 .

    Arguments:
        q: The momentum transfer vector magnitudes (i.e. 2 pi sin(theta/lambda)/wavelength )
        temperature: Desired water temperature in Kelvin
        volume: if this is not None, the scattering factor will be multiplied by the number of molecules in volume

    Returns:


    """

    qmag, intensity, temp = load_data()
    n = len(temp)
    out_of_range = False
    for i in range(n):
        if temperature < temp[i]:
            break
        if i == n:
            out_of_range = True

    if i == 0:
        Iavg = intensity[:, 0]
    elif out_of_range:
        Iavg = intensity[:, n - 1]
    else:
        Tl = temp[i - 1]
        Tr = temp[i]
        DT = Tr - Tl
        dl = (temperature - Tl) / DT
        dr = (Tr - temperature) / DT
        Iavg = dr * intensity[:, i - 1] + dl * intensity[:, i]

    II = np.interp(q, qmag, Iavg)

    return II


def water_scattering_factor_squared(q, temperature=298, volume=None):
    """
    Get water scattering profile from Greg Hura's PhD thesis data.  Interpolates the data for given
    q-vector magnitudes and temperature.  Temperatures range from 1, 11, 25, 44, 55, 66, 77 degrees
    Celcius.  Bornagain, of course, uses SI units, including for temperature.  To go from Celcius to Kelvin, add 273.16.
    Note that this is the scattering factor F for a single water molecule.  The number density of water is approximately
    n = 33.3679e27 / m^3 .

    Arguments:
        q: The momentum transfer vector magnitudes (i.e. 2 pi sin(theta/lambda)/wavelength )
        temperature: Desired water temperature in Kelvin
        volume: if this is not None, the scattering factor will be multiplied by the number of molecules in volume

    Returns:


    """

    qmag, intensity, temp = load_data()
    n = len(temp)
    out_of_range = False
    for i in range(n):
        if temperature < temp[i]:
            break
        if i == n:
            out_of_range = True

    if i == 0:
        Iavg = intensity[:, 0]
    elif out_of_range:
        Iavg = intensity[:, n - 1]
    else:
        Tl = temp[i - 1]
        Tr = temp[i]
        DT = Tr - Tl
        dl = (temperature - Tl) / DT
        dr = (Tr - temperature) / DT
        Iavg = dr * intensity[:, i - 1] + dl * intensity[:, i]

    F2 = np.interp(q, qmag, Iavg)**2

    if volume is not None:
        F2 *= volume * water_number_density

    return F2
