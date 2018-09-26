import pkg_resources
import numpy as np
from bornagain.utils import memoize
# import matplotlib.pyplot as plt

file_name = pkg_resources.resource_filename('bornagain.simulate',
                                            'data/water_scattering_data.txt')


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

    Args:
        q: The momentum transfer vector magnitudes (i.e. 2 pi sin(theta/lambda)/wavelength )
        temperature: Desired water temperature in Kelvin

    Returns:


    """

    Q, I, T = load_data()
    n = len(T)
    out_of_range = False
    for i in range(n):
        if temperature < T[i]:
            break
        if i == n:
            out_of_range = True

    if i == 0:
        Iavg = I[:, 0]
    elif out_of_range:
        Iavg = I[:, n - 1]
    else:
        Tl = T[i - 1]
        Tr = T[i]
        DT = Tr - Tl
        dl = (temperature - Tl) / DT
        dr = (Tr - temperature) / DT
        Iavg = dr * I[:, i - 1] + dl * I[:, i]

    II = np.interp(q, Q, Iavg)
    return II
