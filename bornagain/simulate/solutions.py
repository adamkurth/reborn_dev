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
    temperatures = np.array([float(v) for v in h])

    d = np.loadtxt(file_name, skiprows=1)
    Q = d[:, 0]
    errors = d[:, -1]
    intensities = d[:, 1:-1]

    return 1e10*Q/np.pi/2.0, intensities, temperatures


def get_water_profile(q, temperature=30):

    """
    Get water scattering profile from Greg Hura's PhD thesis data.  Interpolates the data for given
    q-vector magnitudes and temperature.  Temperatures range from 1, 11, 25, 44, 55, 66, 77 degrees
    Celcius.

    Args:
        q: The momentum transfer vectors
        temperature: Desired water temperature

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
        Iavg = I[:, n-1]
    else:
        Tl = T[i-1]
        Tr = T[i]
        DT = Tr - Tl
        dl = (temperature - Tl)/DT
        dr = (Tr - temperature)/DT
        Iavg = dr*I[:, i-1] + dl*I[:, i]
        # print(Tl, Tr, dl, dr)

    II = np.interp(q, Q, Iavg)
    return II









# for i in range(1,len(temperatures)):
#     plt.plot(Q,np.log10(intensities[:,i]), label=('%2g $^\circ$C' % temperatures[i-1]))
#
# plt.legend()
# plt.xlabel('Q')
# plt.ylabel('Intensity')
# plt.show()
# print(temperatures)
# print(Q)