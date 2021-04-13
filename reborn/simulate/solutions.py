
import pkg_resources
import numpy as np
import reborn

file_name = pkg_resources.resource_filename('reborn', 'data/scatter/water_scattering_data.txt')


def water_number_density():
    """ Number density of water in SI units. """
    return 33.3679e27


@reborn.utils.memoize
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


def get_pad_solution_intensity(pad_geometry, beam, thickness=10e-6, liquid='water', temperature=298, poisson=True):
    r"""
    Given a list of |PADGeometry| instances along with a |Beam| instance, calculate the scattering intensity of a
    liquid with given thickness.

    Args:
        pad_geometry (list of |PADGeometry| instances): PAD geometry info.
        beam (|Beam|): X-ray beam parameters.
        thickness (float): Thickness of the liquid (assumed to be a sheet geometry)
        liquid (str): We can only do "water" at this time...
        temperature (float): Temperature of the liquid.

    Returns:
        List of |ndarray| instances containing intensities.
    """
    if liquid != 'water':
        raise ValueError('Sorry, we can only simulate water at this time...')
    pad_geometry = reborn.utils.ensure_list(pad_geometry)
    n_water_molecules = thickness * beam.diameter_fwhm ** 2 * water_number_density()
    qmags = reborn.detector.concat_pad_data([p.q_mags(beam=beam) for p in pad_geometry])
    J = beam.fluence
    P = reborn.detector.concat_pad_data([p.polarization_factors(beam=beam) for p in pad_geometry])
    SA = reborn.detector.concat_pad_data([p.solid_angles() for p in pad_geometry])
    F = get_water_profile(qmags, temperature=temperature)
    F2 = F ** 2 * n_water_molecules
    I = 2.8179403262e-15 ** 2 * J * P * SA * F2 / beam.photon_energy
    if poisson: I = np.double(np.random.poisson(I))
    I = reborn.detector.split_pad_data(pad_geometry, I)
    return I