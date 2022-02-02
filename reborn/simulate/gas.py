import numpy as np
from .. import utils, const, detector
from ..target import crystal, atoms
from ..target.molecule import Molecule
from ..fortran import scatter_f


def isotropic_gas_intensity_profile(r_vecs=None, q_mags=None, atomic_numbers=None, photon_energy=None, molecule=None,
                                    beam=None):
    r""" Calculate the isotropic scatter from a gas molecule.  Taken from Tom Grant's DENSS software.

    Arguments:
        r_vecs (|ndarray|): Position vectors.
        q_mags (|ndarray|): Q vectors.
        atomic_numbers (|ndarray|): Atomic numbers.
        photon_energy (float): Photon energy.
        molecule (reborn.target.molecule.Molecule or str): Molecule (overrides r_vecs and atomic_numbers)
        beam (|Beam|): Beam instance.  Overrides photon_energy.

    Returns:
        |ndarray|: Intensity profile I(q)
    """
    if molecule is not None:
        if isinstance(molecule, str):
            m = molecule.lower()
            if m == "n2":
                molecule = Molecule(coordinates=np.array([[0, 0, 0], [0, 0, 1.07e-10]]), atomic_numbers=[7, 7])
            if m == "o2":
                molecule = Molecule(coordinates=np.array([[0, 0, 0], [0, 0, 1.21e-10]]), atomic_numbers=[8, 8])
            if m == "he":
                molecule = Molecule(coordinates=np.array([[0, 0, 0]]), atomic_numbers=[2])
        r_vecs = molecule.coordinates
        atomic_numbers = molecule.atomic_numbers
    if beam is not None:
        photon_energy = beam.photon_energy
    atomic_numbers = utils.atleast_1d(np.array(atomic_numbers))
    if atomic_numbers.size == 1:
        f = atoms.cmann_henke_scattering_factors(q_mags=q_mags, atomic_number=atomic_numbers[0], photon_energy=photon_energy)
        return np.abs(f)**2
    r_vecs = utils.atleast_2d(r_vecs)  # Make sure it works with a single atom, just to keep things general
    q_mags = np.float64(q_mags)
    uz = np.sort(np.unique(atomic_numbers))
    f_idx = np.sum(np.greater.outer(atomic_numbers, uz), 1).astype(int)  # Map atom type to scattering factor
    ff = np.zeros((uz.size, q_mags.size), dtype=np.complex128)  # Scatter factor array.  One row for each unique atom type.
    for i in range(uz.size):
        ff[i, :] = atoms.cmann_henke_scattering_factors(q_mags=q_mags, atomic_number=uz[i], photon_energy=photon_energy)
    intensity = np.zeros(q_mags.size, dtype=np.float64)
    r_vecs = np.ascontiguousarray(r_vecs).astype(np.float64)
    scatter_f.debye(r_vecs.T, q_mags, f_idx, ff.T, intensity)
    return np.real(intensity)


def air_intensity_profile(q_mags=None, beam=None):
    r""" Calls isotropic_gas_intensity_profile and sums the contributions from O2 and N2. Read the docs for """
    n2_profile = isotropic_gas_intensity_profile(molecule='N2', beam=beam, q_mags=q_mags)
    o2_profile = isotropic_gas_intensity_profile(molecule='O2', beam=beam, q_mags=q_mags)
    return n2_profile*0.79 + o2_profile*0.21


def get_gas_background(pad_geometry,
                        beam,
                        path_length=[0.0, 1.0],
                        gas_type:str='he',
                        temperature:float=293.15,
                        pressure:float=101325.0,
                        n_simulation_steps:int=20,
                        poisson:bool=False):

    r"""
    Given a list of |PADGeometry| instances along with a |Beam| instance, calculate the scattering intensity
    :math:`I(q)` of a helium of given path length.

    Args:
        pad_geometry (list of |PADGeometry| instances): PAD geometry info.
        beam (|Beam|): X-ray beam parameters.
        path_length (list of |float|): Path length of gas the beam is 'propagating' through
        liquid (str): We can only do "water" at this time.
        temperature (float): Temperature of the gas.
        poisson (bool): If True, add Poisson noise (default=True)

    Returns:
        List of |ndarray| instances containing intensities.
    """


    gas_options = ['he', 'helium', 'air']
    if gas_type not in gas_options:
        raise ValueError(f'Sorry, the only options are {gas_options}. Considering writing your own function for other gases.')

    pads = detector.PADGeometryList(pad_geometry)
    pads0 = pads
    pads = pads0.copy()
    q_mags = pads.q_mags(beam)

    for i in range(2):
        if path_length[i] == 0:
            path_length[i] = 1e-6  # Avoid values close to the detector.

    iter_list = np.linspace(path_length[0], path_length[1], n_simulation_steps)
    dx = iter_list[1] - iter_list[0]

    volume = np.pi * dx * (beam.diameter_fwhm/2) ** 2  # volume of a cylinder
    n_molecules = pressure * volume / (const.k * temperature)

    # initialize a zeros array the same shape as the detector
    I_total = pads.zeros()

    alpha = const.r_e ** 2 * beam.photon_number_fluence
    for step in iter_list:
        print(step)
        for (p0, p) in zip(pads0, pads):  # change the detector distance
            p.t_vec = p0.t_vec - beam.beam_vec * step
        ang = pads.scattering_angles(beam)
        mask = np.ones(pads.n_pixels)
        mask[ang >= np.pi/2*0.98] = 0
        polarization = pads.polarization_factors(beam=beam)  # calculate the polarization factors
        solid_angles = pads.solid_angles2()  # Approximate solid angles
        scatt = isotropic_gas_intensity_profile(molecule='He', beam=beam, q_mags=q_mags)  # 1D intensity profile
        F2 = np.abs(scatt) ** 2 * n_molecules
        I = alpha * polarization * solid_angles * F2  # calculate the scattering intensity
        I_total += I*mask  # sum the results

    if poisson:
        I_total = np.random.poisson(I_total).astype(np.double)  # add in some Poisson noise for funsies

    return I_total
