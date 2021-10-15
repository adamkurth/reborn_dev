import numpy as np
from .. import utils
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
        photon_energy (|float|): Photon energy.
        molecule (reborn.target.molecule.Molecule): Molecule (overrides r_vecs and atomic_numbers)
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


# def get_pad_gas_intensity(pad_geometry, beam, gas='air', temperature=298.0, pressure=101325,
#                           poisson=True):
#
