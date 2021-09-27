import numpy as np
from .. import utils
from ..target import crystal, atoms
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
        r_vecs = molecule.coordinates
        atomic_numbers = molecule.atomic_numbers
    if beam is not None:
        photon_energy = beam.photon_energy
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
