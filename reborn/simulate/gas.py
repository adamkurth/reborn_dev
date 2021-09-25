import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

from reborn import utils
from reborn.target import crystal, atoms

try: 
    import numba as nb
    numba = True
    #suppress some unnecessary deprecation warnings
    #though there is still an OMP deprecation warning I can't seem to get rid of
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
    import warnings
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
except:
    numba = False


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
    rij = spatial.distance.squareform(spatial.distance.pdist(r_vecs))
    max_dist = rij.max()
    if max_dist == 0:
        max_dist = 2e-10  # What is the appropriate Shannon sampling for a single atom?
    n_atoms = r_vecs.shape[0]
    wsh = np.pi / max_dist  # Width of Shannon sample
    nsh = np.ceil(q_mags.max() / wsh).astype(int) + 5  # Number of Shannon samples (plus five)
    qsh = (np.arange(nsh) + 1) * wsh  # q magnitudes for each Shannon sample
    ff = np.zeros((n_atoms, nsh), dtype=np.complex)  # FIXME: This is a highly redundant array
    unique_atomic_numbers = np.unique(atomic_numbers)
    max_z = np.max(unique_atomic_numbers)
    z_map = np.zeros(max_z)  # This maps atomic number to index of scatter factor
    ff = np.empty((max_z, nsh), dtype=np.complex)
    for i in range(n_atoms):
        ff[i, :] = atoms.cmann_henke_scattering_factors(q_mags=qsh, atomic_number=atomic_numbers[i],
                                                        photon_energy=photon_energy)
    if numba:
        intensity_shannon = debye_nb(rij, qsh, ff)
    else:
        intensity_shannon = debye(rij, qsh, ff)
    N = np.arange(nsh)+1
    N = N[:, None]
    kernel = (N*np.pi)**2 * np.sinc(q_mags*max_dist/np.pi) * (-1)**(N+1) / ((N*np.pi)**2-(q_mags*max_dist)**2)
    intensity = 2*np.einsum('k,ki->i', intensity_shannon, kernel)
    return intensity


def debye(rij, q, ff):
    """
    FIXME: There is something wrong with this function.  It does not produce the same output as the numba version.
    Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.
    This function is slower than the similar function implemented with numba.
    rij - distance matrix, ie. output from scipy.spatial.distance.pdist(pdb.coords) (after squareform)
    q - q values to use for calculations..
    ff - an array of form factors calculated for each atom in a pdb object. q's much match q array.
    """
    s = np.sinc(q * rij[..., None] / np.pi)
    I = np.einsum('iq,jq,ijq->q', ff, ff, s)
    return I


if numba:
    @nb.njit(fastmath=True, parallel=True, error_model="numpy", cache=True)
    def debye_nb(rij, q, ff):
        """Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.
        This function is intended to be used with the numba njit decorator for speed.
        rij - distance matrix, ie. output from scipy.spatial.distance.pdist(pdb.coords) (after squareform)
        q - q values to use for calculations.
        ff - an array of form factors calculated for each atom in a pdb object. q's much match q array.
        """
        nr = rij.shape[0]
        nq = q.shape[0]
        I = np.empty(nq)
        ff_T = np.ascontiguousarray(ff.T)
        for qi in nb.prange(nq):
            acc = 0
            for ri in range(nr):
                for rj in range(nr):
                    #acc += ff[ri,qi]*ff[rj,qi]*np.sinc(q[qi]*rij[ri,rj]/np.pi)
                    if q[qi]*rij[ri, rj] != 0:
                        acc += ff_T[qi, ri]*ff_T[qi, rj]*np.sin(q[qi]*rij[ri, rj])/(q[qi]*rij[ri, rj])
                    else:
                        acc += ff_T[qi, ri]*ff_T[qi, rj]
            I[qi] = np.abs(acc)
        return I
