import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

from .. import utils
from ..target import crystal, atoms

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
    r_vecs = utils.atleast_2d(r_vecs)
    rij = spatial.distance.squareform(spatial.distance.pdist(r_vecs))
    max_dist = rij.max()
    if max_dist == 0:
        max_dist = 2e-10
    n_atoms = r_vecs.shape[0]
    wsh = np.pi / max_dist
    nsh = np.ceil(q_mags.max() / wsh).astype(int) + 5
    qsh = (np.arange(nsh) + 1) * wsh
    ff = np.zeros((n_atoms, nsh), dtype=np.complex)
    for i in range(n_atoms):
        ff[i, :] = atoms.cmann_henke_scattering_factors(q_mags=qsh, atomic_number=atomic_numbers[i],
                                                        photon_energy=photon_energy)
    if numba:
        intensity_shannon = _pdb2sas_nb(rij, qsh, ff)
    else:
        intensity_shannon = _pdb2sas(rij, qsh, ff)
    intensity = Ish2Iq(Ish=intensity_shannon, D=max_dist, q=q_mags)
    return intensity


def Ish2Iq(Ish, D, q=(np.arange(500)+1.)/1000):
    """Calculate I(q) from intensities at Shannon points."""
    n = len(Ish)
    N = np.arange(n)+1
    denominator = (N[:, None]*np.pi)**2-(q*D)**2
    I = 2*np.einsum('k,ki->i', Ish, (N[:, None]*np.pi)**2 / denominator * np.sinc(q*D/np.pi) * (-1)**(N[:, None]+1))
    return I


def _pdb2sas(rij, q, ff):
    """
    FIXME: There is something wrong with this function.  It does not produce the same output as the numba version.
    Calculate the scattering of an object from a set of 3D coordinates using the Debye formula.
    This function is slower than the similar function implemented with numba.
    rij - distance matrix, ie. output from scipy.spatial.distance.pdist(pdb.coords) (after squareform)
    q - q values to use for calculations..
    ff - an array of form factors calculated for each atom in a pdb object. q's much match q array.
    """
    s = np.sinc(q * rij[...,None]/np.pi)
    I = np.einsum('iq,jq,ijq->q',ff,ff,s)
    return I


if numba:
    @nb.njit(fastmath=True, parallel=True, error_model="numpy", cache=True)
    def _pdb2sas_nb(rij, q, ff):
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


if __name__ == "__main__":
    # Configuration
    pdb_id = '1LYZ'
    cryst = crystal.CrystalStructure(pdb_id)
    mol = cryst.molecule
    q_mags = np.linspace(0, 1, 1001) * 1e10
    photon_energies = np.linspace(280, 290, 20)*1.602e-19
    # Fancy colors for plotting
    n_colors = len(photon_energies)
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / n_colors) for i in range(n_colors)])
    # Calculate and plot
    for E in photon_energies:
        Iq = isotropic_gas_intensity_profile(molecule=cryst.molecule, q_mags=q_mags, photon_energy=E)
        plt.semilogy(q_mags, Iq, label=('%6.2f' % E))
        # Save files?
        if False:
            np.savetxt('Iq_%6.2f.dat' % E, Iq, delimiter=' ', fmt='%.8e')
            np.savetxt('Pr_%6.2f.dat' % E, Pr, delimiter=' ', fmt='%.8e')
    plt.legend()
    plt.show()
