from bornagain.utils import joulesPerEv, hc
# from bornagain.simulate import cycore, clcore
import bornagain.simulate.clcore as core
import numpy as np
import xraylib


def atomicFormFactor(Z, stol):

    """ Atomic form factor, a function of atomic number and
        sin(theta/2)/lambda . """

    return np.array([xraylib.FF_Rayl(Z, s) for s in stol * 1e-10])

def atomicFi(Z, E):

    """ Atomic scattering factor correction to real part. """

    return xraylib.Fi(Z, E / joulesPerEv)

def atomicFii(Z, E):

    """ Atomic scattering factor correction to complex part. """

    return xraylib.Fii(Z, E / joulesPerEv)

def atomicF(Z, E, stol):

    """ Atomic form factor with anamalous contributions. A function of atomic
        number, photon energy, and sin(theta/2)/lambda. """

    return atomicFormFactor(Z, stol) + atomicFi(Z, E) + 1j * atomicFii(Z, E)

def simulateMolecule(mol, det):

    F = 0
    E = hc / det.beam.wavelength
    stol = det.stol
    gr, gZ = mol.groupedElements()
    q = det.Q

    for (r, Z) in zip(gr, gZ):

#         t = time.time()
        ph = core.phaseFactor(q, r)
        F += ph * atomicF(Z, E, stol)
#         elapsed = time.time() - t
#         print("Z = %3d (%6d atoms) in %.3f seconds" % (Z, len(r), elapsed))

    return np.abs(F) ** 2