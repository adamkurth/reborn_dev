# from Bio import PDB
from pydiffract import structure, utils, detector
from pydiffract.utils import joulesPerEv, hc
from pydiffract.simulate import cycore, clcore
import numpy as np
import matplotlib.pyplot as plt
import time
import xraylib


pdbFile = "4H92.pdb"

mol = structure.molecule()

mol.fromPdb(pdbFile)

nF = 1000
nS = 1000
det = detector.panelList()
det.simpleSetup(nF=nF, nS=nS, pixSize=100e-6, distance=0.05, wavelength=2e-10)

q = det.Q
r = mol.r
# f = mol.f(8000 * utils.joulesPerEv)

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
    gr, gZ = mol.groupedElements()

    for (r, Z) in zip(gr, gZ):

        t = time.time()
        ph = clcore.phaseFactor(q, r)
        elapsed = time.time() - t
        print("Z = %3d (%6d atoms) in %.3f seconds" % (Z, len(r), elapsed))
        F += ph * atomicF(Z, E, det.stol)

    return F


# t = time.time()
# ph = cycore.phaseFactor(q, r)
# elapsed = time.time() - t
# print(elapsed)

F = simulateMolecule(mol, det)

I = np.abs(F) ** 2
I = I.reshape(nS, nF)
dispim = I
dispim = np.log(dispim)
plt.imshow(dispim, interpolation='nearest')
plt.show()





# plt.scatter(mol.r[:, 0], mol.r[:, 1])
# plt.show()
