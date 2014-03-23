from Bio import PDB
from pydiffract import structure, utils, detector
import numpy as np
import matplotlib.pyplot as plt
from pydiffract.simulate import core
import time

pdbFile = "4H92.pdb"

mol = structure.molecule()

mol.fromPdb(pdbFile)

nF = 1000
nS = 1000
det = detector.panelList()
det.simpleSetup(nF=nF, nS=nS, pixSize=100e-6, distance=0.05, wavelength=2e-10)

q = det.Q
r = mol.r
nr = mol.nAtoms

t = time.time()
ph = core.phaseFactor(q, r)
elapsed = time.time() - t
print(elapsed)

I = np.abs(ph) ** 2
I = I.reshape(nS, nF)
plt.imshow(np.log(I), interpolation='none')
plt.show()


# plt.scatter(mol.r[:, 0], mol.r[:, 1])
# plt.show()
