from Bio import PDB
from pydiffract import structure, utils, detector
import numpy as np
import matplotlib.pyplot as plt
from pydiffract.simulate import cycore, clcore
from pydiffract import utils
import time



pdbFile = "4H92.pdb"

mol = structure.molecule()

mol.fromPdb(pdbFile)

nF = 100
nS = 100
det = detector.panelList()
det.simpleSetup(nF=nF, nS=nS, pixSize=100e-6, distance=0.05, wavelength=2e-10)

q = det.Q
r = mol.r
f = mol.f(8000 * utils.joulesPerEv)

t = time.time()
ph = cycore.phaseFactor(q, r)
elapsed = time.time() - t
print(elapsed)

t = time.time()
ph = clcore.phaseFactor(q, r)
elapsed = time.time() - t
print(elapsed)

I = np.abs(ph) ** 2
I = I.reshape(nS, nF)
dispim = I
dispim = np.log(dispim)
plt.imshow(dispim, interpolation='nearest')
plt.show()





# plt.scatter(mol.r[:, 0], mol.r[:, 1])
# plt.show()
