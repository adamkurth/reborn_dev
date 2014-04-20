# from Bio import PDB
from pydiffract import structure, detector
from pydiffract.utils import joulesPerEv, hc, vecNorm
from pydiffract.simulate import cycore, clcore, simulate
import numpy as np
# from numpy.random import random, randn
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
det.beam.spectralWidth = 0.001
det.beam.divergence = 0.001

# t = time.time()
# ph = cycore.phaseFactor(q, r)
# elapsed = time.time() - t
# print(elapsed)


F2 = simulate.simulateMolecule(mol, det)

dispim = F2.reshape(nS, nF)
dispim = np.log(dispim)
plt.imshow(dispim, interpolation='nearest')
plt.show()

# fig, ax = plt.subplots()
# ax.scatter(i, j, marker='.')
# ax.set_yticks(np.arange(-0.5, 99.5), minor=False)
# ax.set_xticks(np.arange(-0.5, 99.5), minor=False)
# ax.yaxis.grid(True, which='major')
# ax.xaxis.grid(True, which='major')
# plt.show()


# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1)
# fig.clf()
# ax = Axes3D(fig)
# ax.plot(q[:, 0], q[:, 1], q[:, 2], '.')
# plt.show()