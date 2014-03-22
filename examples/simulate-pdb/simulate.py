from Bio import PDB
from pydiffract import structure, utils
import numpy as np
import matplotlib.pyplot as plt

pdbFile = "4H92.pdb"

mol = structure.molecule()

mol.fromPdb(pdbFile)

Z = mol.Z
f = mol.f(5000.0 * utils.joulesPerEv)


# plt.scatter(mol.r[:, 0], mol.r[:, 1])
# plt.show()
