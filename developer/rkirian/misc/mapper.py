import sys
sys.path.append('../..')
from time import time
import numpy as np
import matplotlib.pyplot as plt
from reborn.target import density, crystal

cryst = crystal.CrystalStructure()
cryst.set_spacegroup('P 1')
a = .3e-9
b = .3e-9
c = .3e-9
cryst.set_cell(a, b, c, 90 * np.pi / 180, 90 * np.pi / 180, 90 * np.pi / 180)

mt = density.CrystalMeshTool(cryst, 0.05e-9, 3)
print(mt.N)
print(mt.s)
print(np.max(mt.get_x_vecs()))

atom_x = np.array([[0, 0, 0], [.3, .4, .5]])
fs = np.array([1, 1])
t = time()
fmap = mt.place_atoms_in_map(atom_x, fs)
print(time() - t)
t = time()
fmap = mt.place_atoms_in_map(atom_x, fs)
print(time() - t)
fmap = mt.reshape(fmap)
plt.imshow(np.sum(np.abs(fmap), axis=0))
plt.show()