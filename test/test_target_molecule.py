import numpy as np
from reborn.target import molecule


def test_01():
    mol = molecule.Molecule(coordinates=[1, 2, 3], atomic_numbers=[1, 2, 3])
    assert(np.abs(mol.get_atomic_weights()[0] - 1.67714446e-27) < 1e-6)
