import json
import numpy as np
import spglib


ops = []
for i in range(530):
    ops.append(spglib.get_symmetry_from_database(i+1))
np.savez('spacegroups.npz', ops=ops, allow_pickle=True)

def sym_ops_by_hall_number(hall_number):
    r""" Get spacegroup symmetry operators by Hall number.  """
    return np.load('spacegroups.npz', allow_pickle=True)['ops'][hall_number-1]

print(ops[50])
print(sym_ops_by_hall_number(51))
