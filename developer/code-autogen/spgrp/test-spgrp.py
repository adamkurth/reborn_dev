import sys
sys.path.append("../..")
from bornagain.target import crystal, spgrp
import spglib
import numpy as np

n = 230
for i in range(1,n+1):
    hmsym = spgrp._hmsym[i-1]
    print(i,hmsym)
    Rs, Ts = crystal.get_symmetry_operators_from_space_group(hmsym)
    symops = spglib.get_symmetry_from_database(i)
    for j in range(0,len(Ts)):
        a = np.max(np.abs(Rs[j] - symops['rotations'][j]))
        if a != 0:
            print(Rs[j])
            print(symops['rotations'][j])
        b = np.max(np.abs(Ts[j] - symops['translations'][j]))
        if b != 0:
            print(Ts[j])
            print(symops['translations'][j])
