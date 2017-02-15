'''
Created on Feb 15, 2017

@author: rkirian
'''

import sys

sys.path.append("../..")
from bornagain import simulate as sim

sym = sim.atoms.atomic_symbols
print(sym)
print(sim.atoms.get_atomic_numbers(sym))

num = range(1,119)
print(num)
print(sim.atoms.get_atomic_symbols(num))
print(len(sim.atoms.get_atomic_symbols(num)))

