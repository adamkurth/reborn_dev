import numpy as np
from bornagain import simulate as sim
from time import time

def test_element_numbers_and_symbols():
        
    """Round trip conversion of atomic numbers to element symbols"""
        
    Z1 = np.arange(1,119)
    sym = sim.atoms.atomic_numbers_to_symbols(Z1)
    Z2 = sim.atoms.atomic_symbols_to_numbers(sym)
    
    assert(all(Z1 == Z2))
    
def test_load_henke_data(atomic_number):
    
    for i in range(1,10):
        t = time()
        data = sim.atoms.get_henke_data(atomic_number)
        print(time()-t)
    
    return data

print(test_load_henke_data(1).keys())