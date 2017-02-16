from bornagain import simulate as sim
import numpy as np


def test_element_numbers_and_symbols():
        
    """Round trip conversion of atomic numbers to element symbols"""
        
    Z1 = np.arange(1,119)
    sym = sim.atoms.atomic_numbers_to_symbols(Z1)
    Z2 = sim.atoms.atomic_symbols_to_numbers(sym)
    
    assert(all(Z1 == Z2))
    
    