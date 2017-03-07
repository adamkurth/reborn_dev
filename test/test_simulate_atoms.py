import sys

import numpy as np

sys.path.append('..')
from bornagain import simulate as sim
from bornagain import units

def test_element_numbers_and_symbols():
        
    """Round trip conversion of atomic numbers to element symbols"""
        
    Z1 = np.arange(1,119)
    sym = sim.atoms.atomic_numbers_to_symbols(Z1)
    Z2 = sim.atoms.atomic_symbols_to_numbers(sym)
    
    assert(all(Z1 == Z2))
    
    Z3 = 5
    sym = sim.atoms.atomic_numbers_to_symbols(Z3)
    Z4 = sim.atoms.atomic_symbols_to_numbers(sym)
    
    assert(all(Z3 == Z4))
    
def test_get_henke_data():
        
    """Check that we can read at least one dataset (carbon)"""
        
    data = sim.atoms.get_henke_data(6)
    
    assert(data['Element Symbol'] == 'C')
    
def test_get_scattering_factors():
    
    """Check that the first ten elements have sensible scattering factors"""
    
    Z = [1,2,3,4,1,2,3,4,5,6,7,8,9,10]
    
    f = sim.atoms.get_scattering_factors(Z,units.hc/1.5e-10)
    
    assert(np.abs(np.real(f) - np.array(Z)).max() < 0.1)
    assert(all(np.imag(f) > 0))
    
    