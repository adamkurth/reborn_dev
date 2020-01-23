import numpy as np
from bornagain import simulate as sim
from scipy import constants as const

eV = const.value('electron volt')
hc = const.h*const.c


def test_element_numbers_and_symbols():
        
    """Round trip conversion of atomic numbers to element symbols"""
        
    Z1 = np.arange(1, 119)
    sym = sim.atoms.atomic_numbers_to_symbols(Z1)
    Z2 = sim.atoms.atomic_symbols_to_numbers(sym)
    
    assert(all(Z1 == Z2))
    
    Z3 = 5
    sym = sim.atoms.atomic_numbers_to_symbols(Z3)
    Z4 = sim.atoms.atomic_symbols_to_numbers(sym)

    assert(Z3 == Z4)


def test_get_henke_data():
        
    """Check that we can read at least one dataset (carbon)"""

    data = sim.atoms._get_henke_data(6)
    assert(data['Element Symbol'] == 'C')
    Z = 29
    f_correct = 16.5705 + 1j * 2.98532
    f_lookup = sim.atoms.get_scattering_factors([Z], 798.570 * eV)[0]
    assert np.abs(f_lookup - f_correct) == 0
    f_lookup = sim.atoms.get_scattering_factors_fixed_z(Z, np.array([798.570 * eV]))[0]
    assert np.abs(f_lookup - f_correct) == 0


if __name__ == '__main__':
    test_element_numbers_and_symbols()
    test_get_henke_data()