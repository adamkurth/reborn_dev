import numpy as np
from reborn.target import atoms
from scipy import constants as const

eV = const.value('electron volt')
hc = const.h*const.c


def test_element_numbers_and_symbols():
    # Round trip conversion of atomic numbers to element symbols
    z1 = np.arange(1, 119)
    sym = atoms.atomic_numbers_to_symbols(z1)
    z2 = atoms.atomic_symbols_to_numbers(sym)
    assert(all(z1 == z2))
    z3 = 5
    sym = atoms.atomic_numbers_to_symbols(z3)
    z4 = atoms.atomic_symbols_to_numbers(sym)
    assert(z3 == z4)


def test_henke_data():
    # Check that we are doing the lookup of Henke tables correctly
    data = atoms._get_henke_data(6)
    assert(data['Element Symbol'] == 'C')
    z = 29
    f_correct = 16.5705 + 1j * 2.98532
    f_lookup = atoms.get_scattering_factors([z], 798.570 * eV)[0]
    assert(np.abs(f_lookup - f_correct) == 0)
    f_lookup = atoms.get_scattering_factors_fixed_z(z, np.array([798.570 * eV]))[0]
    assert(np.abs(f_lookup - f_correct) == 0)
    f_lookup = atoms.henke_scattering_factors(z, 798.570 * eV)
    assert(np.abs(f_lookup - f_correct) == 0)
    # Dispersion corrections simply subtract the atomic number
    Z = np.array([2, 5])
    E = np.array([200, 2000, 3000])*1.6022e-19
    f = atoms.henke_dispersion_corrections(Z, E)
    fp = atoms.henke_dispersion_corrections(Z, E)
    assert(isinstance(f, np.ndarray))
    assert(np.max(np.round(np.abs(f - fp)) - np.abs(f - fp)) == 0)


def test_hubbel_form_factors():
    # Check one of the values from the Hubbel et al. 1975 paper:
    assert(np.abs(24.461 - atoms.hubbel_form_factors(0.175*4*np.pi*1e10, 29)) == 0)
