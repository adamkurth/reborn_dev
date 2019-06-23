# coding=utf-8
r'''
Utilities for manipulating molecules
'''

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from bornagain.utils import max_pair_distance
from bornagain.simulate import atoms
from scipy import constants as const

hc = const.h*const.c

class Molecule(object):

    coordinates = None
    atomic_numbers = None
    n_atoms = None
    occupancies = None
    _max_atomic_pair_distance = None

    def __init__(self, coordinates=None, atomic_numbers=None, atomic_symbols=None):

        self.coordinates = coordinates
        if atomic_numbers is not None:
            self.atomic_numbers = atomic_numbers
            self.atomic_symbols = atoms.atomic_numbers_to_symbols(atomic_numbers)
        elif atomic_symbols is not None:
            self.atomic_symbols = atomic_symbols
            self.atomic_numbers = atoms.atomic_symbols_to_numbers(atomic_symbols)
        else:
            self.atomic_numbers = np.ones(len(coordinates.shape[0]))
            self.atomic_symbols = atoms.atomic_numbers_to_symbols(atomic_numbers)
        self.n_atoms = len(self.atomic_symbols)
        self.occupancies = np.ones(self.n_atoms)

    def get_scattering_factors(self, wavelength=None, beam=None):

        if beam is not None:
            wavelength = beam.wavelength
        return atoms.get_scattering_factors(self.atomic_numbers, hc / wavelength)

    @property
    def max_atomic_pair_distance(self):
        if self._max_atomic_pair_distance is None:
            self._max_atomic_pair_distance = max_pair_distance(self.coordinates)
        return self._max_atomic_pair_distance