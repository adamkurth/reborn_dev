# coding=utf-8
'''
Basic utilities for dealing with crystalline objects.

from Derek: Is using meters really the best here, given the PDB standard is Angstrom
and we will likey always deal with Angstrom scale coordinates?

for example print( "Lattice dim is %.3f"%(0.0000008)) will print 0.000, which can
cause problems...
'''

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from bornagain import units
from bornagain.simulate import atoms


class Molecule(object):

    coordinates = None
    atomic_numbers = None
    n_atoms = None
    occupancies = None

    def __init__(self, coordinates, atomic_numbers=None, atomic_symbols=None):

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
        return atoms.get_scattering_factors(self.atomic_numbers, units.hc / wavelength)
