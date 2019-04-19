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
from bornagain.utils import max_pair_distance
from bornagain.simulate import atoms


class Molecule(object):

    coordinates = None
    atomic_numbers = None
    n_atoms = None
    occupancies = None

    def __init__(self, coordinates=None, atomic_numbers=None, atomic_symbols=None, pdb_file=None):

        if pdb_file is not None:
            self.load_pdb(pdb_file)
        else:
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

    def load_pdb(self, pdb_file):

        max_atoms = int(1e4)
        coordinates = np.zeros([max_atoms, 3])
        atomic_symbols = []
        atom_index = int(0)

        with open(pdb_file) as pdbfile:

            for line in pdbfile:

                if line[:6] == 'ATOM  ' or line[:6] == "HETATM":
                    coordinates[atom_index, 0] = float(line[30:38]) * 1e-10
                    coordinates[atom_index, 1] = float(line[38:46]) * 1e-10
                    coordinates[atom_index, 2] = float(line[46:54]) * 1e-10
                    atomic_symbols.append(line[76:78].strip().capitalize())
                    atom_index += 1

                if atom_index == max_atoms:
                    coordinates = np.append(coordinates, np.zeros([3, max_atoms]), axis=1)

        self.coordinates = coordinates[:atom_index, :]
        self.atomic_symbols = atomic_symbols[:atom_index]
        self.atomic_numbers = atoms.atomic_symbols_to_numbers(atomic_symbols)

    @property
    def max_atomic_pair_distance(self):

        return max_pair_distance(self.coordinates)