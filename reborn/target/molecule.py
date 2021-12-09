# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

# coding=utf-8
r"""
Utilities for manipulating molecules.
"""
import numpy as np
from ..utils import max_pair_distance
from . import atoms
from .. import const

hc = const.hc
avogadros_number = const.avogadros_number


class Molecule(object):
    r"""
    Simple container for organizing atomic numbers, coordinates, etc.  There is presently no information about
    bonds, as the name of the class might suggest...
    """
    _coordinates = None
    atomic_numbers = None
    n_atoms = None
    occupancies = None
    _max_atomic_pair_distance = None

    def __init__(self, coordinates=None, atomic_numbers=None, atomic_symbols=None):
        r"""
        Args:
            coordinates (numpy array): An array of shape Nx3 vectors in cartesiaon coordinates
            atomic_numbers (numpy array): Array of integer atomic numbers
            atomic_symbols (numpy array): Array of atomic symbols (as an alternative to providing atomic numbers)
        """
        self.coordinates = coordinates
        if atomic_numbers is not None:
            self.atomic_numbers = atomic_numbers
            self.atomic_symbols = atoms.atomic_numbers_to_symbols(atomic_numbers)
        elif atomic_symbols is not None:
            self.atomic_symbols = atomic_symbols
            self.atomic_numbers = atoms.atomic_symbols_to_numbers(atomic_symbols)
        else:
            self.atomic_numbers = np.ones(len(self._coordinates.shape[0]))
            self.atomic_symbols = atoms.atomic_numbers_to_symbols(atomic_numbers)
        self.n_atoms = len(self.atomic_symbols)
        self.occupancies = np.ones(self.n_atoms)

    @property
    def coordinates(self):
        return self._coordinates.copy()

    @coordinates.setter
    def coordinates(self, value):
        self._max_atomic_pair_distance = None
        self._coordinates = np.array(value)

    def get_scattering_factors(self, photon_energy=None, beam=None):
        r"""
        Get atomic scattering factors.  You need to specify the photon energy or pass a Beam class instance in.

        This wraps the function :func:`reborn.target.atoms.get_scattering_factors` for more details; see the docs
        for more details.

        Args:
            photon_energy: In SI units as always
            beam: Optionally, provide a :class:`reborn.beam.Beam` instance instead of photon_energy

        Returns:
            Numpy array of complex scattering factors
        """
        if beam is not None:
            photon_energy = beam.photon_energy

        return atoms.get_scattering_factors(self.atomic_numbers, photon_energy)

    @property
    def max_atomic_pair_distance(self):
        r"""
        Return the maximum distance between two atoms in the molecule.  This was intended to be useful for determining an
        appropriate angular or q-space sampling frequency.
        """
        if self._max_atomic_pair_distance is None:
            self._max_atomic_pair_distance = max_pair_distance(self.coordinates)
        return self._max_atomic_pair_distance

    def get_molecular_weight(self):
        r"""
        Returns the molecular weight in SI units (kg).
        """
        return np.sum(atoms.atomic_weights[self.atomic_numbers])

    def get_centered_coordinates(self):
        r"""
        Get the coordinates with center of mass set to the origin.

        Returns:
            |ndarray|: An Nx3 array of coordinates.
        """
        return self.coordinates - np.sum((self.atomic_numbers*self.coordinates.T).T, axis=0)/np.sum(self.atomic_numbers)

    def get_atomic_weights(self):
        r"""
        Returns the atomic weights in SI units (kg).
        """
        import xraylib
        return np.array([xraylib.AtomicWeight(z) for z in self.atomic_numbers])*1e-3/avogadros_number
