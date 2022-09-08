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

import pickle
import numpy as np
from .. import target


def load_pickle(pickle_file):
    r""" Open a pickle file.
    Arguments:
        pickle_file (str): Path to the pickle file
    Returns: data
    """
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pickle_file):
    r""" Save something in pickle format.
    Arguments:
        data (anything, usually a dict): The data you want to pickle
        pickle_file (str): Path to the pickle file
    """
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)


def load_xyz(fname):
    r""" Load an "xyz" file.

    Arguments:
        fname (str): Path to the xyz file.

    Returns: dict with keys "position_vecs" and "atomic_numbers"
    """
    r = np.genfromtxt('files/C60.xyz', skip_header=2)[:, 1:] * 1e-10
    z = target.atoms.atomic_symbols_to_numbers(np.genfromtxt('files/C60.xyz', skip_header=2, dtype=str)[:, :1].ravel())
    return {'atomic_numbers': z, 'position_vecs': r}
