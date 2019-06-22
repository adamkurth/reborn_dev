# coding=utf-8
'''
Basic utilities for dealing with crystalline objects.
'''

from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import urllib
import pkg_resources
import numpy as np
from bornagain.target.molecule import Molecule

pdb_data_path = pkg_resources.resource_filename('bornagain.simulate', os.path.join('data', 'pdb'))

def get_pdb_file(pdb_id, save_path='.'):
    r"""
    Fetch a pdb file from the web and return the path to the file.
    The default location for the file is the current working directory.
    If the file already exists, just return the path to the existing file.

    Args:
        pdb_id: for example: "101M" or "101M.pdb"

    Returns:
        string path to file
    """
    if save_path is None:
        save_path = pdb_data_path
    if not pdb_id.endswith('.pdb'):
        pdb_id += '.pdb'
    pdb_path = os.path.join(save_path, pdb_id)
    if not os.path.isfile(pdb_id):
        try:
            urllib.request.urlretrieve('https://files.rcsb.org/download/' + pdb_id, pdb_path)
        except urllib.error.HTTPError:
            return None
    return pdb_path


class UnitCell(object):

    r"""
    Simple class for unit cell information.  Provides the convenience methods r2x, x2r, q2h and h2q for transforming
    between fractional and orthogonal coordinates in real space and reciprocal space.
    """

    a = None  #: Lattice constant
    b = None  #: Lattice constant
    c = None  #: Lattice constant
    alpha = None  #: Lattice angle
    beta = None  #: Lattice angle
    gamma = None  #: Lattice angle
    volume = None  #: Unit cell volume
    o_mat = None  #: Orthogonalization matrix (3x3 array).  Does the transform r = O.x on fractional coordinates x.
    o_mat_inv = None  #: Inverse orthogonalization matrix (3x3 array)
    a_mat = None  #: Orthogonalization matrix transpose (3x3 array). Does the transform q = A.h, with Miller indices h.
    a_mat_inv = None  #: A inverse

    def __init__(self, a, b, c, alpha, beta, gamma):
        r"""

        Always initialize with the lattice parameters.  Units are SI and radians.

        Args:
            a: Lattice constant
            b: Lattice constant
            c: Lattice constant
            alpha: Lattice angle
            beta:  Lattice angle
            gamma: Lattice angle
        """

        al = alpha
        be = beta
        ga = gamma

        self.a = a
        self.b = b
        self.c = c
        self.alpha = al
        self.beta = be
        self.gamma = ga

        vol = a * b * c * np.sqrt(1 - np.cos(al)**2 - np.cos(be) **
                             2 - np.cos(ga)**2 + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
        o_mat = np.array([
                [a, b * np.cos(ga), c * np.cos(be)],
                [0, b * np.sin(ga), c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)],
                [0, 0, vol / (a * b * np.sin(ga))]
                ])
        o_inv = np.array([
                [1 / a, -np.cos(ga) / (a * np.sin(ga)), 0],
                [0, 1 / (b * np.sin(ga)), 0],
                [0, 0, a * b * np.sin(ga) / vol]
                ])
        self.o_mat = o_mat
        self.o_mat_inv = o_inv
        self.a_mat = o_inv.T.copy()
        self.a_mat_inv = o_mat.T.copy()
        self.volume = vol

    def r2x(self, r_vecs):
        r""" Transform orthogonal coordinates to fractional coordinates. """
        return np.dot(r_vecs, self.o_mat_inv.T)

    def x2r(self, x_vecs):
        r""" Transform fractional coordinates to orthogonal coordinates. """
        return np.dot(x_vecs, self.o_mat.T)

    def q2h(self, q_vecs):
        r""" Transform reciprocal coordinates to fractional Miller coordinates. """
        return np.dot(q_vecs, self.a_mat_inv.T)

    def h2q(self, h_vecs):
        r""" Transform fractional Miller coordinates to reciprocal coordinates. """
        return np.dot(h_vecs, self.a_mat.T)

    @property
    def a_vec(self):
        r""" Crystal basis vector a. """
        return self.o_mat[:, 0].copy()

    @property
    def b_vec(self):
        r""" Crystal basis vector b. """
        return self.o_mat[:, 1].copy()

    @property
    def c_vec(self):
        r""" Crystal basis vector c. """
        return self.o_mat[:, 2].copy()

    @property
    def as_vec(self):
        r""" Reciprocal basis vector a*. """
        return self.a_mat[:, 0].copy()

    @property
    def bs_vec(self):
        r""" Reciprocal basis vector b*. """
        return self.a_mat[:, 1].copy()

    @property
    def cs_vec(self):
        r""" Reciprocal basis vector c*. """
        return self.a_mat[:, 2].copy()

    def __str__(self):

        return 'a=%g b=%g c=%g, alpha=%g beta=%g gamma=%g' % (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)


class SpaceGroup(object):
    r"""
    Container for crystallographic spacegroup information.  Most importantly, transformation matices and vectors.  These
    transformations are in the fractional coordinate basis; this class has no awareness of a unit cell and hence
    cannot work with real-space orthogonal coordinates.
    """

    spacegroup_symbol = None  #: Spacegroup symbol (free format...)
    sym_rotations = None      #: 3x3 transformation matrices (fractional coordinates)
    sym_translations = None   #: Symmetry translations (fractional coordinates)

    def __init__(self, spacegroup_symbol, sym_rotations, sym_translations):
        r"""
        Initialization requires that you determine the rotations and translations.
        """
        self.spacegroup_symbol = spacegroup_symbol
        self.sym_rotations = sym_rotations
        self.sym_translations = sym_translations

    @property
    def n_molecules(self):
        return len(self.sym_rotations)

    def apply_symmetry_operation(self, op_num, x_vecs):
        r"""
        Apply a symmetry operation to an asymmetric unit.

        Args:
            op_num (int): The number of the operation to be applied.
            x_vecs (Nx3 array): The atomic coordinates in the crystal basis.

        Returns: Nx3 array.
        """
        rot = self.sym_rotations[op_num]
        trans = self.sym_translations[op_num]
        return np.dot(rot, x_vecs.T).T + trans


class CrystalStructure(object):
    r"""
    A container class for stuff needed when dealing with crystal structures.  Combines information regarding the
    asymmetric unit, unit cell, and spacegroup.
    """
    # TODO: Needs documentation!

    fractional_coordinates = None  # : Fractional coordinates of the asymmetric unit (expanded w/ non-cryst. symmetry)
    molecule = None  # : Molecule class instance containing the asymmetric unit
    unitcell = None  # : UnitCell class instance
    spacegroup = None  # : Spacegroup class instance
    mosaicity_fwhm = 0
    crystal_size = 1e-6
    crystal_size_fwhm = 0.0
    mosaic_domain_size = 0.5e-6
    mosaic_domain_size_fwhm = 0.0
    cryst1 = ""
    pdb_dict = None

    def __init__(self, pdb_file_path):

        dic = pdb_to_dict(pdb_file_path)
        self.pdb_dict = dic

        a, b, c, al, be, ga = dic['unit_cell']
        self.unitcell = UnitCell(a*1e-10, b*1e-10, c*1e-10, al*np.pi/180, be*np.pi/180, ga*np.pi/180)
        S = dic['scale_matrix']
        U = dic['scale_translation']
        # print(self.unitcell.o_mat_inv - S)

        # These are the initial coordinates with strange origin
        r = dic['atomic_coordinates']
        # Check for non-crystallographic symmetry.  Construct asymmetric unit from them.
        ncs_partners = [r]
        n_ncs_partners = len(dic['ncs_rotations'])
        if n_ncs_partners > 0:
            for i in range(n_ncs_partners):
                R = dic['ncs_rotations'][i]
                T = dic['ncs_translations'][i]
                ncs_partners.append(np.dot(r, R.T) + T)
        r_au = np.vstack(ncs_partners)
        # Transform to fractional coordinates
        x_au = np.dot(S, r_au.T).T + U
        self.fractional_coordinates = x_au

        n_sym = len(dic['spacegroup_rotations'])
        rotations = []
        translations = []
        for i in range(n_sym):
            R = dic['spacegroup_rotations'][i]
            T = dic['spacegroup_translations'][i]
            W = np.round(np.dot(S, np.dot(R, np.linalg.inv(S))))
            Z = np.dot(S, T) + np.dot(np.eye(3)-W, U)
            w = Z != 0
            Z[w] = 1/np.round(1/Z[w])
            rotations.append(W)
            translations.append(Z)
        self.spacegroup = SpaceGroup(dic['spacegroup_symbol'], rotations, translations)

        r_au_mod = np.dot(np.linalg.inv(S), x_au.T).T
        self.molecule = Molecule(coordinates=r_au_mod, atomic_symbols=dic['atomic_symbols'])

    @property
    def x_vecs(self):

        r"""

        Fractional coordinates of atoms.

        Returns: Nx3 numpy array

        """

        if self._x is None:
            self._x = np.dot(self.unitcell.o_mat_inv, self.molecule.coordinates.T).T
        return self._x

    @property
    def x(self):
        return self.x_vecs

    def get_symmetry_expanded_coordinates(self):
        r"""

        Returns: All atomic coordinates including spacegroup symmetry partners.

        """
        x0 = self.x
        xs = []
        for (R, T) in zip(self.spacegroup.sym_rotations, self.spacegroup.sym_translations):
            xs.append(np.dot(x0, R.T) + T)
        return np.dot(np.concatenate(xs), self.unitcell.o_mat.T)


class FiniteLattice(object):
    r"""
    A utility for creating finite crystal lattices.  Enables the generation of lattice vector positions, lattice
    occupancies, shaping of crystals. Under development.
    """

    def __init__(self, max_size=None, unitcell=None):
        r"""
        Arguments:

            max_size: Integer N that sets the size of the lattice to N x N x N.
            unitcell: A crystal.UnitCell type that is needed to generate
        """

        if max_size is None:
            raise ValueError("You need to choose a maximum lattice size.")

        if not isinstance(unitcell, UnitCell):
            raise ValueError("You must provide a unitcell of crystal.UnitCell type.")

        if not isinstance(max_size, int):
            raise ValueError("max_size must be an int.")

        if max_size <= 0:
            raise ValueError("max_size must be >= 0.")

        self.max_size = max_size
        self.occupancies = np.ones([self.max_size]*3)
        self.unitcell = unitcell
        ran = np.arange(-(self.max_size-1)/2.0, (self.max_size+1)/2.0)
        x, y, z = np.meshgrid(ran, ran, ran, indexing='ij')
        self.all_x_coordinates = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T.copy()
        self.all_x_coordinates.flags.writeable = False
        self._all_r_coordinates = None

    @property
    def all_r_coordinates(self):

        if self._all_r_coordinates is None:
            self._all_r_coordinates = np.dot(self.all_x_coordinates, self.unitcell.o_mat.T)
            self._all_r_coordinates.flags.writeable = False

        return self._all_r_coordinates

    @property
    def occupied_indices(self):

        return np.where(self.occupancies.ravel() != 0)[0]

    @property
    def occupied_x_coordinates(self):

        return self.all_x_coordinates[self.occupied_indices, :]

    @property
    def occupied_r_coordinates(self):

        return np.dot(self.occupied_x_coordinates, self.unitcell.o_mat.T)

    def add_facet(self, plane=None, length=None, shift=0):

        proj = (self.all_x_coordinates+shift).dot(np.array(plane))
        w = np.where(proj > length)[0]
        if len(w) > 0:
            self.occupancies.flat[w] = 0

    def reset_occupancies(self):

        self.occupancies = np.ones([self.max_size]*3)

    def make_hexagonal_prism(self, n_cells=None):

        self.reset_occupancies()
        self.add_facet(plane=[-1, 1, 0], length=n_cells)
        self.add_facet(plane=[1, -1, 0], length=n_cells)
        self.add_facet(plane=[1, 0, 0], length=n_cells)
        self.add_facet(plane=[0, 1, 0], length=n_cells)
        self.add_facet(plane=[-1, 0, 0], length=n_cells)
        self.add_facet(plane=[0, -1, 0], length=n_cells)


def pdb_to_dict(pdb_file_path):
    r"""Return a :class:`CrystalStructure` object with a subset of PDB information.  If there are multiple atomic
    models, only the first will be extracted.  Units are the standard PDB units: angstrom and degrees.

    Arguments:
        pdb_file_path: path to pdb file

    Returns:
        A dictionary with the following keys:

           'scale_matrix'
           'scale_translation'
           'atomic_coordinates'
           'atomic_symbols'
           'unit_cell'
           'spacegroup_symbol'
           'spacegroup_rotations'
           'spacegroup_translations'
           'ncs_rotations'
           'ncs_translations'

    """

    atomic_coordinates = np.zeros([10000, 3])
    atomic_symbols = []
    scale = np.zeros([3, 4])
    smtry = np.zeros([1000, 6])
    mtrix = np.zeros([10000, 4])

    smtry_index = 0
    mtrix_index = 0
    atom_index = 0

    with open(pdb_file_path) as pdbfile:

        for line in pdbfile:

            # Check for a model ID; we will only load in the first model for now
            if line[0:5] == 'MODEL':
                model_number = int(line[10:14])
                if model_number > 1:
                    break

            # Transformations from orthogonal coordinates to fractional coordinates
            if line[:5] == 'SCALE':
                n = int(line[5]) - 1
                scale[n, 0] = float(line[10:20])
                scale[n, 1] = float(line[20:30])
                scale[n, 2] = float(line[30:40])
                scale[n, 3] = float(line[45:55])

            # The crystal lattice and spacegroup
            if line[:6] == 'CRYST1':
                cryst1 = line
                a = float(cryst1[6:15])
                b = float(cryst1[15:24])
                c = float(cryst1[24:33])
                alpha = float(cryst1[33:40])
                beta = float(cryst1[40:47])
                gamma = float(cryst1[47:54])
                spacegroup_symbol = cryst1[55:66].strip()

            # Spacegroup symmetry operations
            if line[13:18] == 'SMTRY':
                smtry[smtry_index, 0] = float(line[18])
                smtry[smtry_index, 1] = float(line[19:23])
                smtry[smtry_index, 2] = float(line[24:34])
                smtry[smtry_index, 3] = float(line[34:44])
                smtry[smtry_index, 4] = float(line[44:54])
                smtry[smtry_index, 5] = float(line[54:69])
                smtry_index += 1

            # Non-crystallographic symmetry operations
            # TODO: check if we should skip entries with iGiven == 1.  See PDB documentation.  I'm confused.
            if line[0:5] == 'MTRIX':
                mtrix[mtrix_index, 0] = float(line[10:20])
                mtrix[mtrix_index, 1] = float(line[20:30])
                mtrix[mtrix_index, 2] = float(line[30:40])
                mtrix[mtrix_index, 3] = float(line[45:55])
                igiven = int(line[59])
                mtrix_index += 1

            # Atomic (orthogonal) coordinates
            if line[:6] == 'ATOM  ' or line[:6] == "HETATM":
                atomic_coordinates[atom_index, 0] = float(line[30:38])
                atomic_coordinates[atom_index, 1] = float(line[38:46])
                atomic_coordinates[atom_index, 2] = float(line[46:54])
                atomic_symbols.append(line[76:78].strip().capitalize())
                atom_index += 1
            if atom_index == atomic_coordinates.shape[0]:  # Make the array larger if need be.
                atomic_coordinates = np.vstack([atomic_coordinates, np.zeros((atomic_coordinates.shape[0], 3))])

    spacegroup_rotations = []
    spacegroup_translations = []
    for i in range(int(smtry_index/3)):
        spacegroup_rotations.append(smtry[i*3:i*3+3, 2:5])
        spacegroup_translations.append(smtry[i * 3:i * 3 + 3, 5])

    ncs_rotations = []
    ncs_translations = []
    for i in range(int(mtrix_index/3)):
        ncs_rotations.append(mtrix[i*3:i*3+3, 0:3])
        ncs_translations.append(mtrix[i*3:i*3+3, 3])

    dic = {'scale_matrix': scale[:, 0:3],
           'scale_translation': scale[:, 3],
           'atomic_coordinates': atomic_coordinates[:atom_index, :],
           'atomic_symbols': atomic_symbols[:atom_index],
           'unit_cell': (a, b, c, alpha, beta, gamma),
           'spacegroup_symbol': spacegroup_symbol,
           'spacegroup_rotations': spacegroup_rotations,
           'spacegroup_translations': spacegroup_translations,
           'ncs_rotations': ncs_rotations,
           'ncs_translations': ncs_translations
           }

    return dic