# coding=utf-8
'''
Basic utilities for dealing with crystalline objects.

from Derek: Is using meters really the best here, given the PDB standard is Angstrom
and we will likey always deal with Angstrom scale coordinates?

for example print( "Lattice dim is %.3f"%(0.0000008)) will print 0.000, which can
cause problems...
'''

from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import urllib
# Python 2 and 3 compatibility...
try:
    basestring
except NameError:
    basestring = str

import pkg_resources
import numpy as np
from numpy import sin, cos, sqrt
import bornagain.target
from bornagain import utils
from bornagain.utils import rotate
from bornagain.simulate import simutils

try:
    import spglib
except ImportError:
    spglib = None

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

        Set the unit cell lattice.

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

        vol = a * b * c * sqrt(1 - cos(al)**2 - cos(be) **
                             2 - cos(ga)**2 + 2 * cos(al) * cos(be) * cos(ga))
        o_mat = np.array([
                [a, b * cos(ga), c * cos(be)],
                [0, b * sin(ga), c * (cos(al) - cos(be) * cos(ga)) / sin(ga)],
                [0, 0, vol / (a * b * sin(ga))]
                ])
        o_inv = np.array([
                [1 / a, -cos(ga) / (a * sin(ga)), 0],
                [0, 1 / (b * sin(ga)), 0],
                [0, 0, a * b * sin(ga) / vol]
                ])
        self.o_mat = o_mat
        self.o_mat_inv = o_inv
        self.a_mat = o_inv.T.copy()
        self.a_mat_inv = o_mat.T.copy()
        self.volume = vol

    def r2x(self, r_vecs):
        r""" Transform orthogonal coordinates to fractional coordinates. """
        return rotate(self.o_mat_inv, r_vecs)

    def x2r(self, x_vecs):
        r""" Transform fractional coordinates to orthogonal coordinates. """
        return rotate(self.o_mat, x_vecs)

    def q2h(self, q_vecs):
        r""" Transform reciprocal coordinates to fractional Miller coordinates. """
        return rotate(self.a_mat_inv, q_vecs)

    def h2q(self, h_vecs):
        r""" Transform fractional Miller coordinates to reciprocal coordinates. """
        return rotate(self.a_mat, h_vecs)

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
    Container for crystallographic spacegroup information.  Note that the 230 ITOC numbers that
    specify space groups refer only to the actual symmetry properties.  There are multiple Hall numbers and Hermann
    Mauguin symbols that correspond to the same spacegroup.  The duplicates are just different ways of specifying the
    same spacegroup -- e.g. some may apply rotations around different axes than others.
    """

    hall_number = None  #: Space group Hall number
    itoc_number = None  #: Space group number in the International Tables of Crystallography (ITOC)
    hermann_mauguin_symbol = None  #: Spacegroup Hermann Mauguin symbol (e.g. as it appears in a PDB file for example)
    sym_rotations = None  #: Symmetry 3x3 transformation matrices (in crystal basis)
    sym_translations = None  #: Symmetry translations (in crystal basis)
    n_molecules = None  #: Number of symmetry-related molecules

    def __init__(self, hermann_mauguin_symbol=None, hall_number=None, itoc_number=None):
        r"""
        Initialize the spacegroup.  You must identify the spacegroup by either a Hall number, a
        Hermann Mauguin symbol, or an ITOC number (see International Tables of Crystallography).

        Args:
            hermann_mauguin_symbol (string): Hermann Mauguin symbol (for example: P63)
            hall_number (int): One of the 1-530 Hall numbers.
            itoc_number (int): One of the 230 ITOC numbers.
        """

        if hall_number is not None:
            self.hall_number = hall_number
            self.itoc_number = itoc_number_from_hall_number(hall_number)
            self.hermann_mauguin_symbol = hermann_mauguin_symbol_from_hall_number(hall_number)
        elif hermann_mauguin_symbol is not None:
            self.hermann_mauguin_symbol = hermann_mauguin_symbol
            self.itoc_number = itoc_number_from_hermann_mauguin_symbol(hermann_mauguin_symbol)
            self.hall_number = hall_number_from_hermann_mauguin_symbol(hermann_mauguin_symbol)
        elif itoc_number is not None:
            self.itoc_number = itoc_number
            self.hall_number = hall_number_from_itoc_number(itoc_number)
            self.hermann_mauguin_symbol = hermann_mauguin_symbol_from_hall_number(self.hall_number)
        else:
            raise ValueError('You must initialize SpaceGroup with a spacegroup identifier')

        self.sym_rotations, self.sym_translations = get_symmetry_operators_from_hall_number(self.hall_number)
        self.n_molecules = len(self.sym_rotations)

    def apply_symmetry_operation(self, op_num=None, x_vecs=None, inverse=False):
        r"""
        Apply a symmetry operation to an asymmetric unit.

        Args:
            op_num (int): The number of the operation to be applied.
            x_vecs (Nx3 array): The atomic coordinates in the crystal basis.
            inverse (bool): If true, do the inverse operation.  Default is False.

        Returns: Nx3 array.
        """
        rot = self.sym_rotations[op_num]
        trans = self.sym_translations[op_num]
        return utils.rotate(rot, x_vecs) + trans


class CrystalStructure(object):
    r"""
    A container class for stuff needed when dealing with crystal structures.
    """
    # TODO: Needs documentation!

    _x = None  #: Fractional coordinates (3xN array)
    T = None  #: Translation vector that goes with orthogonalization matrix (What is this used for???)

    molecule = None
    unitcell = None
    spacegroup = None
    mosaicity_fwhm = 0
    crystal_size = 1e-6
    crystal_size_fwhm = 0.0
    mosaic_domain_size = 0.5e-6
    mosaic_domain_size_fwhm = 0.0
    cryst1 = ""

    def __init__(self, pdbFilePath=None, spacegroup=None):

        if pdbFilePath is not None:
            dic = pdb_to_dict(pdbFilePath)
            self.set_molecule(dic['orthogonal_coordinates'], atomic_symbols=dic['atomic_symbols'])
            self.set_cell(*dic['cell'])
            if spacegroup is None:
                self.set_spacegroup(hermann_mauguin_symbol=dic['spacegroup_symbol'])
        if spacegroup is not None:
            self.spacegroup = spacegroup

    def load_pdb(self, pdb_file_path):

        r"""

        Populate the class with all the info from a PDB file.

        Args:
            pdb_file_path: Path to the PDB file

        """
        parse_pdb(pdb_file_path, self)

    def set_cell(self, *args, **kwargs):
        r"""

        Set the unit cell lattice.  Takes the same arguments as ``

        """

        self.unitcell = UnitCell(*args, **kwargs)

    def set_spacegroup(self, *args, **kwargs):

        r"""

        Set the spacegroup of the crystal.  This produces a cache of the symmetry transformation operations.

        Args:
            hermann_mauguin_symbol:  A string like 'P 63'
            hall_number: Hall number (between 1-530)
            itoc_number: Spacegroup number from International Tables of Crystallography (between 1-230)

        """

        self.spacegroup = SpaceGroup(*args, **kwargs)

    def set_molecule(self, *args, **kwargs):

        r"""
        See docs for target.Molecule
        """

        self.molecule = bornagain.target.molecule.Molecule(*args, **kwargs)

    @property
    def elements(self):
        utils.depreciate('Use CrystalStructure.molecule')
        return self.molecule.atomic_symbols

    @property
    def Z(self):
        utils.depreciate('Use CrystalStructure.molecule')
        return self.molecule.atomic_numbers

    @property
    def nAtoms(self):
        utils.depreciate('Use CrystalStructure.molecule')
        return self.molecule.n_atoms

    @property
    def r(self):
        utils.depreciate('Use CrystalStructure.molecule')
        return self.molecule.coordinates

    @property
    def a(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.a

    @property
    def b(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.b

    @property
    def c(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.c

    @property
    def alpha(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.alpha

    @property
    def beta(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.beta

    @property
    def gamma(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.gamma

    @property
    def V(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.volume

    @property
    def O(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.o_mat

    @property
    def Oinv(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.o_mat_inv

    @property
    def A(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.a_mat

    @property
    def Ainv(self):
        utils.depreciate('Use CrystalStructure.unitcell')
        return self.unitcell.a_mat_inv

    @property
    def hermannMauguinSymbol(self):
        utils.depreciate('Use CrystalStructure.spacegroup')
        return self.spacegroup.hermann_mauguin_symbol

    @property
    def spaceGroupNumber(self):
        utils.depreciate('Use CrystalStructure.spacegroup')
        return self.spacegroup.itoc_number

    @property
    def nMolecules(self):
        utils.depreciate('Use CrystalStructure.spacegroup')
        return self.n_molecules

    @property
    def n_molecules(self):
        utils.depreciate('Use CrystalStructure.spacegroup')
        return self.spacegroup.n_molecules

    @property
    def symRs(self):
        utils.depreciate('Use CrystalStructure.spacegroup')
        return self.spacegroup.sym_rotations

    @property
    def symTs(self):
        utils.depreciate('Use CrystalStructure.spacegroup')
        return self.spacegroup.sym_translations

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

        x0 = self.x
        xs = []
        for (R, T) in zip(self.spacegroup.sym_rotations, self.spacegroup.sym_translations):
            xs.append(utils.rotate(R, x0) + T)
        return utils.rotate(self.unitcell.o_mat, np.concatenate(xs))


class Structure(CrystalStructure):

    def __init__(self, *args, **kwargs):

        utils.depreciate('The class "crystal.Structure" is depreciated.  Use "crystal.CrystalStructure" instead.')

        CrystalStructure.__init__(self, *args, **kwargs)


class structure(CrystalStructure):

    def __init__(self, *args, **kwargs):

        utils.depreciate('The class "crystal.structure" is depreciated.  Use "crystal.CrystalStructure" instead.')

        CrystalStructure.__init__(self, *args, **kwargs)


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
            self._all_r_coordinates = rotate(self.unitcell.o_mat, self.all_x_coordinates)
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

        return rotate(self.unitcell.o_mat, self.occupied_x_coordinates)

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
    r"""Return a :class:`CrystalStructure` object with PDB information.  The PDB information is converted to
    the units of bornagain.

    Arguments:
        pdb_file_path: path to pdb file

    Returns:
        A dictionary with the following keys:

           'orthogonalization_matrix': S,

           'fractional_coordinates': fractional_coordinates,

           'orthogonal_coordinates': coordinates*1e-10,

           'atomic_symbols': atomic_symbols,

           'cell': (a*1e-10, b*1e-10, c*1e-10, alpha*np.pi/180.0, beta*np.pi/180.0, gamma*np.pi/180.0),

           'spacegroup_symbol': spacegroup_symbol,

           'orthogonal_rotations': orthogonal_rotations,

           'orthogonal_translations': orthogonal_translations*1e-10,

           'fractional_rotations': orthogonal_rotations,

           'fractional_translations': orthogonal_translations * 1e-10

    """

    coordinates = np.zeros([10000, 3])
    atomic_symbols = []
    atom_index = 0
    scale = np.zeros([3, 4])
    smtry = np.zeros([1000, 6])
    n_smtry = 0
    mtrix = np.zeros([10000, 4])
    n_mtrix = 0
    model_number = 1

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
                smtry[n_smtry, 0] = float(line[18])
                smtry[n_smtry, 1] = float(line[19:23])
                smtry[n_smtry, 2] = float(line[24:34])
                smtry[n_smtry, 3] = float(line[34:44])
                smtry[n_smtry, 4] = float(line[44:54])
                smtry[n_smtry, 5] = float(line[54:69])
                n_smtry += 1

            # Non-crystallographic symmetry operations
            # TODO: check if we should skip entries with iGiven == 1.  See PDB documentation.  I'm confused.

            if line[0:5] == 'MTRIX':
                mtrix[n_mtrix, 0] = float(line[10:20])
                mtrix[n_mtrix, 1] = float(line[20:30])
                mtrix[n_mtrix, 2] = float(line[30:40])
                mtrix[n_mtrix, 3] = float(line[45:55])
                igiven = int(line[59])
                n_mtrix += 1

            # Atomic (orthogonal) coordinates

            if line[:6] == 'ATOM  ' or line[:6] == "HETATM":
                coordinates[atom_index, 0] = float(line[30:38])
                coordinates[atom_index, 1] = float(line[38:46])
                coordinates[atom_index, 2] = float(line[46:54])
                atomic_symbols.append(line[76:78].strip().capitalize())
                atom_index += 1
            if atom_index == coordinates.shape[0]:  # Make the array larger if need be.  Double it.
                coordinates = np.append(coordinates, np.zeros([3, coordinates.shape[0]]), axis=1)

    # Truncate atom list since we pre-allocated extra memory
    coordinates = coordinates[:atom_index, :]
    atomic_symbols = atomic_symbols[:atom_index]

    S = scale[:, 0:3]
    U = scale[:, 3]
    fractional_coordinates = rotate(S, coordinates) + U

    orthogonal_rotations = []
    orthogonal_translations = []
    fractional_rotations = []
    fractional_translations = []
    for i in range(int(n_smtry/3)):
        R = smtry[i*3:i*3+3, 2:5]
        T = smtry[i * 3:i * 3 + 3, 5]
        orthogonal_rotations.append(R)
        orthogonal_translations.append(T)
        fractional_rotations.append(np.round(np.dot(S, np.dot(R, np.linalg.inv(S)))))
        TT = np.dot(S, T)
        w = np.where(TT != 0)
        TT[w] = 1/np.round(1/TT[w])
        fractional_translations.append(TT)

    ncs_orthogonal_rotations = []
    ncs_orthogonal_translations = []
    ncs_fractional_rotations = []
    ncs_fractional_translations = []
    for i in range(int(n_mtrix/3)):
        R = mtrix[i*3:i*3+3, 0:3]
        T = mtrix[i*3:i*3+3, 3]
        ncs_orthogonal_rotations.append(R)
        ncs_orthogonal_translations.append(T)
        ncs_fractional_rotations.append(np.dot(S, np.dot(R, np.linalg.inv(S))))
        ncs_fractional_translations.append(np.dot(S, T))

    dic = {'orthogonalization_matrix': S*1e-10,
           'fractional_coordinates': fractional_coordinates,
           'orthogonal_coordinates': coordinates*1e-10,
           'atomic_symbols': atomic_symbols,
           'unit_cell': (a*1e-10, b*1e-10, c*1e-10, alpha*np.pi/180.0, beta*np.pi/180.0, gamma*np.pi/180.0),
           'spacegroup_symbol': spacegroup_symbol,
           'orthogonal_rotations': orthogonal_rotations,
           'orthogonal_translations': [t*1e-10 for t in orthogonal_translations],
           'fractional_rotations': fractional_rotations,
           'fractional_translations': fractional_translations,
           'ncs_orthogonal_rotations': orthogonal_rotations,
           'ncs_orthogonal_translations': [t * 1e-10 for t in orthogonal_translations],
           'ncs_fractional_rotations': fractional_rotations,
           'ncs_fractional_translations': fractional_translations
           }

    return dic


def parse_pdb(pdb_file_path, crystal_structure=None):
    r"""Return a :class:`CrystalStructure` object with PDB information. """

    if crystal_structure is None:
        cryst = CrystalStructure()

    dic = pdb_to_dict(pdb_file_path)
    cryst.set_molecule(dic['orthogonal_coordinates'], atomic_symbols=dic['atomic_symbols'])
    cryst.set_cell(*dic['cell'])
    cryst.set_spacegroup(hermann_mauguin_symbol=dic['spacegroup_symbol'])

    return cryst


def get_symmetry_operators_from_space_group(spacegroup_symbol):

    r"""
    For a given Hermann-Mauguin spacegroup, provide the symmetry operators in the form of
    translation and rotation operators.  These operators are in the crystal basis.
    As of now, this is a rather unreliable function.  My brain hurts badly after attempting
    to make sense of the ways in which space groups are specified... if you are having trouble
    finding your spacegroup check that it is contained in the output of print(crystal.get_hm_symbols()).

    Input:
        spacegroup_symbol (str): A string indicating the spacegroup in Hermann–Mauguin notation (as found in a PDB file)

    Output: Two lists: Rs and Ts.  These correspond to lists of rotation matrices (3x3 numpy
            arrays) and translation vectors (1x3 numpy arrays)
    """

    if spglib is None:
        raise ImportError('You need to install the "spglib" python package')
    sgsym = spacegroup_symbol
    sgsym = sgsym.strip()
    n_found = 0
    hall_number = 0
    for i in range(1, 531):
        # Spacegroup symbol madness.... is there some kind of standard for representing these in ASCII text?
        itoc_full = spglib.get_spacegroup_type(i)['international_full'].replace('_', '').strip()
        if sgsym == itoc_full:
            n_found += 1
            hall_number = i
    if n_found == 0:
        print('Did not find matching spacegroup')
        return None, None, None
    if n_found > 1:
        print('Found multiple spacegroup matches')
        # return None, None, None
    ops = spglib.get_symmetry_from_database(hall_number)
    n_ops = ops['rotations'].shape[0]
    rotations = []
    translations = []
    for i in range(n_ops):
        rotations.append(ops['rotations'][i, :, :])
        translations.append(ops['translations'][i, :])
    return rotations, translations, hall_number

# def assemble(O, n_unit=10, spherical=False):
#     r"""
#     From Derek: assemble and assemble3 are identical functions, hence the try/except at the start
#         n_unit can be a tuple or an integer
#
#     Creates a finite lattice
#     Args:
#         O, Structure attribute O (3x3 ndarray), orientation matrix of crystal
#             (columns are the lattice vectors a,b,c)
#         n_unit (int or 3-tuple): Number of unit cells along each crystal axis
#         spherical (bool): If true, apply a spherical boundary to the crystal.
#     """
#
#     #lattice coordinates
#     try:
#         n_unit_a, n_unit_b, n_unit_c = n_unit
#     except TypeError:
#         n_unit_a = n_unit_b = n_unit_c = n_unit
#
#     a_vec, b_vec, c_vec = O.T
#     vecs = np.array([i * a_vec + j * b_vec + k * c_vec
#                           for i in range(n_unit_a)
#                           for j in range(n_unit_b)
#                           for k in range(n_unit_c)])
#
#     # sphericalize the lattice..
#     if spherical:
#         vecs = simutils.sphericalize(vecs)
#
#     return vecs
