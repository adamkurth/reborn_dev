# coding=utf-8
'''
Basic utilities for dealing with crystalline objects.

from Derek: Is using meters really the best here, given the PDB standard is Angstrom
and we will likey always deal with Angstrom scale coordinates?

for example print( "Lattice dim is %.3f"%(0.0000008)) will print 0.000, which can
cause problems...
'''

from __future__ import (absolute_import, division, print_function, unicode_literals)

# Python 2 and 3 compatibility...
try:
  basestring
except NameError:
  basestring = str

import numpy as np
from numpy import sin, cos, sqrt

import bornagain as ba
from bornagain.target import spgrp
from bornagain import utils
from bornagain.simulate import atoms, simutils


class SpaceGroup(object):

    r"""
    #TODO: document
    Gotchas: note that there are multiple Hall numbers that correspond to each ITOC number.  The 230 ITOC numbers that
    specify space groups refer only to the actual symmetry properties, and there are multiple Hall numbers and Hermann
    Mauguin symbols that have the same spacegroup.  The duplicates are just different ways of specifying the same
    spacegroup.
    """

    hall_number = None #: Space group Hall number
    itoc_number = None #: Space group number in the International Tables of Crystallography (ITOC)
    hermann_mauguin_symbol = None  #: Spacegroup Hermann Mauguin symbol (e.g. as it appears in a PDB file for example)
    sym_rotations = None #: Symmetry 3x3 transformation matrices (in crystal basis)
    sym_translations = None #: Symmetry translations (in crystal basis)
    n_molecules = None

    def __init__(self, hermann_mauguin_symbol=None, hall_number=None, itoc_number=None):

        if hall_number is not None:
            self.itoc_number = itoc_number_from_hall_number(hall_number)
            self.hermann_mauguin_symbol = hermann_mauguin_symbol_from_hall_number(hall_number)
        elif hermann_mauguin_symbol is not None:
            self.itoc_number = itoc_number_from_hermann_mauguin_symbol(hermann_mauguin_symbol)
            self.hall_number = hall_number_from_hermann_mauguin_symbol(hermann_mauguin_symbol)
        elif itoc_number is not None:
            self.hall_number = hall_number_from_itoc_number(itoc_number)
            self.hermann_mauguin_symbol = hermann_mauguin_symbol_from_hall_number(self.hall_number)
        else:
            raise ValueError('You must initialize SpaceGroup with a spacegroup identifier')

        self.sym_rotations, self.sym_translations = get_symmetry_operators_from_hall_number(self.hall_number)
        self.n_molecules = len(self.sym_rotations)


class CrystalStructure(object):
    r"""
    A container class for stuff needed when dealing with crystal structures.
    """
    # TODO: Needs documentation!

    r = None  #: Atomic coordinates (3xN array)
    _x = None  #: Fractional coordinates (3xN array)
    O = None  #: Orthogonalization matrix (3x3 array).  Does the transform r = dot(O, x), with fractional coordinates x.
    Oinv = None  #: Inverse orthogonalization matrix (3x3 array)
    A = None  #: This is Oinv transpose (3x3 array). Does the transform q = dot(A, h), with fractional Miller indices h.
    Ainv = None  #: A inverse
    T = None  #: Translation vector that goes with orthogonalization matrix (What is this used for???)
    elements = None  #: Atomic element symbols
    Z = None  #: Atomic numbers
    # spaceGroupNumber = None  #: Space group number in the Int. Tables
    # hermannMauguinSymbol = None  #: Spacegroup Hermann Mauguin symbol (e.g. as it appears in a PDB file for example)
    a = None  #: Lattice constant
    b = None  #: Lattice constant
    c = None  #: Lattice constant
    alpha = None  #: Lattice angle
    beta = None  #: Lattice angle
    gamma = None  #: Lattice angle
    V = None  #: Unit cell volume
    nAtoms = None  #: Number of atoms
    # nMolecules = None  #: Number of molecules per unit cell
    # symOps = None  #: Symmetry operations that are applied to fractional coordinates
    # symRs = None #: Symmetry 3x3 transformation matrices (in crystal basis)
    # symTs = None #: Symmetry translations (in crystal basis)

    spacegroup = None

    def __init__(self, pdbFilePath=None):

        if pdbFilePath is not None:
            self.load_pdb(pdbFilePath)

    @property
    def n_molecules(self):
        return self.nMolecules

    def load_pdb(self, pdbFilePath):

        r"""

        Populate the class with all the info from a PDB file.

        Args:
            pdbFilePath: Path to the PDB file

        """
        parse_pdb(pdbFilePath, self)

    def set_cell(self, a, b, c, alpha, beta, gamma):
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

        V = a * b * c * sqrt(1 - cos(al)**2 - cos(be) **
                             2 - cos(ga)**2 + 2 * cos(al) * cos(be) * cos(ga))
        O = np.array([
                [a, b * cos(ga), c * cos(be)],
                [0, b * sin(ga), c * (cos(al) - cos(be) * cos(ga)) / sin(ga)],
                [0, 0, V / (a * b * sin(ga))]
                ])
        Oinv = np.array([
                [1 / a, -cos(ga) / (a * sin(ga)), 0],
                [0, 1 / (b * sin(ga)), 0],
                [0, 0, a * b * sin(ga) / V]
                ])
        self.O = O
        self.Oinv = Oinv
        self.A = Oinv.T.copy()
        self.Ainv = O.T.copy()
        self.V = V

    def set_spacegroup(self, *args, **kwargs):

        r"""

        Set the spacegroup of the crystal.  This produces a cache of the symmetry transformation operations.

        Args:
            hermann_mauguin_symbol:  A string like 'P 63'
            hall_number: Hall number (between 1-530)
            itoc_number: Spacegroup number from International Tables of Crystallography (between 1-230)

        """

        self.spacegroup = SpaceGroup(*args, **kwargs)
        # self.hermann_mauguin_symbol = hermann_mauguin_symbol
        # self.symRs, self.symTs = get_symmetry_operators_from_space_group(hermann_mauguin_symbol)
        # self.symRinvs = [np.linalg.inv(R) for R in self.symRs]
        # self.nMolecules = len(self.symTs)

    @property
    def hermannMauguinSymbol(self):
        return self.spacegroup.hermann_mauguin_symbol

    @property
    def spaceGroupNumber(self):
        return self.spacegroup.itoc_number

    @property
    def nMolecules(self):
        return self.n_molecules

    @property
    def n_molecules(self):
        return self.spacegroup.n_molecules

    @property
    def symRs(self):
        return self.spacegroup.sym_rotations

    @property
    def symTs(self):
        return self.spacegroup.sym_translations

    @property
    def x(self):

        r"""

        Fractional coordinates of atoms.

        Returns: Nx3 numpy array

        """

        if self._x is None:
            self._x = np.dot(self.Oinv, self.r.T).T
        return self._x


class Structure(CrystalStructure):

    def __init__(self, *args, **kwargs):

        utils.depreciate('The class "crystal.Structure" is depreciated.  Use "crystal.CrystalStructure" instead.')

        CrystalStructure.__init__(self, *args, **kwargs)


class structure(CrystalStructure):

    def __init__(self, *args, **kwargs):

        utils.depreciate('The class "crystal.structure" is depreciated.  Use "crystal.CrystalStructure" instead.')

        CrystalStructure.__init__(self, *args, **kwargs)


class UnitCell(object):

    def __init__(self):

        pass


class FiniteLattice(object):

    def __init__(self, max_size=None):

        pass


def parse_pdb(pdbFilePath, crystalStruct=None):
    r"""Return a :class:`Structure` object with PDB information. """

    maxAtoms = int(1e5)
    r = np.zeros([3, maxAtoms])
    elements = []
    atomIndex = int(0)
    if crystalStruct is None:
        cryst = CrystalStructure()
    else:
        cryst = crystalStruct
    SCALE = np.zeros([3, 4])

    with open(pdbFilePath) as pdbfile:

        for line in pdbfile:

            # This is the inverse of the "orthogonalization matrix"  along with
            # translation vector.  See Rupp for an explanation.

            if line[:5] == 'SCALE':
                n = int(line[5]) - 1
                SCALE[n, 0] = float(line[10:20])
                SCALE[n, 1] = float(line[20:30])
                SCALE[n, 2] = float(line[30:40])
                SCALE[n, 3] = float(line[45:55])

            # The crystal lattice and symmetry

            if line[:6] == 'CRYST1':
                cryst1 = line
                # As always, everything in our programs are in SI units.
                # PDB files use angstrom units.
                a = float(cryst1[6:15]) * 1e-10
                b = float(cryst1[15:24]) * 1e-10
                c = float(cryst1[24:33]) * 1e-10
                # And of course degrees are converted to radians (though we
                # loose the perfection of rational quotients like 360/4=90...)
                al = float(cryst1[33:40]) * np.pi / 180.0
                be = float(cryst1[40:47]) * np.pi / 180.0
                ga = float(cryst1[47:54]) * np.pi / 180.0

                cryst.set_cell(a, b, c, al, be, ga)
                hermann_mauguin_symbol = cryst1[55:66].strip()

            if line[:6] == 'ATOM  ' or line[:6] == "HETATM":
                r[0, atomIndex] = float(line[30:38]) * 1e-10
                r[1, atomIndex] = float(line[38:46]) * 1e-10
                r[2, atomIndex] = float(line[46:54]) * 1e-10
                elements.append(line[76:78].strip().capitalize())
                # z = atoms.atomic_symbols_to_numbers(elements[atomIndex])
                # #xr.SymbolToAtomicNumber(elements[atomIndex])
                atomIndex += 1

            if atomIndex == maxAtoms:
                r = np.append(r, np.zeros([3, maxAtoms]), axis=1)

    # Truncate atom list since we pre-allocated extra memory
    nAtoms = atomIndex
    r = r[:, :nAtoms]
    elements = elements[:nAtoms]

    T = SCALE[:, 3]

    cryst.cryst1 = cryst1
    cryst.r = utils.vec_check(r)
    cryst.T = utils.vec_check(T)
    cryst.elements = elements
    cryst.Z = atoms.atomic_symbols_to_numbers(elements)
    cryst.nAtoms = nAtoms

    cryst.set_spacegroup(hermann_mauguin_symbol)

    return cryst


def get_hm_symbols():

    symbols = []
    for idx in range(0, 530):
        symbols.append(spgrp._hmsym[idx].strip())

    return symbols

# hermann_mauguin_symbols = get_hm_symbols()


def hermann_mauguin_symbol_from_hall_number(hall_number):

    if hall_number < 0 or hall_number > 530:
        raise ValueError('hall_number must be between 1 and 530')

    return spgrp._hmsym[hall_number-1]


def itoc_number_from_hall_number(hall_number):

    return spgrp._sgnum[hall_number-1]


def hall_number_from_hermann_mauguin_symbol(hermann_mauguin_symbol):

    idx = None
    hermann_mauguin_symbol = hermann_mauguin_symbol.strip()
    for idx in range(0, 530):
        if hermann_mauguin_symbol == spgrp._hmsym[idx]:
            break

    if idx is None:
        raise ValueError('Cannot find Hermann Mauguin symbol')

    return idx+1


def itoc_number_from_hermann_mauguin_symbol(hermann_mauguin_symbol):

    return itoc_number_from_hall_number(hall_number_from_hermann_mauguin_symbol(hermann_mauguin_symbol))


def hall_number_from_itoc_number(itoc_number):

    if itoc_number < 1 or itoc_number > 230:
        raise ValueError('ITOC spacegroup number must be in the range 1 to 230')

    idx = -1
    for idx in range(0, 530):
        if itoc_number == spgrp._sgnum[idx]:
            break
    return idx+1


def hermann_mauguin_symbol_from_itoc_number(itoc_number):

    return hermann_mauguin_symbol_from_hall_number(hall_number_from_itoc_number(itoc_number))


def spacegroup_ops_from_hall_number(hall_number):

    rot = spgrp._spgrp_ops[hall_number-1]["rotations"]
    trans = [utils.vec_check(T) for T in spgrp._spgrp_ops[hall_number-1]['translations']]

    return rot, trans


def get_symmetry_operators_from_hall_number(hall_number):

    if hall_number < 1 or hall_number > 530:
        raise ValueError("Hall number must be between 1 and 530")

    idx = hall_number - 1
    Rs = spgrp._spgrp_ops[idx]["rotations"]
    Ts = [utils.vec_check(T) for T in spgrp._spgrp_ops[idx]['translations']]

    return Rs, Ts


def get_symmetry_operators_from_space_group(hm_symbol):

    r"""
    For a given Hermann-Mauguin spacegroup, provide the symmetry operators in the form of
    translation and rotation operators.  These operators are in the crystal basis.
    As of now, this is a rather unreliable function.  My brain hurts badly after attempting
    to make sense of the ways in which space groups are specified... if you are having trouble
    finding your spacegroup check that it is contained in the output of print(crystal.get_hm_symbols()).

    Input:
        hm_symbol (str): A string indicating the spacegroup in Hermann–Mauguin notation (as found in a PDB file)

    Output: Two lists: Rs and Ts.  These correspond to lists of rotation matrices (3x3 numpy
            arrays) and translation vectors (1x3 numpy arrays)
    """

    # First we try to find the "Hall number" corresponding to the Hermann-Mauguin symbol.
    # This is done in a hacked way using the spglib module.  However, we do not use the
    # spglib module directly in order to avoid that dependency.  Instead, the operators
    # that result from the spglib.get_symmetry_from_database() method have been dumped
    # into the spgrp module included in bornagain.

    # Simply iterate through all HM symbols until we find one that matches:
    symbol_found = False
    idx = None
    if isinstance(hm_symbol, basestring):
        hm_symbol = hm_symbol.strip()
        for idx in range(0, 530):
            if hm_symbol == spgrp._hmsym[idx].strip():
                symbol_found = True
                break
    else:
        idx = hm_symbol
        if idx < 530:
            symbol_found = True

    if not symbol_found:
        return None, None

    Rs = spgrp._spgrp_ops[idx]["rotations"]
    Ts = [utils.vec_check(T) for T in spgrp._spgrp_ops[idx]['translations']]

    return Rs, Ts


def assemble(O, n_unit=10, spherical=False):
    r"""
    From Derek: assemble and assemble3 are identical functions, hence the try/except at the start
        n_unit can be a tuple or an integer

    Creates a finite lattice
    Args:
        O, Structure attribute O (3x3 ndarray), orientation matrix of crystal
            (columns are the lattice vectors a,b,c)
        n_unit (int or 3-tuple): Number of unit cells along each crystal axis
        spherical (bool): If true, apply a spherical boundary to the crystal.
    """

    #lattice coordinates
    try:
        n_unit_a, n_unit_b, n_unit_c = n_unit
    except TypeError:
        n_unit_a = n_unit_b = n_unit_c = n_unit

    a_vec, b_vec, c_vec = O.T
    vecs = np.array([i * a_vec + j * b_vec + k * c_vec
                          for i in range(n_unit_a)
                          for j in range(n_unit_b)
                          for k in range(n_unit_c)])

    # sphericalize the lattice..
    if spherical:
        vecs = simutils.sphericalize(vecs)

    return vecs
