# coding=utf-8
'''
Basic utilities for dealing with crystalline objects.
'''

import numpy as np
from numpy import sin, cos, sqrt

from bornagain.target import spgrp
from bornagain import utils
from bornagain.simulate import atoms


class structure(object):

    r"""
    A container class for stuff needed when dealing with crystal structures.
    """

    r = None  #: Atomic coordinates (3xN array)
    _x = None  #: Fractional coordinates (3xN array)
    O = None  #: Orthogonalization matrix (3x3 array).  Does the transform r = dot(O, x), with fractional coordinates x.
    Oinv = None  #: Inverse orthogonalization matrix (3x3 array)
    A = None  #: This is Oinv transpose (3x3 array).  Does the transform q = dot(A, h), with fractional Miller indices h.
    Ainv = None  #: A inverse
    T = None  #: Translation vector that goes with orthogonalization matrix (What is this used for???)
    elements = None  #: Atomic element symbols
    Z = None  #: Atomic numbers
    spaceGroupNumber = None  #: Space group number in the Int. Tables
    hermannMauguinSymbol = None  #: Spacegroup Hermann Mauguin symbol (e.g. as it appears in a PDB file for example)
    a = None  #: Lattice constant
    b = None  #: Lattice constant
    c = None  #: Lattice constant
    alpha = None  #: Lattice angle
    beta = None  #: Lattice angle
    gamma = None  #: Lattice angle
    V = None  #: Unit cell volume
    nAtoms = None  #: Number of atoms
    nMolecules = None  #: Number of molecules per unit cell
    symOps = None  #: Symmetry operations that are applied to fractional coordinates
    symRs = None #: Symmetry 3x3 transformation matrices (in crystal basis)
    symTs = None #: Symmetry translations (in crystal basis)

    def __init__(self, pdbFilePath=None):

        if pdbFilePath is not None:
            self.load_pdb(pdbFilePath)

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

    def set_spacegroup(self, hermann_mauguin_symbol):

        r"""

        Set the spacegroup of the crystal.  This produces a cache of the symmetry transformation operations.

        Args:
            hermann_mauguin_symbol:  This can be a string (like 'P 63' or a number 1-530)

        """

        self.hermann_mauguin_symbol = hermann_mauguin_symbol
        self.symRs, self.symTs = get_symmetry_operators_from_space_group(hermann_mauguin_symbol)
        self.symRinvs = [np.linalg.inv(R) for R in self.symRs]
        self.nMolecules = len(self.symTs)


    @property
    def x(self):

        r"""

        Fractional coordinates of atoms.

        Returns: Nx3 numpy array

        """

        if self._x is None:
            self._x = np.dot(self.Oinv, self.r.T).T
        return self._x


def parse_pdb(pdbFilePath, crystalStruct=None):

    r"""Return a :class:`structure` object with PDB information. """

    maxAtoms = int(1e5)
    r = np.zeros([3, maxAtoms])
    elements = []
    atomIndex = int(0)
    if crystalStruct is None:
        cryst = structure()
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


def get_symmetry_operators_from_space_group(hm_symbol):
    """
For a given Hermann-Mauguin spacegroup, provide the symmetry operators in the form of
translation and rotation operators.  These operators are in the crystal basis.
As of now, this is a rather unreliable function.  My brain hurts badly after attempting
to make sense of the ways in which space groups are specified...

Input: A string indicating the spacegroup in Hermann–Mauguin notation (as found in a PDB file)

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
    if isinstance(hm_symbol, str):
        hm_symbol = hm_symbol.strip()
        for i in range(0, 530):
            if hm_symbol == spgrp._hmsym[i].strip():
                symbol_found = True
                break
    else:
        i = hm_symbol
        if i < 530:
            symbol_found = True

    if not symbol_found:
        return None, None

    Rs = spgrp._spgrp_ops[i]["rotations"]
    Ts = [utils.vec_check(T) for T in spgrp._spgrp_ops[i]['translations']]

    return Rs, Ts


class Atoms:
    def __init__(self, xyz, atomic_num, elem=None):
        self.xyz = xyz
        self.x = self.xyz[:, 0]
        self.y = self.xyz[:, 1]
        self.z = self.xyz[:, 2]
        self.Z = atomic_num

        self.coor = np.zeros((self.x.shape[0], 4))
        self.coor[:, :3] = self.xyz
        self.coor[:, 3] = self.Z

        self.coor[:, 3] = self.Z

        self.elem = elem
        if elem is not None:
            self.xyz_format = np.zeros((self.x.shape[0], 4), dtype='S16')
            self.xyz_format[:, 0] = self.elem
            self.xyz_format[:, 1:] = self.xyz.astype(str)

    @classmethod
    def aggregate(cls, atoms_list):

        xyz = np.vstack([a.xyz for a in atoms_list])

        if all([a.elem is not None for a in atoms_list]):
            elem = np.hstack([a.elem for a in atoms_list])
        else:
            elem = None

        Z = np.hstack([a.Z for a in atoms_list])

        return cls(xyz, Z, elem)

    def to_xyz(self, fname):
        if self.elem is not None:
            np.savetxt(fname, self.xyz_format, fmt='%s')
        else:
            print("Cannot save to xyz because element strings were not provided...")

    def set_elem(self, elem):
        """sets list of elements names for use in xyz format files"""
        elem = np.array(elem, dtype='S16')
        assert(self.elem.shape[0] == self.x.shape[0])
        self.elem = elem


class Molecule(structure):
    def __init__(self, *args, **kwargs):
        structure.__init__(self, *args, **kwargs)

        self.atom_vecs = self.r * 1e10  # atom positions!

        self.lat = Lattice(self.a * 1e10, self.b * 1e10, self.c * 1e10,
                           self.alpha * 180 / np.pi, self.beta * 180 / np.pi, self.gamma * 180 / np.pi)

        self.atom_fracs = self.mat_mult_many(self.Oinv * 1e-10, self.atom_vecs)

    def _separate_xyz(self, xyz):
        x,y,z = map(np.array, zip(*xyz))
        return x,y,z

    def get_1d_coords(self):
        x, y, z = self._seprate_xyz( self.atom_vecs)
        return x, y, z

    def get_1d_frac_coords(self):
        x, y, z = self._separate_xyz(self.atom_fracs)
        return x, y, z

    def mat_mult_many(self, M, V):
        """ helper for applying matrix multiplications on many vectors"""
        return np.einsum('ij,kj->ki', M, V)

    def shift(self, monomer, na, nb, nc):
        xyz_frac =  self.mat_mult_many( self.Oinv*1e-10, monomer.xyz)
        x,y,z = self._separate_xyz(xyz_frac)
        
        x += na
        y += nb
        z += nc
        
        xyz_new = np.zeros_like(monomer.xyz)
        xyz_new[:,0] = x
        xyz_new[:,1] = y
        xyz_new[:,2] = z
        xyz_new = self.mat_mult_many(self.O * 1e10, xyz_new)
        return Atoms(xyz_new, self.Z, self.elements)


    def transform(self, x, y, z):
        """x,y,z are fractional coordinates"""
        xyz = np.zeros((x.shape[0], 3))
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = z
        xyz = self.mat_mult_many(self.O * 1e10, xyz)
        return Atoms(xyz, self.Z, self.elements)

    def get_monomers(self):
        monomers = []
        for R, T in zip(self.symRs, self.symTs):
            transformed = self.mat_mult_many(R, self.atom_fracs) + T
            transformed = self.mat_mult_many(self.O * 1e10, transformed)
            monomers.append(Atoms(transformed, self.Z, self.elements))
        return monomers


class Lattice:
    def __init__(self, a=281., b=281., c=165.2,
                 alpha=90., beta=90., gamma=120.):
        """
        a,b,c are in Angstroms
        alpha, beta, gamma are in degrees
        default is for PS1
        """
#       unit cell edges
        alpha = alpha * np.pi / 180.
        beta = beta * np.pi / 180.
        gamma = gamma * np.pi / 180.

        cos = np.cos
        sin = np.sin
        self.V = a * b * c * np.sqrt(1 - cos(alpha)**2 - cos(beta) **
                                     2 - cos(gamma)**2 + 2 * cos(alpha) * cos(beta) * cos(gamma))
        self.a = np.array([a, 0, 0])
        self.b = np.array([b * cos(gamma), b * sin(gamma), 0])
        self.c = np.array([c * cos(beta),
                           c * (cos(alpha) - cos(beta) *
                                cos(gamma)) / sin(gamma),
                           self.V / (a * b * sin(gamma))])
        self.O = np.array([self.a, self.b, self.c]).T
        self.Oinv = np.linalg.inv(self.O)

    def assemble(self, n_unit=10, spherical=False):

        #       lattice coordinates
        self.vecs = np.array([i * self.a + j * self.b + k * self.c
                              for i in xrange(n_unit)
                              for j in xrange(n_unit)
                              for k in xrange(n_unit)])

#       sphericalize the lattice..
        if spherical:
            self.vecs = utils.sphericalize(self.vecs)
