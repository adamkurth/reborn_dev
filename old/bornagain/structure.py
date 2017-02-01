import numpy as np
from numpy import cos, sin, array, zeros, complex
from Bio import PDB
import re
import xraylib
from bornagain import utils

class molecule(object):

    """ A collection of atomic positions, with specified elements, 
    and transformation operations. Atomic scattering factors on request."""

    def __init__(self):

        self._r = None  # List of atomic coordinates (Nx3 array)
        self._elem = None  # List of element names
        self._Z = None  # List of atomic numbers
        self._f = None  # List of scattering factors
        self.R = None  # Rotation of this molecule
        self.T = None  # Translation of this molecule
        self._photonEnergy = None  # Photon energy corresponding to scattering factors

    @property
    def r(self):

        """ Atomic coordinates (immutable) """

        return self._r

    @r.setter
    def r(self, r):

        self._r = r

    @property
    def elem(self):

        """ Atomic element (strings, immutable). """

        return self._elem

    @elem.setter
    def elem(self, elem):

        if self.r is not None:
            if len(elem) != len(self.r):
                raise ValueError("Number of elements does not match number of atomic positions.")

        self._elem = [e.capitalize() for e in elem]

    @property
    def Z(self):

        """ Atomic numbers generated from element strings. """

        if self._Z is None:
            if self.elem is None:
                raise ValueError("Elements are not defined.")
            self._Z = array([xraylib.SymbolToAtomicNumber(elem) for elem in self.elem])

        return self._Z

    def nAtoms(self):

        """ Number of atoms """

        return len(self.r)

    def groupedElements(self):

        """ Return coordinates for each element """

        uZ = np.unique(self.Z)
        r = [self.r[self.Z == Z, :] for Z in uZ]
        return (r, uZ)

    def r_t(self):

        """ Transformed coordinates """

        r = self.r.copy()

        if self.R is not None:
            r = self.R * r

        if self.T is not None:
            r = r + self.T

        return r

    def fromPdb(self, pdbFile):

        """ Load atoms from PDB file (not sure if it loads everything, e.g. HETATM) """

        p = PDB.PDBParser()
        s = p.get_structure('dummy', pdbFile)
        m = s[0]

        self.r = array([x.coord for x in m.get_atoms()]) * 1e-10
        self.elem = array([x.element for x in m.get_atoms()])





class crystal(object):

    """ Information pertaining to a single ideal crystal of finite size. """

    def __init__(self):

        self.cellParameters = None
        self._O = None
        self.r = None
        self.elem = None
        self.molecules = None

    @property
    def nMolecules(self):

        if self.molecules is None:
            n = 0
        else:
            n = len(self.molecules)
        return n

    def addMolecule(self, mol):

        if self.molecules is None:
            self.molecules = []

        self.molecules.append(mol)


    @property
    def O(self):

        """ As defined in Rupp's book """

        if self._O is None:

            p = self.cellParameters
            a = p["a"]
            b = p["b"]
            c = p["c"]
            al = p["alpha"]
            be = p["beta"]
            ga = p["gamma"]
            v = a * b * c * np.sqrt(1 - cos(al) ** 2 - cos(be) ** 2 - cos(ga) ** 2 + 2 * cos(al) * cos(be) * cos(ga))
            aa = array([a, 0, 0])
            bb = array([a * cos(ga), b * sin(ga), 0])
            cc = array([c * cos(ga), c * (cos(al) - cos(be) * cos(ga)) / sin(ga), v / a / b / sin(ga)])

            self._O = zeros([3, 3], dtype=np.double)
            self._O[:, 0] = aa
            self._O[:, 1] = bb
            self._O[:, 2] = cc

        return self._O


def loadPdbFile(filePath):

    cryst = crystal()

    # get atomic coordinates of unit cell
    p = PDB.PDBParser()
    struc = p.get_structure('dummy', filePath)
    model = struc[0]
    r = array([x.coord for x in model.get_atoms()]) * 1e-10
    e = array([x.element for x in model.get_atoms()])
    mol = molecule()
    mol.r = r
    mol.elem = e
    cryst.addMolecule(mol)

    # get unit cell
    p = re.compile(r'^CRYST1*')
    cryst1 = None
    f = open(filePath, 'r')
    for line in f:
        if p.match(line):
            cryst1 = line
            break
    f.close()

    if cryst1 is not None:
        a = float(cryst1[6:14])
        b = float(cryst1[15:23])
        c = float(cryst1[24:32])
        alpha = float(cryst1[33:39])
        beta = float(cryst1[40:46])
        gamma = float(cryst1[47:53])
        cell = dict()
        cell['a'] = a
        cell['b'] = b
        cell['c'] = c
        cell['alpha'] = alpha
        cell['beta'] = beta
        cell['gamma'] = gamma
        cryst.cellParameters = cell

    return cryst
