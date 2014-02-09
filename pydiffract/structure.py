import numpy as np
from numpy import cos, sin, array, zeros
from Bio import PDB
import re

class molecule(object):

    """ A collection of atomic positions, with specified elements. """

    def __init__(self):

        self.r = None  # List of atomic coordinates (Nx3 array)
        self.elem = None  # List of element names
        self.Z = None  # List of atomic numbers
        self.f = None  # List of scattering factors
        self.R = None  # Rotation of this molecule
        self.T = None  # Translation of this molecule

    def nAtoms(self):

        return len(self.r)

    def groupedElements(self):

        """ Return coordinates for each element """

        u = np.unique(self.elem)
        r = [self.r[self.elem == uu] for uu in u]
        return (r, u)


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
    r = np.array([x.coord for x in model.get_atoms()]) * 1e-10
    e = np.array([x.element for x in model.get_atoms()])
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





