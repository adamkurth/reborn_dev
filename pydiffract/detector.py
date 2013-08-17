
import numpy as np
from numpy.linalg import norm
from pydiffract.utils import warn
from pydiffract import source
import numexpr
# import operator

"""
Classes for analyzing diffraction data contained in pixel array detectors (PAD)
"""


class panel(object):

    """
    Individual detector panel, assumed to be a 2D lattice of pixels.
    """

    def __init__(self, name=""):

        """
        There are no default initialization parameters.  Zeros generally mean
        "uninitialized"
        """

        self.dtype = np.float64
        self.name = name
        self._F = None
        self._S = None
        self._T = None
        self._nF = 0
        self._nS = 0
        self.aduPerEv = 0
        self.beam = source.beam()
        self.data = None
        self._V = None
        self._K = None

        self.panelList = None

    def copy(self):

        p = panel()
        p.name = self.name
        p.F = self._F.copy()
        p.S = self._S.copy()
        p.T = self._T.copy()
        p.aduPerEv = self.aduPerEv
        p.beam = self.beam.copy()
        p.data = self.data.copy()
        if self._V is not None:
            p._V = self._V.copy()
        if self._K is not None:
            p._K = self._K.copy()

        p.panelList = None

        return p

    def __str__(self):

        s = ""
        s += "name = \"%s\"\n" % self.name
        s += "pixSize = %s\n" % self.pixSize.__str__()
        s += "F = %s\n" % self.F.__str__()
        s += "S = %s\n" % self.S.__str__()
        s += "nF = %d\n" % self.nF
        s += "nS = %d\n" % self.nS
        s += "T = %s\n" % self.T.__str__()
        s += "aduPerEv = %g\n" % self.aduPerEv
        return s

    @property
    def nF(self):
        return self._nF

    @nF.setter
    def nF(self, val):
        if not isinstance(val, int):
            raise ValueError("nS must be an integer")
        if val != self._nS:
            self.deleteGeometryData()
        self._nF = val

    @property
    def nS(self):
        return self._nS

    @nS.setter
    def nS(self, val):
        if not isinstance(val, int):
            raise ValueError("nS must be an integer")
        if val != self._nS:
            self.deleteGeometryData()
        self._nS = val

    @property
    def pixSize(self):
        if self._F is not None:
            return np.mean([norm(self.F), norm(self.S)])

    @property
    def nPix(self):
        if self.data is not None:
            return self.data.size
        return 0

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, val):
        if isinstance(val, np.ndarray) and val.size == 3 and val.ndim == 1:
            self._F = self.dtype(val)
            self.deleteGeometryData()
        else:
            raise ValueError("F must be a numpy ndarray of length 3")

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, val):
        if isinstance(val, np.ndarray) and val.size == 3 and val.ndim == 1:
            self._S = self.dtype(val)
            self.deleteGeometryData()
        else:
            raise ValueError("S must be a numpy array of length 3")

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, val):
        if isinstance(val, np.ndarray) and val.size == 3 and val.ndim == 1:
            self._T = self.dtype(val)
            self.deleteGeometryData()
        else:
            raise ValueError("Must be a numpy array of length 3")

    @property
    def B(self):
        if self.beam is None:
            raise ValueError("Panel has no beam information")
        return self.beam.B

    @property
    def V(self):
        if self._V is None:
            self.computeRealSpaceGeometry()
        return self._V

    @property
    def K(self):
        if self._K is None:
            self.computeReciprocalSpaceGeometry()
        return self._K

    def check(self):

        if self.pixSize <= 0:
            warn("Bad pixel size in panel %s" % self.name)
            return False
        return True

    def checkGeometry(self):

        if self._T is None:
            raise ValueError("Panel translation vector T is not defined")
        if self._F is None:
            raise ValueError("Panel basis vector F is not defined")
        if self._S is None:
            raise ValueError("Panel basis vector S is not defined")
        if self.nF == 0 or self.nS == 0:
            raise ValueError("Data array has zero size (%d x %d)" % (self.nF, self.nS))

    def computeRealSpaceGeometry(self):

        self.checkGeometry()

        i = np.arange(self.nF)
        j = np.arange(self.nS)
        [i, j] = np.meshgrid(i, j)
        i.ravel()
        j.ravel()
        F = np.outer(i, self._F)
        S = np.outer(j, self._S)
        self._V = self.T + F + S
        self._V = self._V.reshape((self._nS, self._nF, 3))

    def computeReciprocalSpaceGeometry(self):

        self._K = self.V - self.B

    def deleteGeometryData(self):
        self._V = None
        self._K = None

    def getRealSpaceBoundingBox(self):

        V = self.V

        r = np.zeros([3, 2])
        r[:, 0] = np.ones(3) * np.finfo(self.dtype).max
        r[:, 1] = np.ones(3) * np.finfo(self.dtype).min
        v = V[[0, 0, -1, -1], [0, -1, -1, 0], :]
        r[0, 0] = min(v[:, 0])
        r[1, 0] = min(v[:, 1])
        r[2, 0] = min(v[:, 2])
        r[0, 1] = max(v[:, 0])
        r[1, 1] = max(v[:, 1])
        r[2, 1] = max(v[:, 2])

        return r

class panelList(list):

    def __init__(self):

        """ Just make an empty panel array """

        self._V = None
        self._K = None
        self.isConsolidated = False
        self.beam = None

    def copy(self):

        pa = panelList()
        for p in self:
            pa.append(p.copy())

    def __str__(self):

        s = ""
        for p in self:
            s += "\n\n" + p.__str__()
        return(s)

    def __getitem__(self, key):

        if isinstance(key, str):
            key = self.getPanelIndexByName(key)
            if key is None:
                raise IndexError("There is no panel named %s" % key)
                return None
        return super(panelList, self).__getitem__(key)

    def __setitem__(self, key, value):

        if not isinstance(value, panel):
            raise TypeError("You may only add panels to a panelList object")
        if value.name == "":
            value.name = "%d" % key
        super(panelList, self).__setitem__(key, value)
        self.isConsolidated = False

    @property
    def nPix(self):
        ntot = 0
        for p in self:
            ntot += p.nPix
        return ntot

    @property
    def nPanels(self):
        return len(self)

    def append(self, p=None, name=""):

        if p is None:
            p = panel()
        if not isinstance(p, panel):
            raise TypeError("You may only append panels to a panelList object")
        p.panelList = self
        if name != "":
            p.name = name
        else:
            p.name = "%d" % len(self)
        super(panelList, self).append(p)
        self.isConsolidated = False

    def getPanelIndexByName(self, name):

        """ Find the integer index of a panel by it's unique name """
        i = 0
        for p in self:
            if p.name == name:
                return i
            i += 1
        return None

    def computeRealSpaceGeometry(self):

        for p in self:
            p.computeRealSpaceGeometry()

    @property
    def V(self):

        if self._V is None or self._V.shape[0] != self.nPix:
            self._V = np.empty((self.nPix, 3))

        n = 0
        for p in self:
            if p.V.base is self._V:
                continue
            nPix = p.nPix
            nF = p.nF
            nS = p.nS
            self._V[n:(n + nPix), :] = p._V.reshape((nPix, 3))
            p._V = self._V[n:(n + nPix)].reshape((nS, nF, 3))
            n += nPix

        return self._V

    @property
    def K(self):

        if self._K is None or self._K.shape[0] != self.nPix:
            self._K = np.empty((self.nPix, 3))

        n = 0
        for p in self:
            if p.K.base is self._K:
                continue
            nPix = p.nPix
            nF = p.nF
            nS = p.nS
            self._K[n:(n + nPix), :] = p._K.reshape((nPix, 3))
            p._K = self._K[n:(n + nPix)].reshape((nS, nF, 3))
            n += nPix

        return self._K

    @property
    def data(self):

        data = np.empty(self.nPix)

        n = 0
        for p in self:
            nPix = p.nPix
            nF = p.nF
            nS = p.nS
            data[n:(n + nPix)] = p.data.ravel()
            p.data = data[n:(n + nPix)].reshape((nS, nF))
            n += nPix

        return data

    @data.setter
    def data(self, data):

        if not isinstance(data, np.ndarray) and data.ndim == 1 and data.size == self.nPix:
            raise ValueError("Must be flattened ndarray of size %d" % self.nPix)
        n = 0
        for p in self:
            nPix = p.nPix
            nF = p.nF
            nS = p.nS
            p.data = data[n:(n + nPix)]
            p.data = p.data.reshape((nS, nF))
            n += nPix

    @property
    def assembledData(self):

        pass

    def getRealSpaceBoundingBox(self):


        r = np.zeros([3, 2])
        r[:, 0] = np.ones(3) * np.finfo(np.float64).max
        r[:, 1] = np.ones(3) * np.finfo(np.float64).min

        for p in self:

            rp = p.getRealSpaceBoundingBox()
            r[0, 0] = min([r[0, 0], rp[0, 0]])
            r[1, 0] = min([r[1, 0], rp[1, 0]])
            r[2, 0] = min([r[2, 0], rp[2, 0]])
            r[0, 1] = max([r[0, 1], rp[0, 1]])
            r[1, 1] = max([r[1, 1], rp[1, 1]])
            r[2, 1] = max([r[2, 1], rp[2, 1]])

        return r

class GeometryError(Exception):
    pass
