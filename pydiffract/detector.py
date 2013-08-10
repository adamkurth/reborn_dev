
import numpy as np
from utils import warn
import source
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
        self.pixSize = 0
        self._F = np.array([1, 0, 0], dtype=self.dtype)
        self._S = np.array([0, 1, 0], dtype=self.dtype)
        self._T = np.array([0, 0, 0], dtype=self.dtype)
        self.aduPerEv = 0
        self.dataPlan = None
        self.beam = source.beam()
        self.data = np.zeros([0, 0], dtype=self.dtype)
        self._V = None
        self._K = None

        self.panelList = None

    def copy(self):

        p = panel()
        p.name = self.name
        p.pixSize = self.pixSize
        p._F = self._F.copy()
        p._S = self._S.copy()
        p._T = self._T.copy()
        p.aduPerEv = self.aduPerEv
        p.dataPlan = self.dataPlan
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
        s += "pixSize = %g\n" % self.pixSize
        s += "F = [%g, %g, %g]\n" % (self.F[0], self.F[1], self.F[2])
        s += "S = [%g, %g, %g]\n" % (self.S[0], self.S[1], self.S[2])
        s += "nF = %d\n" % self.nF
        s += "nS = %d\n" % self.nS
        s += "T = [%g, %g, %g]\n" % (self.T[0], self.T[1], self.T[2])
        s += "aduPerEv = %g\n" % self.aduPerEv
        s += self.data.__str__()
        return s

    @property
    def nF(self):
        if self.data is not None:
            return self.data.shape[1]
        return 0

    @property
    def nS(self):
        if self.data is not None:
            return self.data.shape[0]
        return 0

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
            self._V = None
            self._K = None
        else:
            raise ValueError("Must be a numpy array of length 3")

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, val):
        if isinstance(val, np.ndarray) and val.size == 3 and val.ndim == 1:
            self._S = self.dtype(val)
            self._V = None
            self._K = None
        else:
            raise ValueError("Must be a numpy array of length 3")

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, val):
        if isinstance(val, np.ndarray) and val.size == 3 and val.ndim == 1:
            self._T = self.dtype(val)
            self._V = None
            self._K = None
        else:
            raise ValueError("Must be a numpy array of length 3")

    @property
    def V(self):
        if self._V is None:
            self.computeRealSpaceGeometry()
        return self._V

    def check(self):

        if self.pixSize <= 0:
            warn("Bad pixel size in panel %s" % self.name)
            return False
        return True

    def computeRealSpaceGeometry(self):

        if self.nF == 0 or self.nS == 0:
            raise ValueError("Data array has zero size (%d x %d)" % (self.nF, self.nS))

        if self.pixSize <= 0:
            raise ValueError("Pixel size must be >= 0")

        p = self.pixSize
        i = np.arange(self.nF)
        j = np.arange(self.nS)
        [i, j] = np.meshgrid(i, j)
        i.ravel()
        j.ravel()
        F = np.outer(i, self._F)
        S = np.outer(j, self._S)
        self._V = self.T + F * p + S * p

#     def getRealSpaceBoundingBox(self):
#
#         vmin = np.ones(3) * np.finfo(np.float64).max
#         vmax = np.ones(3) * np.finfo(np.float64).min



class panelList(list):

    def __init__(self):

        """ Just make an empty panel array """

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

        V = np.empty((self.nPix, 3))

        n = 0
        for p in self:
            nPix = p.nPix
            nF = p.nF
            nS = p.nS
            V[n:(n + nPix), :] = p.V.reshape((nPix, 3))
            p._V = V[n:(n + nPix)]
            p._V = p._V.reshape((nS, nF, 3))
            n += nPix

        return V

    @property
    def data(self):

        data = np.empty(self.nPix)

        n = 0
        for p in self:
            nPix = p.nPix
            nF = p.nF
            nS = p.nS
            data[n:(n + nPix)] = p.data.ravel()
            p.data = data[n:(n + nPix)]
            p.data = p.data.reshape((nS, nF))
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

    def read(self, fileName):

        self[0].dataPlan.read(self, fileName)


class GeometryError(Exception):
    pass
