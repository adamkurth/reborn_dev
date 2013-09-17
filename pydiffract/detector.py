
import numpy as np
from numpy.linalg import norm
from pydiffract.utils import vecNorm
from pydiffract import source

"""
Classes for analyzing diffraction data contained in pixel array detectors (PAD).
"""


class panel(object):
    """ Individual detector panel: a 2D lattice of square pixels."""

    def __init__(self, name=""):
        """ Make no assumptions during initialization."""

        self.dtype = np.float64  # Choose the data type (this may go away)
        # Configured parameters
        self.name = name  # Panel name for convenience
        self._F = None  # Fast-scan vector
        self._S = None  # Slow-scan vector
        self._T = None  # Translation of this panel (from interaction region to center of first pixel)
        self._nF = 0  # Number of pixels along the fast-scan direction
        self._nS = 0  # Number of pixels along the slow-scan direction
        self.aduPerEv = 0  # Number of arbitrary data units per eV of photon energy
        self.beam = None  # Container for x-ray beam information
        self.data = None  # Diffraction intensity data
        # Derived parameters
        self._derivedGeometry = ['_pixSize', '_V', '_sa', '_K', '_rsbb']
        self._pixSize = None  # Pixel size derived from F/S vectors
        self._V = None  # 3D vectors pointing from interaction region to pixel centers
        self._sa = None  # Solid angles corresponding to each pixel
        self._K = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        # Sanity checks
        self._validGeometry = False
        # If this panel is a part of a list
        self.panelList = None  # This is the link to the panel list

    def copy(self):
        """ Deep copy of everything.  Parent panel list is stripped away."""

        p = panel()
        p.name = self.name
        p.F = self._F.copy()
        p.S = self._S.copy()
        p.T = self._T.copy()
        p.nF = self.nF
        p.nS = self.nS
        p.aduPerEv = self.aduPerEv
        p.beam = self.beam.copy()
        p.data = self.data.copy()
        if self._V is not None:
            p._V = self._V.copy()
        if self._K is not None:
            p._K = self._K.copy()
        if self._sa is not None:
            p._sa = self._sa.copy()
        p.panelList = None

        return p

    def __str__(self):
        """ Print something useful when in interactive mode."""

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
        """ Number of fast-scan pixels."""
        return self._nF

    @nF.setter
    def nF(self, val):
        """ Changing the fast-scan pixel count destroys all derived geometry 
        data as well as intensity data."""
        self._nF = np.int32(val)
        self.deleteGeometryData()
        self.data = None

    @property
    def nS(self):
        """ Number of slow-scan pixels."""
        return self._nS

    @nS.setter
    def nS(self, val):
        """ Changing the slow-scan pixel count destroys all derived geometry 
        data as well as intensity data."""
        self._nS = np.int32(val)
        self.deleteGeometryData()
        self.data = None

    @property
    def pixSize(self):
        """ Return the pixel size only if both fast/slow-scan vectors are the same length."""
        if self._pixSize is None:
            if self._validGeometry == False:
                self.checkGeometry()
            p1 = norm(self.F)
            p2 = norm(self.S)
            if abs(p1 - p2) / float(p2) > 1e-6 or abs(p1 - p2) / float(p1) > 1e-6:
                raise ValueError("Pixel size is not consistent between F and S vectors (%10f, %10f)." % (p1, p2))
            self._pixSize = np.mean([p1, p2])
        return self._pixSize.copy()

    @property
    def nPix(self):
        """ Total number of pixels."""
        return self._nF * self._nS

    @property
    def F(self):
        """ Fast-scan vector (length equal to pixel size)."""
        return self._F

    @F.setter
    def F(self, val):
        """ Must be a numpy ndarray of length 3."""
        if isinstance(val, np.ndarray) and val.size == 3 and val.ndim == 1:
            self._F = self.dtype(val)
            self.deleteGeometryData()
        else:
            raise ValueError("F must be a numpy ndarray of length 3.")

    @property
    def S(self):
        """ Slow-scan vector (length equal to pixel size)."""
        return self._S

    @S.setter
    def S(self, val):
        """ Must be a numpy ndarray of length 3."""
        if isinstance(val, np.ndarray) and val.size == 3 and val.ndim == 1:
            self._S = self.dtype(val)
            self.deleteGeometryData()
        else:
            raise ValueError("S must be a numpy array of length 3.")

    @property
    def T(self):
        """ Translation vector pointing from interaction region to center of first pixel."""
        return self._T

    @T.setter
    def T(self, val):
        """ Must be an ndarray of length 3."""
        if isinstance(val, np.ndarray) and val.size == 3 and val.ndim == 1:
            self._T = self.dtype(val)
            self.deleteGeometryData()
        else:
            raise ValueError("Must be a numpy array of length 3.")

    @property
    def B(self):
        """ Beam direction vector."""
        if self.beam is None:
            raise ValueError("Panel has no beam information.")
        return self.beam.B

    @property
    def V(self):
        """ Vectors pointing from interaction region to pixel centers."""
        if self._V is None:
            self.computeRealSpaceGeometry()
        return self._V

    @property
    def K(self):
        """ Scattering vectors multiplied by wavelength."""
        if self._K is None:
            self.computeReciprocalSpaceGeometry()
        return self._K

    @property
    def N(self):
        """ Normal vector to panel surface."""
        N = np.cross(self.F, self.S)
        return N / norm(N)

    @property
    def solidAngle(self):
        """ Solid angles of pixels."""
        if self._sa is None:
            v = vecNorm(self.V)
            n = self.N
            V2 = np.sum(self.V ** 2, axis=-1)
            A = norm(np.cross(self.F, self.S))
            self._sa = A / V2 * np.dot(v, n)
        return self._sa

    def check(self):
        """ Check for any known issues with this panel."""

        self.checkGeometry()
        return True

    def checkGeometry(self):
        """ Check for valid geometry configuration."""

        if self._T is None:
            raise ValueError("Panel translation vector T is not defined.")
        if self._F is None:
            raise ValueError("Panel basis vector F is not defined.")
        if self._S is None:
            raise ValueError("Panel basis vector S is not defined.")
        if self.nF == 0 or self.nS == 0:
            raise ValueError("Data array has zero size (%d x %d)." % (self.nF, self.nS))

        self._geometryChecked = True

    def computeRealSpaceGeometry(self):

        if self._validGeometry == False:
            self.checkGeometry()

        i = np.arange(self.nF)
        j = np.arange(self.nS)
        [i, j] = np.meshgrid(i, j)
        i.ravel()
        j.ravel()
        self._V = self.pixelsToVectors(i, j)
        # self._V = self._V.reshape((self._nS, self._nF, 3))

    def pixelsToVectors(self, i, j):
        """ Convert pixel indices to translation vectors (i=fast scan, j=slow scan)."""

        F = np.outer(i, self.F)
        S = np.outer(j, self.S)
        V = self.T + F + S
        return V

    def computeReciprocalSpaceGeometry(self):

        self._K = self.V - self.B

    def deleteGeometryData(self):

        """ Clear out all generated geometry data."""

        self._V = None
        self._sa = None
        self._K = None
        self._rsbb = None
        if self.panelList is not None:
            self.panelList.deleteGeometryData()
        self._validGeometry = False

    def getVertices(self, corners=False):
        """ Get panel getVertices; positions of corner pixels."""

        nF = self.nF - 1
        nS = self.nS - 1
        v = self.pixelsToVectors([0, 0, nF, nF], [0, nS, nS, 0])
        return v

    def getRealSpaceBoundingBox(self):
        """ Return the minimum and maximum values of the four corners."""

        if self._rsbb is not None:
            return self._rsbb

        r = np.zeros([3, 2])
        r[:, 0] = np.ones(3) * np.finfo(self.dtype).max
        r[:, 1] = np.ones(3) * np.finfo(self.dtype).min
        v = self.getVertices()
        r[0, 0] = min(v[:, 0])
        r[1, 0] = min(v[:, 1])
        r[2, 0] = min(v[:, 2])
        r[0, 1] = max(v[:, 0])
        r[1, 1] = max(v[:, 1])
        r[2, 1] = max(v[:, 2])

        self._rsbb = r

        return r

class panelList(list):

    def __init__(self):

        """ Create an empty panel array."""

        self._data = None
        self._sa = None
        self._V = None
        self._K = None
        self.isConsolidated = False
        self.beam = None
        self._rsbb = None

    def copy(self):

        """ Create a deep copy."""

        pa = panelList()
        for p in self:
            pa.append(p.copy())

        return pa

    def __str__(self):

        """ Print something useful when in interactive mode."""

        s = ""
        for p in self:
            s += "\n\n" + p.__str__()
        return(s)

    def __getitem__(self, key):

        """ Get a panel. Panels may be referenced by name or index."""

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

        if self._V is None:
            self._V = np.empty((self.nPix, 3))

        n = 0
        for p in self:
            if p.V.base is self._V:
                continue
            nPix = p.nPix
            nF = p.nF
            nS = p.nS
            self._V[n:(n + nPix), :] = p._V.reshape((nPix, 3))
            # p._V = self._V[n:(n + nPix)].reshape((nS, nF, 3))
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
            self._K[n:(n + nPix), :] = p._K.reshape((nPix, 3))
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
    def solidAngle(self):

        if self._sa == None:
            sa = np.empty(self.nPix)
            n = 0
            for p in self:
                nPix = p.nPix
                nF = p.nF
                nS = p.nS
                sa[n:(n + nPix)] = p.solidAngle.ravel()
                # p.data = data[n:(n + nPix)].reshape((nS, nF))
                n += nPix
            self._sa = sa
        return self._sa

    @property
    def simpleRealSpaceProjection(self):

        pixSize = self[0].pixSize
        r = self.realSpaceBoundingBox
        rpix = r / pixSize
        rpix[:, 0] = np.floor(rpix[:, 0])
        rpix[:, 1] = np.ceil(rpix[:, 1])

        adat = np.zeros([rpix[1, 1] - rpix[1, 0] + 1, rpix[0, 1] - rpix[0, 0] + 1])
        cdat = adat.copy()

        V = self.V[:, 0:2] / pixSize - rpix[0:2, 0]
        Vll = np.round(V).astype(np.int32)
#         Vll = np.floor(V).astype(np.int32)
#         Vuu = Vll + 1
#         Vlu = Vll.copy()
#         Vlu[:, 0] = Vll[:, 0]
#         Vlu[:, 1] = Vuu[:, 1]
#         Vul = Vll.copy()
#         Vul[:, 0] = Vuu[:, 0]
#         Vul[:, 1] = Vll[:, 1]

#         Wuu = 1 / np.sqrt(np.sum((V - Vuu) ** 2, axis=-1))
#         Wll = 1 / np.sqrt(np.sum((V - Vll) ** 2, axis=-1))
#         Wul = 1 / np.sqrt(np.sum((V - Vul) ** 2, axis=-1))
#         Wlu = 1 / np.sqrt(np.sum((V - Vlu) ** 2, axis=-1))

        d = self.data
        nPix = self.nPix

        adat[Vll[:, 1], Vll[:, 0]] += d  # * Wll
#         adat[Vuu[:, 1], Vuu[:, 0]] += d * Wuu
#         adat[Vul[:, 1], Vul[:, 0]] += d * Wul
#         adat[Vlu[:, 1], Vlu[:, 0]] += d * Wlu

#         cdat[Vll[:, 1], Vll[:, 0]] += Wll
#         cdat[Vuu[:, 1], Vuu[:, 0]] += Wuu
#         cdat[Vul[:, 1], Vul[:, 0]] += Wul
#         cdat[Vlu[:, 1], Vlu[:, 0]] += Wlu
#
#         adat /= cdat

        # adat /= Wsum.reshape(adat.shape)

        return adat

    @property
    def realSpaceBoundingBox(self):

        if self._rsbb is not None:
            return self._rsbb

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

        self._rsbb = r
        return r.copy()

    def deleteGeometryData(self):

        self._sa = None
        self._rsbb = None
        self._K = None
        self._V = None


class GeometryError(Exception):
    pass
