
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
        self._pixSize = None  # Pixel size derived from F/S vectors
        self._V = None  # 3D vectors pointing from interaction region to pixel centers
        self._sa = None  # Solid angles corresponding to each pixel
        self._K = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        self._derivedGeometry = ['_pixSize', '_V', '_sa', '_K', '_rsbb']  # Default values of these are 'None'

        # Other internal data
        self._validGeometry = False  # True when geometry configuration is valid
        self.dtype = np.float64  # Choose the data type (this may go away)

        # If this panel is a part of a list
        self.panelList = None  # This is the link to the panel list

    def copy(self, derived=True):

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
        if derived == True:
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
        and intensity data."""

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

    @pixSize.setter
    def pixSize(self, val):

        pf = norm(self.F)
        ps = norm(self.S)
        self.F *= val / pf
        self.S *= val / ps

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

        self._validGeometry = True

    def computeRealSpaceGeometry(self):

        """ Compute arrays relevant to real-space geometry."""

        if self._validGeometry == False:
            self.checkGeometry()

        i = np.arange(self.nF)
        j = np.arange(self.nS)
        [i, j] = np.meshgrid(i, j)
        i.ravel()
        j.ravel()
        self._V = self.pixelsToVectors(j, i)
#         self._V = self._V.reshape((self._nS, self._nF, 3))

    def pixelsToVectors(self, j, i):

        """ Convert pixel indices to translation vectors (i=fast scan, j=slow scan)."""

        F = np.outer(i, self.F)
        S = np.outer(j, self.S)
        V = self.T + F + S
        return V

    def computeReciprocalSpaceGeometry(self):

        """ Compute the reciprocal-space scattering vectors, multiplied by wavelength."""

        self._K = self.V - self.B

    def deleteGeometryData(self):

        """ Clear out all derived geometry data."""

        for i in self._derivedGeometry:
            setattr(self, i, None)
        if self.panelList is not None:
            self.panelList.deleteGeometryData()
        self._validGeometry = False

    def getVertices(self, edge=False, loop=False):

        """ Get panel getVertices; positions of corner pixels."""

        nF = self.nF - 1
        nS = self.nS - 1
        z = 0

        if edge == True:
            z -= 0.5
            nF += 0.5
            nS += 0.5

        j = [z, nF, nF, z]
        i = [z, z, nS, nS]

        if loop == True:
            i.append(i[0])
            j.append(j[0])

        return self.pixelsToVectors(i, j)

    def getCenter(self):

        """ Vector to center of panel."""

        return np.mean(self.getVertices(), axis=0)

    def getRealSpaceBoundingBox(self):

        """ Return the minimum and maximum values of the four corners."""

        if self._rsbb is None:
            v = self.getVertices()
            r = np.zeros((2, 3))
            r[0, :] = np.min(v, axis=0)
            r[1, :] = np.max(v, axis=0)
            self._rsbb = r

        return self._rsbb.copy()

    def simpleSetup(self, nF=None, nS=None, pixSize=None, distance=None, T=None):

        """ Simple way to create a panel with centered beam."""

        if nF is None:
            raise ValueError("Number of fast scan pixels is unspecified.")
        if nS is None:
            raise ValueError("Number of slow scan pixels is unspecified.")
        if pixSize is None:
            raise ValueError("Pixel size is unspecified.")

        self.nF = nF
        self.nS = nS
        self.F = np.array([1, 0, 0]) * pixSize
        self.S = np.array([0, 1, 0]) * pixSize

        if T is None:
            if distance is None:
                raise ValueError("Distance is unspecified")
            self.T = np.array([0, 0, distance]) - self.F * self.nF / 2.0 - self.S * self.nS / 2.0
        else:
            self.T = T



class panelList(list):

    """ List container for detector panels, with extra functionality."""

    def __init__(self):

        """ Create an empty panel array."""

        # Configured data
        self.beam = None  # X-ray beam information, common to all panels
        # Derived data (concatenated from individual panels)
        self._data = None  # Concatenated intensity data
        self._pixSize = None  # Common pixel size (if not common amongst panels, this will be 'None')
        self._sa = None  # Concatenated solid angles
        self._V = None  # Concatenated pixel positions
        self._K = None  # Concatenated reciprocal-space vectors
        self._rsbb = None  # Real-space bounding box of entire panel list
        self._vll = None  # Look-up table for simple projection
        self._rpix = None  # junk
        self._derivedGeometry = ['_pixSize', '_V', '_sa', '_K', '_rsbb', '_vll', '_rpix']  # Default values of these are 'None'


    def copy(self, derived=True):

        """ Create a deep copy."""

        pa = panelList()
        for p in self:
            pa.append(p.copy(derived=derived))

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

        """ Set a panel, check that it is the appropriate type."""

        if not isinstance(value, panel):
            raise TypeError("You may only add panels to a panelList object")
        if value.name == "":
            value.name = "%d" % key
        super(panelList, self).__setitem__(key, value)

    @property
    def nPix(self):

        """ Total number of pixels in all panels."""

        ntot = 0
        for p in self:
            ntot += p.nPix
        return ntot

    @property
    def nPanels(self):

        """ Number of panels."""

        return len(self)

    def append(self, p=None, name=""):

        """ Append a panel, check that it is of the correct type."""

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

    def getPanelIndexByName(self, name):

        """ Find the integer index of a panel by it's unique name """

        i = 0
        for p in self:
            if p.name == name:
                return i
            i += 1
        return None

    def computeRealSpaceGeometry(self):

        """ Compute real-space geometry of all panels."""

        for p in self:
            p.computeRealSpaceGeometry()

    @property
    def V(self):

        """ Concatenated pixel position."""

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

        """ Concatenated reciprocal-space vectors."""

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

        """ Concatenated intensity data."""

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

        """ Check that intensity data is of the correct type."""

        if not isinstance(data, np.ndarray) and data.ndim == 1 and data.size == self.nPix:
            raise ValueError("Must be flattened ndarray of size %d" % self.nPix)
        n = 0
        for p in self:
            nPix = p.nPix
            nF = p.nF
            nS = p.nS
            p.data = data[n:(n + nPix)]
            p.data = p.data.reshape((p.nS, p.nF))
            n += nPix

    @property
    def wavelength(self):

        return self.beam.wavelength

    @wavelength.setter
    def wavelength(self, val):

        if self.beam is None:
            self.beam = source.beam()
        self.beam.wavelength = val
        for p in self:
            p.beam = self.beam

    @property
    def solidAngle(self):

        """ Concatenated solid angles."""

        if self._sa == None:
            sa = np.empty(self.nPix)
            n = 0
            for p in self:
                nPix = p.nPix
                nF = p.nF
                nS = p.nS
                sa[n:(n + nPix)] = p.solidAngle.ravel()
                n += nPix
            self._sa = sa
            p._sa = p._sa.reshape((p.nS, p.nF))
        return self._sa

    def assembledData(self):

        """ Project all intensity data along the beam direction.  Nearest neighbor interpolation."""

        if len(self) == 1:
            return self[0].data

        if self._vll is None:
            pixSize = self.pixSize
            r = self.realSpaceBoundingBox
            rpix = r / pixSize
            rpix[0, :] = np.floor(rpix[0, :])
            rpix[1, :] = np.ceil(rpix[1, :])
            V = self.V[:, 0:2] / pixSize - rpix[0, 0:2]
            Vll = np.round(V).astype(np.int32)
            self._vll = Vll
            self._rpix = rpix
        else:
            Vll = self._vll
            rpix = self._rpix

        adat = np.zeros([rpix[1, 1] - rpix[0, 1] + 1, rpix[1, 0] - rpix[0, 0] + 1])

        adat[Vll[:, 1], Vll[:, 0]] = self.data

        return adat

    @property
    def realSpaceBoundingBox(self):

        if self._rsbb is None:

            r = np.zeros([2, 3])
            v = np.zeros([2 * self.nPanels, 3])

            i = 0
            for p in self:
                v[i:(i + 2), :] = p.getRealSpaceBoundingBox()
                i += 2

            r[0, :] = np.min(v, axis=0)
            r[1, :] = np.max(v, axis=0)
            self._rsbb = r

        return self._rsbb.copy()

    @property
    def pixSize(self):

        """ Return the average pixel size, if all pixels are identical."""

        if self._pixSize is None:

            pix = np.zeros(self.nPanels)
            i = 0
            for p in self:
                pix[i] = p.pixSize
                i += 1
            mnpix = np.mean(pix)
            if all(np.absolute((pix - mnpix) / mnpix) < 1e-6):
                self._pixSize = mnpix
            else:
                raise ValueError("Pixel sizes in panel list are not all the same.")

        return self._pixSize.copy()

    def getCenter(self):

        """ Mean center of panel list."""

        c = np.zeros((self.nPanels, 3))
        i = 0
        for p in self:
            c[i, :] = p.getCenter()
            i += 1

        return np.mean(c, axis=0)

    def deleteGeometryData(self):

        """ Delete derived data relevant to geometry.  Normally used internally. """

        for i in self._derivedGeometry:
            setattr(self, i, None)

    def simpleSetup(self, nF=None, nS=None, pixSize=None, distance=None, T=None):

        """ Append a panel using the simple setup method."""

        p = panel()
        p.simpleSetup(nF, nS, pixSize, distance, T)
        self.append(p)
