import numpy as np
from numpy.linalg import norm
from numpy.random import random, randn
from bornagain.utils import vecNorm, vecMag
from bornagain import source

"""
Classes for analyzing diffraction data contained in pixel array detectors (PADs).
"""

class panel(object):

    """ Individual detector panel: a 2D lattice of square pixels."""

    def __init__(self, name=""):

        """ Make no assumptions during initialization."""

        # Configured parameters
        self._name = name  # Panel name for convenience
        self._F = None  # Fast-scan vector
        self._S = None  # Slow-scan vector
        self._T = None  # Translation of this panel (from interaction region to center of first pixel)
        self._nF = 0  # Number of pixels along the fast-scan direction
        self._nS = 0  # Number of pixels along the slow-scan direction
        self.aduPerEv = 0  # Number of arbitrary data units per eV of photon energy
        self._beam = source.beam()  # Container for x-ray beam information
        self._data = None  # Diffraction intensity data
        self._mask = None

        # Cached parameters
        self._pixelSize = None  # Pixel size derived from F/S vectors
        self._V = None  # 3D vectors pointing from interaction region to pixel centers
        self._sa = None  # Solid angles corresponding to each pixel
        self._pf = None # Polarization factor
        self._K = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        self._geometryHash = None  # Hash of the configured geometry parameters

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
            if self._pf is not None:
                p._pf = self._pf.copy()
            if self._rsbb is not None:
                p._rsbb = self._rsbb.copy()
        p.panelList = None

        return p

    def __str__(self):

        """ Print something useful when in interactive mode."""

        s = ""
        s += "name = \"%s\"\n" % self.name
        s += "pixelSize = %s\n" % self.pixelSize.__str__()
        s += "F = %s\n" % self.F.__str__()
        s += "S = %s\n" % self.S.__str__()
        s += "nF = %d\n" % self.nF
        s += "nS = %d\n" % self.nS
        s += "T = %s\n" % self.T.__str__()
        s += "aduPerEv = %g\n" % self.aduPerEv
        return s

    @property
    def name(self):

        """ Name of this panel. """

        return self._name

    @name.setter
    def name(self, val):

#        if self._name != "":
#            raise ValueError('Cannot rename a detector panel')

        self._name = val

    @property
    def data(self):

        """ Intensity data. """

        return self._data

    @data.setter
    def data(self, val):
       
        if val.shape[0] != self.nS or val.shape[1] != self.nF:
            raise ValueError('Panel data should have shape (%d,%d), not (%d,%d)' % (self.nS,self.nF,val.shape[0],val.shape[1]))
                     
        self._data = val
        # Must clear out any derived data that depends on this input
        self.deleteDerivedData()

    @property
    def mask(self):

        """ Bad pixel mask. """

        if self._mask is None:
            self._mask = np.ones(self.data.shape)
        return self._mask

    @mask.setter
    def mask(self, mask):

        self._mask = mask

    @property
    def beam(self):

        """ X-ray beam data, taken from parent panel list if one exists."""

        if self.panelList is not None:
            beam = self.panelList._beam
            self._beam = None
        else:
            beam = self._beam

        return beam

    @beam.setter
    def beam(self, beam):

        if not isinstance(beam, source.beam):
            raise TypeError("Beam info must be a source.beam class")

        if self.panelList is not None:
            self.panelList._beam = beam
            self._beam = None
        else:
            self._beam = beam

    @property
    def nF(self):

        """ Number of fast-scan pixels."""

        if self._nF is 0:
            if self.data is not None:
                self.nF = self.data.shape[1]
        return self._nF

    @nF.setter
    def nF(self, val):

        """ Changing the fast-scan pixel count destroys all derived geometry data, and 
        any unmatched intensity data."""

        self._nF = np.int(val)
        self.clearGeometryCache()
        if self.data is not None:
            if self.data.shape[1] != self._nF:
                self._data = None

    @property
    def nS(self):

        """ Number of slow-scan pixels."""

        if self._nS is 0:
            if self.data is not None:
                self._nS = self.data.shape[0]
        return self._nS

    @nS.setter
    def nS(self, val):

        """ Changing the fast-scan pixel count destroys all derived geometry data, and 
        any unmatched intensity data."""

        self._nS = np.int(val)
        self.clearGeometryCache()
        if self.data is not None:
            if self.data.shape[0] != self._nS:
                self._data = None

    @property
    def pixelSize(self):

        """ Return the pixel size only if both fast/slow-scan vectors are the same length.
            Setting this value will modify the fast- and slow-scan vectors F and S. """

        if self._pixelSize is None:
            p1 = norm(self.F)
            p2 = norm(self.S)
            if abs(p1 - p2) / np.float(p2) > 1e-6 or abs(p1 - p2) / np.float(p1) > 1e-6:
                raise ValueError("Pixel size is not consistent between F and S vectors (%10f, %10f)." % (p1, p2))
            self._pixelSize = np.mean([p1, p2])
        return self._pixelSize.copy()

    @pixelSize.setter
    def pixelSize(self, val):

        val = val
        pf = norm(self.F)
        ps = norm(self.S)
        self._F *= val / pf
        self._S *= val / ps
        self.clearGeometryCache()

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
            self._F = val
            self.clearGeometryCache()
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
            self._S = val
            self.clearGeometryCache()
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
            self._T = val
            self.clearGeometryCache()
        else:
            raise ValueError("Must be a numpy array of length 3.")

    @property
    def B(self):

        """ Nominal beam direction vector (normalized)."""

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

        """ Normalized scattering vectors.   Similar to the conventional scattering 
            vectors, but multiplied by wavelength.  This does not have the 2*pi factor
            included."""

        if self._K is None:
            self.computeReciprocalSpaceGeometry()
        return self._K

    @property
    def Q(self):

        """ Scattering vectors, with 2*pi factor and wavelength included."""

        if self.beam.wavelength is None:
            raise ValueError("No wavelength is defined.  Cannot compute Q vectors.")

        return 2.0 * np.pi * self.K / self.beam.wavelength

    @property
    def Qmag(self):

        """ Scattering vector magnitudes, with 2*pi factor and wavelength included."""

        return vecMag(self.Q)

    @property
    def mcQ(self):

        """ Monte Carlo q vectors; add jitter to wavelength, pixel position, incident 
            beam direction for each pixel independently. """

        i = np.arange(self.nF)
        j = np.arange(self.nS)
        [i, j] = np.meshgrid(i, j)
        i = i.ravel() + random(self.nPix) - 0.5
        j = j.ravel() + random(self.nPix) - 0.5
        F = np.outer(i, self.F)
        S = np.outer(j, self.S)
        V = self.T + F + S
        B = np.outer(np.ones(self.nPix), self.B) + randn(self.nPix, 3) * self.beam.divergence
        K = vecNorm(V) - vecNorm(B)
        lam = self.beam.wavelength * (1 + randn(self.nPix) * self.beam.spectralWidth)
        return 2 * np.pi * K / np.outer(lam, np.ones(3))

    @property
    def stol(self):

        """ sin(theta)/lambda, where theta is the half angle """

        if self.beam.wavelength is None:
            raise ValueError("No wavelength is defined.  Cannot compute stol.")

        return 0.5 * vecMag(self.K) / self.beam.wavelength


    @property
    def N(self):

        """ Normal vector to panel surface (F X S)."""

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

    @property
    def polarizationFactor(self):

        """ The scattering polarization factor. """

        if self.beam.polarizationRatio != 1:
            raise ValueError("Only linear polarization handled at this time.")

        if self._pf is None:
            p = self.beam.P
            u = vecNorm(self.V)
            self._pf = 1.0 - np.abs(u.dot(p)) ** 2

        return self._pf

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

        return True

    @property
    def geometryHash(self):

        """ Hash all of the configured geometry values. """

        if self._geometryHash is None:

            F = self._F
            S = self._S
            T = self._T

            if F is None:
                self._geometryHash = None
                return self._geometryHash
            elif S is None:
                self._geometryHash = None
                return self._geometryHash
            elif T is None:
                self._geometryHash = None
                return self._geometryHash

            self._geometryHash = hash((F[0], F[1], F[2], S[0], S[1], S[2], T[0], T[1], T[2], self._nF, self._nS))

        return self._geometryHash

    def computeRealSpaceGeometry(self):

        """ Compute arrays relevant to real-space geometry."""

        i = np.arange(self.nF)
        j = np.arange(self.nS)
        [i, j] = np.meshgrid(i, j)
        i.ravel()
        j.ravel()
        self._V = self.pixelsToVectors(j, i)

    def pixelsToVectors(self, j, i):

        """ Convert pixel indices to translation vectors (i=fast scan, j=slow scan)."""

        F = np.outer(i, self.F)
        S = np.outer(j, self.S)
        V = self.T + F + S
        return V

    def computeReciprocalSpaceGeometry(self):

        """ Compute the reciprocal-space scattering vectors, multiplied by wavelength."""

        self._K = vecNorm(self.V) - self.B

    def clearGeometryCache(self):

        """ Clear out all derived geometry data."""

        self._pixelSize = None  # Pixel size derived from F/S vectors
        self._V = None  # 3D vectors pointing from interaction region to pixel centers
        self._sa = None  # Solid angles corresponding to each pixel
        self._K = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        self._geometryHash = None  # Hash of the configured geometry parameters           
            
        if self.panelList is not None:
            self.panelList.clearGeometryCache()

    def deleteDerivedData(self):

        if self.panelList is not None:
            self.panelList._data = None

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

    @property
    def realSpaceBoundingBox(self):

        return self.getRealSpaceBoundingBox()

    def getRealSpaceBoundingBox(self):

        """ Return the minimum and maximum values of the four corners."""

        if self._rsbb is None:
            v = self.getVertices()
            r = np.zeros((2, 3))
            r[0, :] = np.min(v, axis=0)
            r[1, :] = np.max(v, axis=0)
            self._rsbb = r

        return self._rsbb.copy()

    def simpleSetup(self, nF=None, nS=None, pixelSize=None, distance=None, wavelength=None, T=None):

        """ Simple way to create a panel with centered beam."""

        if nF is None:
            raise ValueError("Number of fast scan pixels is unspecified.")
        if nS is None:
            raise ValueError("Number of slow scan pixels is unspecified.")
        if pixelSize is None:
            raise ValueError("Pixel size is unspecified.")

        self.nF = nF
        self.nS = nS
        self.F = np.array([1, 0, 0]) * pixelSize
        self.S = np.array([0, 1, 0]) * pixelSize

        if T is None:
            if distance is None:
                raise ValueError("Distance is unspecified")
            self.T = np.array([0, 0, distance]) - self.F * self.nF / 2.0 - self.S * self.nS / 2.0
        else:
            self.T = T

        if wavelength is not None:
            if self.beam is None:
                self.beam = source.beam()
            self.beam.wavelength = wavelength


class panelList(object):

    """ List container for detector panels, with extra functionality."""

    def __init__(self):

        """ Create an empty panel array."""

        # Configured data
        self._name = None  # The name of this list (useful for rigid groups)
        self._beam = source.beam()  # X-ray beam information, common to all panels
        self._panelList = []  # List of individual panels
        
        # Derived data (concatenated from individual panels)
        self._data = None  # Concatenated intensity data
        self._mask = None  # Identify bad pixels [1 means good, 0 means bad]
        self._pixelSize = None  # Common pixel size (if not common amongst panels, this will be 'None')
        self._sa = None  # Concatenated solid angles
        self._V = None  # Concatenated pixel positions
        self._K = None  # Concatenated reciprocal-space vectors
        self._rsbb = None  # Real-space bounding box of entire panel list
        self._vll = None  # Look-up table for simple projection
        self._rpix = None  # junk
        self._pf = None # Polarization facto
        self._geometryHash = None  # Hash of geometries
        self._rigidGroups = None  # Groups of panels that might be manipulated together
        self._derivedGeometry = ['_pixelSize', '_V', '_sa', '_K', '_pf', '_rsbb', '_vll', '_rpix', '_geometryHash']  # Default values of these are 'None'


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
        return self._panelList[key]

    def __setitem__(self, key, value):

        """ Set a panel, check that it is the appropriate type."""

        if not isinstance(value, panel):
            raise TypeError("You may only add panels to a panelList object")
        if value.name == "":
            value.name = "%d" % key  # Give the  panel a name if it wasn't provided
        self._panelList[key] = value

    def __iter__(self):
        
        """ Iterate through panels. """
        
        return iter(self._panelList)

    def __len__(self):
        
        """ Return length of panel list."""
        
        return len(self._panelList)

    @property
    def beam(self):

        """ X-ray beam data. """

        if self._beam is None:
            self._beam = source.beam()

        return self._beam

    @beam.setter
    def beam(self, beam):

        """ X-ray beam data setter. """

        if not isinstance(beam, source.beam):
            raise TypeError("Beam info must be a source.beam class")

        self._beam = beam
        for p in self:
            p.beam = beam
        

    @property
    def nPix(self):

        """ Total number of pixels in all panels."""

        return np.sum([p.nPix for p in self])

    @property
    def nPanels(self):

        """ Number of panels."""

        return len(self._panelList)

    def append(self, p=None, name=""):

        """ Append a panel, check that it is of the correct type."""

        if p is None:
            p = panel()
        if not isinstance(p, panel):
            raise TypeError("You may only append panels to a panelList object")
        p.panelList = self

        # Create name if one doesn't exist
        if name != "":
            p.name = name
        elif p.name == "":
            p.name = "%d" % self.nPanels

        # Inherit first beam from append
        if self._beam is None:
            self._beam = p.beam.copy()

        p._beam = None

        self._panelList.append(p)

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

            self._V = np.concatenate([p.V.reshape(np.product(p.V.shape) / 3, 3) for p in self])

        return self._V

    @property
    def K(self):

        """ Concatenated reciprocal-space vectors."""

        if self._K is None or self._K.shape[0] != self.nPix:

            self._K = np.concatenate([p.K.reshape(np.product(p.K.shape) / 3, 3) for p in self])

        return self._K

    @property
    def Q(self):

        """ Concatenated reciprocal-space vectors."""

        return 2 * np.pi * self.K / self.beam.wavelength

    @property
    def Qmag(self):

        """ Concatentated reciprocal-space vector magnitudes."""

        return vecMag(self.Q)

    @property
    def mcQ(self):

        """ Monte Carlo q vectors; add jitter to wavelength, pixel position, incident 
            beam direction for each pixel independently. """

        return np.concatenate([p.mcQ.reshape(np.product(p.mcQ.shape) / 3, 3) for p in self])

    @property
    def stol(self):

        """ sin(theta)/lambda, where theta is the half angle """

        if self.beam.wavelength is None:

            raise ValueError("No wavelength is defined.  Cannot compute stol.")

        return 0.5 * vecMag(self.K) / self.wavelength

    @property
    def data(self):

        """ Concatenated intensity data."""

        if self._data is None:

            self._data = np.concatenate([p.data.reshape(np.product(p.data.shape)) for p in self])

        return self._data

    @data.setter
    def data(self, data):

        """ Check that intensity data is of the correct type."""

        if not isinstance(data, np.ndarray) and data.ndim == 1 and data.size == self.nPix:
            raise ValueError("Must be flattened ndarray of size %d" % self.nPix)

        self._data = data
        n = 0
        for p in self:
            nPix = p.nPix
            p.data = data[n:(n + nPix)]
            p.data = p.data.reshape((p.nS, p.nF))
            n += nPix

    @property
    def mask(self):

        """ Concatenated mask."""

        if self._mask is None:

            self._mask = np.concatenate([p.mask.reshape(np.product(p.mask.shape)) for p in self])

        return self._mask

    @mask.setter
    def mask(self, mask):

        """ Check that mask is of the correct type."""

        if not isinstance(mask, np.ndarray) and mask.ndim == 1 and mask.size == self.nPix:
            raise ValueError("Must be flattened ndarray of size %d" % self.nPix)

        self._mask = mask
        n = 0
        for p in self:
            nPix = p.nPix
            p.mask = mask[n:(n + nPix)]
            p.mask = p.mask.reshape((p.nS, p.nF))
            n += nPix

    @property
    def wavelength(self):

        return self.beam.wavelength

    @property
    def name(self):

        """ The name of this panel list """

        return self._name

    @name.setter
    def name(self, name):

        self._name = name

    @property
    def rigidGroups(self):

        """ Groups of panels that one might like to manipulate together. """

        return self._rigidGroups

    def addRigidGroup(self, name, vals):

        """ Append to the list of rigid groups. A rigid group is techically just
         a truncated panel list pointing to members of the parent panel list.  """

        if self._rigidGroups is None:
            self._rigidGroups = []

        pl = panelList()
        pl.beam = self.beam
        pl.name = name
        for val in vals:
            pl.append(self[val])
            pl[-1].panelList = self
        self._rigidGroups.append(pl)

    def appendToRigidGroup(self, name, vals):

        """ Append more panels to an existing rigid group. """

        g = self.rigidGroup(name)
        for val in vals:
            g.append(val)

    def rigidGroup(self, name):

        """ Get a rigid group by name. """

        thisGroup = None
        for g in self.rigidGroups:
            if g.name == name:
                thisGroup = g
                break

        return thisGroup


    @wavelength.setter
    def wavelength(self, val):

        if self.beam is None:
            self.beam = source.beam()
        self.beam.wavelength = val

    @property
    def solidAngle(self):

        """ Concatenated solid angles."""

        if self._sa == None:

            self._sa = np.concatenate([p.solidAngle.reshape(np.product(p.solidAngle.shape)) for p in self])

        return self._sa

    @property
    def polarizationFactor(self):

        """ Concatenated polarization factors."""

        if self._pf == None:

            self._pf = np.concatenate([p.polarizationFactor.reshape(np.product(p.polarizationFactor.shape)) for p in self])

        return self._pf


    def assembleData(self, data=None):

        """ Project all intensity data along the beam direction.  Nearest neighbor interpolation.  This is a crude way to display data... """

        if self._vll is None:
            pixelSize = self.pixelSize
            r = self.realSpaceBoundingBox
            rpix = r / pixelSize
            rpix[0, :] = np.floor(rpix[0, :])
            rpix[1, :] = np.ceil(rpix[1, :])
            V = self.V[:, 0:2] / pixelSize - rpix[0, 0:2]
            Vll = np.round(V).astype(np.int32)
            self._vll = Vll.astype(np.int)
            self._rpix = rpix.astype(np.int)
            
        Vll = self._vll
        rpix = self._rpix

        adat = np.zeros([rpix[1, 1] - rpix[0, 1] + 1, rpix[1, 0] - rpix[0, 0] + 1])

        if data is None:
            data = self.data

        adat[Vll[:, 1], Vll[:, 0]] = data

        return adat

    @property
    def assembledData(self):

        """ Project all intensity data along the beam direction.  Nearest neighbor interpolation.  This is a crude way to display data... """

        if len(self) == 1:
            return self[0].data

        if self._vll is None:
            pixelSize = self.pixelSize
            r = self.realSpaceBoundingBox
            rpix = r / pixelSize
            rpix[0, :] = np.floor(rpix[0, :])
            rpix[1, :] = np.ceil(rpix[1, :])
            V = self.V[:, 0:2] / pixelSize - rpix[0, 0:2]
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

        """ This returns the coordinates of a box just big enough to fit all of the panels.  Useful for display purposes. """

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
    def pixelSize(self):

        """ Return the average pixel size, if all pixels are identical."""

        if self._pixelSize is None:

            pix = np.zeros(self.nPanels)
            i = 0
            for p in self:
                pix[i] = p.pixelSize
                i += 1
            mnpix = np.mean(pix)
            if all(np.absolute((pix - mnpix) / mnpix) < 1e-6):
                self._pixelSize = mnpix
            else:
                raise ValueError("Pixel sizes in panel list are not all the same.")

        return self._pixelSize.copy()

    def getCenter(self):

        """ Mean center of panel list. """

        c = np.zeros((self.nPanels, 3))
        i = 0
        for p in self:
            c[i, :] = p.getCenter()
            i += 1

        return np.mean(c, axis=0)

    def checkGeometry(self):

        """ Check that geometry is sane. """

        for p in self:

            if not p.checkGeometry():

                return False
        
        return True

    def clearGeometryCache(self):

        """ Delete derived data relevant to geometry.  Normally used internally. """

        self._pixelSize = None  # Pixel size derived from F/S vectors
        self._V = None  # 3D vectors pointing from interaction region to pixel centers
        self._sa = None  # Solid angles corresponding to each pixel
        self._K = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        self._geometryHash = None  # Hash of the configured geometry parameters

    @property
    def geometryHash(self):

        """ Hash all of the configured geometry values. """

        if self._geometryHash is None:

            a = tuple([p.geometryHash for p in self])

            if None in a:
                self._geometryHash = None
                return self._geometryHash

            self._geometryHash = hash(a)

        return self._geometryHash

    def simpleSetup(self, nF=None, nS=None, pixelSize=None, distance=None, wavelength=None, T=None):

        """ 

        Append a panel using the simple setup method.
        
        Arguments:
        nF         : Number of fast-scan pixels
        nS         : Number of slow-scan pixels
        pixelSize  : Pixel size in meters
        distance   : Distance in meters
        wavelength : Wavelength in meters
        T          : Translation from sample to center of first pixel in meters
        
        """

        p = panel()
        p.simpleSetup(nF, nS, pixelSize, distance, wavelength, T)
        self.beam = p.beam
        self.append(p)

    def radialProfile(self):

        q = self.Q / 2 / np.pi
        qmag = vecMag(q)
        maxq = np.max(qmag)

        nBins = 100
        binSize = maxq / (nBins - 1)
        bins = (np.arange(0,nBins) + 0.5)*binSize
        binIndices = np.int64(np.floor(qmag / binSize))

        c = np.bincount(binIndices, self.mask, nBins)
        c = c.astype(np.double)
        c[c == 0] = 1e100
        c[np.isnan(c)] = 1e100
        pr = np.bincount(binIndices, self.data*self.mask, nBins)
        pr /= c

        return pr, bins
