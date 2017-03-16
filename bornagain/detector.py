"""
Classes for analyzing/simulating diffraction data contained in pixel array detectors (PADs).
"""

import numpy as np
from numpy.linalg import norm
from numpy.random import random, randn

from utils import vecNorm, vecMag
import source


class Panel(object):

    """ Individual detector Panel: a 2D lattice of square pixels."""

    def __init__(self, name=""):
        """ Make no assumptions during initialization."""

        # Configured parameters
        self._name = name  # Panel name for convenience
        self._F = None  # Fast-scan vector
        self._S = None  # Slow-scan vector
        # Translation of this Panel (from interaction region to center of first
        # pixel)
        self._T = None
        self._nF = 0  # Number of pixels along the fast-scan direction
        self._nS = 0  # Number of pixels along the slow-scan direction
        # Number of arbitrary data units per eV of photon energy
        self.adu_per_ev = 0
        self._beam = None  # Container for x-ray beam information
        self._data = None  # Diffraction intensity data
        self._mask = None
        self._dark = None

        # Cached parameters
        self._pixel_size = None  # Pixel size derived from F/S vectors
        # 3D vectors pointing from interaction region to pixel centers
        self._V = None
        self._sa = None  # Solid angles corresponding to each pixel
        self._pf = None  # Polarization factor
        self._K = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        # Hash of the configured geometry parameters
        self._geometry_hash = None

        # If this Panel is a part of a list
        self.PanelList = None  # This is the link to the Panel list

    def copy(self, derived=True):
        """ Deep copy of everything.  Parent Panel list is stripped away."""

        p = Panel()
        p.name = self.name
        p.F = self._F.copy()
        p.S = self._S.copy()
        p.T = self._T.copy()
        p.nF = self.nF
        p.nS = self.nS
        p.adu_per_ev = self.adu_per_ev
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
        p.PanelList = None

        return p

    def __str__(self):
        """ Print something useful when in interactive mode."""

        s = ""
        s += "name = \"%s\"\n" % self.name
        s += "pixel_size = %s\n" % self.pixel_size.__str__()
        s += "F = %s\n" % self.F.__str__()
        s += "S = %s\n" % self.S.__str__()
        s += "nF = %d\n" % self.nF
        s += "nS = %d\n" % self.nS
        s += "T = %s\n" % self.T.__str__()
        s += "adu_per_ev = %g\n" % self.adu_per_ev
        return s

    @property
    def name(self):
        """ Name of this Panel. """

        return self._name

    @name.setter
    def name(self, val):

        #        if self._name != "":
        #            raise ValueError('Cannot rename a detector Panel')

        self._name = val

    @property
    def data(self):
        """ Intensity data. """

        return self._data

    @data.setter
    def data(self, val):

        if val.shape[0] != self.nS or val.shape[1] != self.nF:
            raise ValueError('Panel data should have shape (%d,%d)' % (self.nS, self.nF))

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
    def dark(self):
        """ Dark signal. """

        if self._dark is None:
            self._dark = np.zeros(self.data.shape)
        return self._dark

    @dark.setter
    def dark(self, dark):

        self._dark = dark

    @property
    def beam(self):
        """ X-ray beam data, taken from parent Panel list if one exists."""

        if self.PanelList is not None:
            beam = self.PanelList._beam
            self._beam = None
        else:
            if self._beam is None:
                self._beam = source.beam()
            beam = self._beam

        return beam

    @beam.setter
    def beam(self, beam):

        if not isinstance(beam, source.beam):
            raise TypeError("Beam info must be a source.beam class")

        if self.PanelList is not None:
            self.PanelList._beam = beam
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
        self.clear_geometry_cache()
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
        self.clear_geometry_cache()
        if self.data is not None:
            if self.data.shape[0] != self._nS:
                self._data = None

    @property
    def pixel_size(self):
        """ Return the pixel size only if both fast/slow-scan vectors are the same length.
            Setting this value will modify the fast- and slow-scan vectors F and S. """

        if self._pixel_size is None:
            p1 = norm(self.F)
            p2 = norm(self.S)
            if abs(p1 - p2) / np.float(p2) > 1e-6 or abs(p1 - p2) / np.float(p1) > 1e-6:
                raise ValueError(
                    "Pixel size is not consistent between F and S vectors (%10f, %10f)." % (p1, p2))
            self._pixel_size = np.mean([p1, p2])
        return self._pixel_size.copy()

    @pixel_size.setter
    def pixel_size(self, val):

        val = val
        pf = norm(self.F)
        ps = norm(self.S)
        self._F *= val / pf
        self._S *= val / ps
        self.clear_geometry_cache()

    @property
    def n_pixels(self):
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
            self.clear_geometry_cache()
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
            self.clear_geometry_cache()
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
            self.clear_geometry_cache()
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
            raise ValueError(
                "No wavelength is defined.  Cannot compute Q vectors.")

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
        i = i.ravel() + random(self.n_pixels) - 0.5
        j = j.ravel() + random(self.n_pixels) - 0.5
        F = np.outer(i, self.F)
        S = np.outer(j, self.S)
        V = self.T + F + S
        B = np.outer(np.ones(self.n_pixels), self.B) + \
            randn(self.n_pixels, 3) * self.beam.divergence
        K = vecNorm(V) - vecNorm(B)
        lam = self.beam.wavelength * \
            (1 + randn(self.n_pixels) * self.beam.spectralWidth)
        return 2 * np.pi * K / np.outer(lam, np.ones(3))

    @property
    def stol(self):
        """ sin(theta)/lambda, where theta is the half angle """

        if self.beam.wavelength is None:
            raise ValueError("No wavelength is defined.  Cannot compute stol.")

        return 0.5 * vecMag(self.K) / self.beam.wavelength

    @property
    def N(self):
        """ Normal vector to Panel surface (F X S)."""

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
    def polarization_factor(self):
        """ The scattering polarization factor. """

        if self.beam.polarizationRatio != 1:
            raise ValueError("Only linear polarization handled at this time.")

        if self._pf is None:
            p = self.beam.P
            u = vecNorm(self.V)
            self._pf = 1.0 - np.abs(u.dot(p)) ** 2

        return self._pf

    def check(self):
        """ Check for any known issues with this Panel."""

        self.check_geometry()
        return True

    def check_geometry(self):
        """ Check for valid geometry configuration."""

        if self._T is None:
            raise ValueError("Panel translation vector T is not defined.")
        if self._F is None:
            raise ValueError("Panel basis vector F is not defined.")
        if self._S is None:
            raise ValueError("Panel basis vector S is not defined.")
        if self.nF == 0 or self.nS == 0:
            raise ValueError(
                "Data array has zero size (%d x %d)." % (self.nF, self.nS))

        return True

    @property
    def geometry_hash(self):
        """ Hash all of the configured geometry values. """

        if self._geometry_hash is None:

            F = self._F
            S = self._S
            T = self._T

            if F is None:
                self._geometry_hash = None
                return self._geometry_hash
            elif S is None:
                self._geometry_hash = None
                return self._geometry_hash
            elif T is None:
                self._geometry_hash = None
                return self._geometry_hash

            self._geometry_hash = hash(
                (F[0], F[1], F[2], S[0], S[1], S[2], T[0], T[1], T[2], self._nF, self._nS))

        return self._geometry_hash

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

    def clear_geometry_cache(self):
        """ Clear out all derived geometry data."""

        self._pixel_size = None  # Pixel size derived from F/S vectors
        # 3D vectors pointing from interaction region to pixel centers
        self._V = None
        self._sa = None  # Solid angles corresponding to each pixel
        self._K = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        # Hash of the configured geometry parameters
        self._geometry_hash = None

        if self.PanelList is not None:
            self.PanelList.clear_geometry_cache()

    def deleteDerivedData(self):

        if self.PanelList is not None:
            self.PanelList._data = None

    def get_vertices(self, edge=False, loop=False):
        """ Get Panel get_vertices; positions of corner pixels."""

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

    def get_center(self):
        """ Vector to center of Panel."""

        return np.mean(self.get_vertices(), axis=0)

    @property
    def realSpaceBoundingBox(self):

        return self.getRealSpaceBoundingBox()

    def getRealSpaceBoundingBox(self):
        """ Return the minimum and maximum values of the four corners."""

        if self._rsbb is None:
            v = self.get_vertices()
            r = np.zeros((2, 3))
            r[0, :] = np.min(v, axis=0)
            r[1, :] = np.max(v, axis=0)
            self._rsbb = r

        return self._rsbb.copy()

    def simple_setup(self, nF=None, nS=None, pixelSize=None, distance=None, wavelength=None, T=None):
        """ Simple way to create a Panel with centered beam."""

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
            self.T = np.array([0, 0, distance]) - self.F * \
                self.nF / 2.0 - self.S * self.nS / 2.0
        else:
            self.T = T

        if wavelength is not None:
            if self.beam is None:
                self.beam = source.beam()
            self.beam.wavelength = wavelength


class PanelList(object):

    """ List container for detector panels, with extra functionality."""

    def __init__(self):
        """ Create an empty Panel array."""

        # Configured data
        self._name = None  # The name of this list (useful for rigid groups)
        # X-ray beam information, common to all panels
        self._beam = None
        self._PanelList = []  # List of individual panels

        # Derived data (concatenated from individual panels)
        self._data = None  # Concatenated intensity data
        self._mask = None  # Identify bad pixels [1 means good, 0 means bad]
        self._dark = None  # Counts on detector when there are no x-rays
        # Common pixel size (if not common amongst panels, this will be 'None')
        self._pixel_size = None
        self._sa = None  # Concatenated solid angles
        self._V = None  # Concatenated pixel positions
        self._K = None  # Concatenated reciprocal-space vectors
        self._rsbb = None  # Real-space bounding box of entire Panel list
        self._vll = None  # Look-up table for simple projection
        self._rpix = None  # junk
        self._pf = None  # Polarization facto
        self._geometry_hash = None  # Hash of geometries
        # Groups of panels that might be manipulated together
        self._rigidGroups = None
        self._derivedGeometry = ['_pixel_size', '_V', '_sa', '_K', '_pf', '_rsbb',
                                 '_vll', '_rpix', '_geometry_hash']  # Default values of these are 'None'

    def copy(self, derived=True):
        """ Create a deep copy."""

        pa = PanelList()
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
        """ Get a Panel. Panels may be referenced by name or index."""

        if isinstance(key, str):
            key = self.getPanelIndexByName(key)
            if key is None:
                raise IndexError("There is no Panel named %s" % key)
                return None
        return self._PanelList[key]

    def __setitem__(self, key, value):
        """ Set a Panel, check that it is the appropriate type."""

        if not isinstance(value, Panel):
            raise TypeError("You may only add panels to a PanelList object")
        if value.name == "":
            # Give the  Panel a name if it wasn't provided
            value.name = "%d" % key
        self._PanelList[key] = value

    def __iter__(self):
        """ Iterate through panels. """

        return iter(self._PanelList)

    def __len__(self):
        """ Return length of Panel list."""

        return len(self._PanelList)

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
            p._beam = None

    @property
    def n_pixels(self):
        """ Total number of pixels in all panels."""

        return np.sum([p.n_pixels for p in self])

    @property
    def nPanels(self):
        """ Number of panels."""

        return len(self._PanelList)

    def append(self, p=None, name=""):
        """ Append a Panel, check that it is of the correct type."""

        if p is None:
            p = Panel()
        if not isinstance(p, Panel):
            raise TypeError("You may only append panels to a PanelList object")
        p.PanelList = self

        # Create name if one doesn't exist
        if name != "":
            p.name = name
        elif p.name == "":
            p.name = "%d" % self.nPanels

        # Inherit beam from first append
        if self._beam is None:
            self._beam = p._beam

        self._PanelList.append(p)

    def getPanelIndexByName(self, name):
        """ Find the integer index of a Panel by it's unique name """

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

            self._V = np.concatenate(
                [p.V.reshape(np.product(p.V.shape) / 3, 3) for p in self])

        return self._V

    @property
    def K(self):
        """ Concatenated reciprocal-space vectors."""

        if self._K is None or self._K.shape[0] != self.n_pixels:

            self._K = np.concatenate(
                [p.K.reshape(np.product(p.K.shape) / 3, 3) for p in self])

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

            self._data = np.concatenate(
                [p.data.reshape(np.product(p.data.shape)) for p in self])

        return self._data

    @data.setter
    def data(self, data):
        """ Check that intensity data is of the correct type."""

        if not isinstance(data, np.ndarray) and data.ndim == 1 and data.size == self.n_pixels:
            raise ValueError(
                "Must be flattened ndarray of size %d" % self.n_pixels)

        self._data = data
        n = 0
        for p in self:
            nPix = p.n_pixels
            p.data = data[n:(n + nPix)].reshape((p.nS, p.nF))
            n += nPix

    @property
    def mask(self):
        """ Concatenated mask."""

        if self._mask is None:

            self._mask = np.concatenate(
                [p.mask.reshape(np.product(p.mask.shape)) for p in self])

        return self._mask

    @mask.setter
    def mask(self, mask):
        """ Check that mask is of the correct type."""

        if not isinstance(mask, np.ndarray) and mask.ndim == 1 and mask.size == self.n_pixels:
            raise ValueError(
                "Must be flattened ndarray of size %d" % self.n_pixels)

        self._mask = mask
        n = 0
        for p in self:
            nPix = p.n_pixels
            p.mask = mask[n:(n + nPix)]
            p.mask = p.mask.reshape((p.nS, p.nF))
            n += nPix

    @property
    def dark(self):
        """ Concatenated dark."""

        if self._dark is None:

            self._dark = np.concatenate(
                [p.dark.reshape(np.product(p.dark.shape)) for p in self])

        return self._dark

    @dark.setter
    def dark(self, dark):
        """ Check that dark is of the correct type."""

        if not isinstance(dark, np.ndarray) and dark.ndim == 1 and dark.size == self.n_pixels:
            raise ValueError(
                "Must be flattened ndarray of size %d" % self.n_pixels)

        self._dark = dark
        n = 0
        for p in self:
            nPix = p.n_pixels
            p.dark = dark[n:(n + nPix)]
            p.dark = p.dark.reshape((p.nS, p.nF))
            n += nPix

    @property
    def wavelength(self):

        return self.beam.wavelength

    @property
    def name(self):
        """ The name of this Panel list """

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
         a truncated Panel list pointing to members of the parent Panel list.  """

        if self._rigidGroups is None:
            self._rigidGroups = []

        pl = PanelList()
        pl.beam = self.beam
        pl.name = name
        for val in vals:
            pl.append(self[val])
            pl[-1].PanelList = self
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

            self._sa = np.concatenate(
                [p.solidAngle.reshape(np.product(p.solidAngle.shape)) for p in self])

        return self._sa

    @property
    def polarization_factor(self):
        """ Concatenated polarization factors."""

        if self._pf == None:

            self._pf = np.concatenate([p.polarization_factor.reshape(
                np.product(p.polarization_factor.shape)) for p in self])

        return self._pf

    def assembleData(self, data=None):
        """ Project all intensity data along the beam direction.  Nearest neighbor interpolation.  This is a crude way to display data... """

        if self._vll is None:
            pixelSize = self.pixel_size
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

        adat = np.zeros(
            [rpix[1, 1] - rpix[0, 1] + 1, rpix[1, 0] - rpix[0, 0] + 1])

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
            pixelSize = self.pixel_size
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

        adat = np.zeros(
            [rpix[1, 1] - rpix[0, 1] + 1, rpix[1, 0] - rpix[0, 0] + 1])

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
    def pixel_size(self):
        """ Return the average pixel size, if all pixels are identical."""

        if self._pixel_size is None:

            pix = np.zeros(self.nPanels)
            i = 0
            for p in self:
                pix[i] = p.pixel_size
                i += 1
            mnpix = np.mean(pix)
            if all(np.absolute((pix - mnpix) / mnpix) < 1e-6):
                self._pixel_size = mnpix
            else:
                raise ValueError(
                    "Pixel sizes in Panel list are not all the same.")

        return self._pixel_size.copy()

    def get_center(self):
        """ Mean center of Panel list. """

        c = np.zeros((self.nPanels, 3))
        i = 0
        for p in self:
            c[i, :] = p.get_center()
            i += 1

        return np.mean(c, axis=0)

    def check_geometry(self):
        """ Check that geometry is sane. """

        for p in self:

            if not p.check_geometry():

                return False

        return True

    def clear_geometry_cache(self):
        """ Delete derived data relevant to geometry.  Normally used internally. """

        self._pixel_size = None  # Pixel size derived from F/S vectors
        # 3D vectors pointing from interaction region to pixel centers
        self._V = None
        self._sa = None  # Solid angles corresponding to each pixel
        self._K = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        # Hash of the configured geometry parameters
        self._geometry_hash = None

    @property
    def geometry_hash(self):
        """ Hash all of the configured geometry values. """

        if self._geometry_hash is None:

            a = tuple([p.geometry_hash for p in self])

            if None in a:
                self._geometry_hash = None
                return self._geometry_hash

            self._geometry_hash = hash(a)

        return self._geometry_hash

    def simple_setup(self, nF=None, nS=None, pixelSize=None, distance=None, wavelength=None, T=None):
        """

        Append a Panel using the simple setup method.

        Arguments:
        nF         : Number of fast-scan pixels
        nS         : Number of slow-scan pixels
        pixel_size  : Pixel size in meters
        distance   : Distance in meters
        wavelength : Wavelength in meters
        T          : Translation from sample to center of first pixel in meters

        """

        p = Panel()
        p.simple_setup(nF, nS, pixelSize, distance, wavelength, T)
        self.beam = p.beam
        self.append(p)

    def RadialProfile(self):

        q = self.Q / 2 / np.pi
        qmag = vecMag(q)
        maxq = np.max(qmag)

        nBins = 100
        binSize = maxq / (nBins - 1)
        bins = (np.arange(0, nBins) + 0.5) * binSize
        binIndices = np.int64(np.floor(qmag / binSize))

        c = np.bincount(binIndices, self.mask, nBins)
        c = c.astype(np.double)
        c[c == 0] = 1e100
        c[np.isnan(c)] = 1e100
        pr = np.bincount(binIndices, self.data * self.mask, nBins)
        pr /= c

        return pr, bins
