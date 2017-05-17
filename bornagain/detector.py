"""
Classes for analyzing/simulating diffraction data contained in pixel array
detectors (PADs).
"""
import sys

import numpy as np
from numpy.linalg import norm
try:
    import matplotlib
    import pylab as plt
except ImportError:
    pass
# from numpy.random import random, randn

from utils import vec_norm, vec_mag, vec_check
import source
import units


class Panel(object):
    """ Individual detector Panel: a 2D lattice of square pixels."""

    def __init__(self, name=""):
        """ Make no assumptions during initialization."""

        # Configured parameters
        self.name = name  # Panel name for convenience
        self._F = None  # Fast-scan vector
        self._S = None  # Slow-scan vector
        self._T = None  # Translation to center of first pixel
        self._nF = 0  # Number of pixels along the fast-scan direction
        self._nS = 0  # Number of pixels along the slow-scan direction
        self.adu_per_ev = 0  # Arbitrary data units per eV of photon energy
        self.beam = source.Beam()  # Placeholder for Beam object

        # Cached parameters
        self._ps = None  # Pixel size derived from F/S vectors
        self._v = None  # Vectors to pixel centers
        self._sa = None  # Solid angles corresponding to each pixel
        self._pf = None  # Polarization factor
        self._k = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        self._gh = None  # Hash of the configured geometry parameters

        # If this Panel is a part of a list
        self.panellist = None  # This is the link to the Panel list

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

    def reshape(self, data):
        """ Reshape array to panel shape. """

        return data.reshape(self.nS, self.nF)

    def ones(self):
        """ Return array of ones with shape (nS,nF). """

        return np.ones((self.nS, self.nF))

    def zeros(self):
        """ Return array of zeros with shape (nS,nF). """

        return np.zeros((self.nS, self.nF))

    def check_data(self, data):
        """ Check that a data array is consistent with panel shape."""

        try:
            d = data.reshape(self.nS, self.nF)
        except:
            raise ValueError('Data array is not of correct type.  Must be'
                             ' a numpy array of shape %dx%d or length %d'
                             % (self.nS, self.nF, self.nS * self.nF))
        return d

    @property
    def nF(self):
        """ Number of fast-scan pixels."""

        return self._nF

    @nF.setter
    def nF(self, val):
        """ Destroy geometry cache."""

        self._nF = np.int(val)
        self.clear_geometry_cache()

    @property
    def nS(self):
        """ Number of slow-scan pixels."""

        return self._nS

    @nS.setter
    def nS(self, val):
        """ Destroy geometry cache."""

        self._nS = np.int(val)
        self.clear_geometry_cache()

    @property
    def pixel_size(self):
        """ Return the pixel size only if both fast/slow-scan vectors are the
        same length. Setting this value will modify the fast- and slow-scan
        vectors F and S. """

        if self._ps is None:
            p1 = norm(self.F)
            p2 = norm(self.S)
            if abs(p1 - p2) / np.float(p2) > 1e-6 or abs(p1 - p2) / np.float(
                    p1) > 1e-6:
                raise ValueError(
                    """Pixel size is not consistent between F and S vectors
                    (%10f, %10f)."""
                    % (p1, p2))
            self._ps = np.mean([p1, p2])
        return self._ps.copy()

    @pixel_size.setter
    def pixel_size(self, val):
        """ Destroy geometry cache."""

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
    def F(self, F):
        """ Must be a numpy ndarray of length 3."""

        self._F = vec_check(F)
        self.clear_geometry_cache()

    @property
    def S(self):
        """ Slow-scan vector (length equal to pixel size)."""

        return self._S

    @S.setter
    def S(self, S):
        """ Must be a numpy ndarray of length 3."""

        self._S = vec_check(S)
        self.clear_geometry_cache()

    @property
    def T(self):
        """ Translation vector pointing from interaction region to center of
        first pixel in memory."""

        return self._T

    @T.setter
    def T(self, T):
        """ Must be an ndarray of length 3."""

        self._T = vec_check(T)
        self.clear_geometry_cache()

    @property
    def B(self):
        """ Nominal Beam direction vector (normalized)."""

        if self.beam is None:
            raise ValueError("Panel has no Beam information.")
        return self.beam.B

    @property
    def V(self):
        """ Vectors pointing from interaction region to pixel centers."""

        if self._v is None:
            i = np.arange(self.nF)
            j = np.arange(self.nS)
            [i, j] = np.meshgrid(i, j)
            i.ravel()
            j.ravel()
            self._v = self.pixels_to_vectors(j, i)
        
        return self._v

    @property
    def K(self):
        """ Normalized scattering vectors.   Similar to the conventional
        scattering vectors, but multiplied by wavelength.  This does not have
        the 2*pi factor included."""

        if self._k is None:
            self.compute_reciprocal_space_geometry()
        return self._k

    @property
    def Q(self):
        """ Scattering vectors, with 2*pi factor and wavelength included."""

        if self.beam.wavelength is None:
            raise ValueError(
                "No wavelength is defined.  Cannot compute Q vectors.")

        return 2.0 * np.pi * self.K / self.beam.wavelength

    @property
    def Qmag(self):
        """ Scattering vector magnitudes, with 2*pi factor and wavelength
        included."""

        return vec_mag(self.Q)

#     @property
#     def mcQ(self):
#         """ Monte Carlo q vectors; add jitter to wavelength, pixel position,
#         incident beam direction for each pixel independently. """
#
#         i = np.arange(self.nF)
#         j = np.arange(self.nS)
#         [i, j] = np.meshgrid(i, j)
#         i = i.ravel() + random(self.n_pixels) - 0.5
#         j = j.ravel() + random(self.n_pixels) - 0.5
#         F = np.outer(i, self.F)
#         S = np.outer(j, self.S)
#         V = self.T + F + S
#         B = np.outer(np.ones(self.n_pixels), self.B) + \
#             randn(self.n_pixels, 3) * self.beam.divergence
#         K = vec_norm(V) - vec_norm(B)
#         lam = self.beam.wavelength * \
#             (1 + randn(self.n_pixels) * self.beam.spectralWidth)
#         return 2 * np.pi * K / np.outer(lam, np.ones(3))

    @property
    def stol(self):
        """ sin(theta)/lambda, where theta is the half angle """

        if self.beam.wavelength is None:
            raise ValueError("No wavelength is defined.  Cannot compute stol.")

        return 0.5 * vec_mag(self.K) / self.beam.wavelength

    @property
    def N(self):
        """ Normal vector to Panel surface (F X S)."""

        N = np.cross(self.F, self.S)
        return N / norm(N)

    @property
    def solid_angle(self):
        """ Solid angles of pixels."""

        if self._sa is None:
            v = vec_norm(self.V)
            n = self.N
            V2 = np.sum(self.V**2, axis=-1)
            A = norm(np.cross(self.F, self.S))
            self._sa = A / V2 * np.dot(v, n.T)

        return self._sa

    @property
    def polarization_factor(self):
        """ The scattering polarization factor. """

        if self._pf is None:
            p = self.beam.P
            u = vec_norm(self.V)
            self._pf = 1.0 - np.abs(u.dot(p))**2

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
            raise ValueError("Data array has zero size (%d x %d)." %
                             (self.nF, self.nS))

        return True

    @property
    def geometry_hash(self):
        """ Hash all of the configured geometry values. Useful for determining
        if anything has changed, or if two panels are different."""

        if self._gh is None:

            F = self._F
            S = self._S
            T = self._T
            nS = self._nS
            nF = self._nF

            if F is None:
                self._gh = None
                return self._gh
            elif S is None:
                self._gh = None
                return self._gh
            elif T is None:
                self._gh = None
                return self._gh
            elif nS == 0:
                self._gh = None
                return self._gh
            elif nF == 0:
                self._gh = None
                return self._gh

            self._gh = hash((F.flat[0], F.flat[1], F.flat[2], S.flat[0], S.flat[1], S.flat[2],
                             T.flat[0], T.flat[1], T.flat[2], nF, nS))

        return self._gh

    def pixels_to_vectors(self, j, i):
        """ Convert pixel indices to translation vectors (i=fast scan, j=slow
        scan)."""

        F = np.outer(i, self.F)
        S = np.outer(j, self.S)
        V = self.T + F + S
        return V

    def compute_reciprocal_space_geometry(self):
        """ Compute the reciprocal-space scattering vectors, multiplied by
        wavelength."""

        self._k = vec_norm(self.V) - self.B

    def clear_geometry_cache(self):
        """ Clear out all derived geometry data."""

        self._ps = None  # Pixel size derived from F/S vectors
        # 3D vectors pointing from interaction region to pixel centers
        self._v = None
        self._sa = None  # Solid angles corresponding to each pixel
        self._pf = None
        self._k = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        # Hash of the configured geometry parameters
        self._gh = None

        if self.panellist is not None:
            self.panellist.clear_geometry_cache()

    def get_vertices(self, edge=False, loop=False):
        """ Get Panel get_vertices; positions of corner pixels."""

        nF = self.nF - 1
        nS = self.nS - 1
        z = 0

        if edge is True:
            z -= 0.5
            nF += 0.5
            nS += 0.5

        j = [z, nF, nF, z]
        i = [z, z, nS, nS]

        if loop is True:
            i.append(i[0])
            j.append(j[0])

        return self.pixels_to_vectors(i, j)

    def get_center(self):
        """ Vector to center of Panel."""

        return np.mean(self.get_vertices(), axis=0)

    @property
    def real_space_bounding_box(self):
        """ Minimum and maximum values of the four corners.  Useful for 
        displays"""

        return self.get_real_space_bounding_box()

    def get_real_space_bounding_box(self):
        """ Return the minimum and maximum values of the four corners."""

        if self._rsbb is None:
            v = self.get_vertices()
            r = np.zeros((2, 3))
            r[0, :] = np.min(v, axis=0)
            r[1, :] = np.max(v, axis=0)
            self._rsbb = r

        return self._rsbb.copy()

    def simple_setup(self,
                     nF=None,
                     nS=None,
                     pixel_size=None,
                     distance=None,
                     wavelength=None,
                     T=None):
        """ Simple way to create a Panel with centered Beam."""

        if nF is None:
            raise ValueError("Number of fast scan pixels is unspecified.")
        if nS is None:
            raise ValueError("Number of slow scan pixels is unspecified.")
        if pixel_size is None:
            raise ValueError("Pixel size is unspecified.")

        self.nF = nF
        self.nS = nS
        self.F = np.array([1, 0, 0]) * pixel_size
        self.S = np.array([0, 1, 0]) * pixel_size

        if T is None:
            if distance is None:
                raise ValueError("Distance is unspecified")
            self.T = np.array([0, 0, distance]) \
                - self.F * (self.nF / 2.0 - 0.5) \
                - self.S * (self.nS / 2.0 - 0.5)
        else:
            self.T = T

        if wavelength is not None:
            if self.beam is None:
                self.beam = source.Beam()
            self.beam.wavelength = wavelength


class PanelList(object):
    """ List container for detector panels, with extra functionality."""

    def __init__(self):
        """ Create an empty Panel array."""

        # Configured data
        self._name = "" # The name of this list (useful for rigid groups)
        self.beam = source.Beam() # X-ray Beam information, common to all panels
        self._panels = []  # List of individual panels

        # Derived data (concatenated from individual panels)
        self._data = None  # Concatenated intensity data
        self._mask = None  # Identify bad pixels [1 means good, 0 means bad]
        self._dark = None  # Counts on detector when there are no x-rays
        # Common pixel size (if not common amongst panels, this will be 'None')
        self._ps = None
        self._sa = None  # Concatenated solid angles
        self._v = None  # Concatenated pixel positions
        self._k = None  # Concatenated reciprocal-space vectors
        self._rsbb = None  # Real-space bounding box of entire Panel list
        self._vll = None  # Look-up table for simple projection
        self._rpix = None  # junk
        self._pf = None  # Polarization factor
        self._gh = None  # Hash of geometries
        # Groups of panels that might be manipulated together
        self._derived_geometry = [
            '_ps', '_v', '_sa', '_k', '_pf', '_rsbb', '_vll', '_rpix',
            '_gh'
        ]  # Default values of these are 'None'

    def __str__(self):
        """ Print something useful when in interactive mode."""

        s = ""
        for p in self:
            s += "\n\n" + p.__str__()
        return (s)

    def __getitem__(self, key):
        """ Get a Panel. Panels may be referenced by name or index."""

        if isinstance(key, str):
            key = self.get_panel_index_by_name(key)
            if key is None:
                raise IndexError("There is no Panel named %s" % key)
                return None
        return self._panels[key]

    def __setitem__(self, key, value):
        """ Set a Panel, check that it is the appropriate type."""

        if not isinstance(value, Panel):
            raise TypeError("You may only add panels to a PanelList object")
        if value.name == "":
            # Give the  Panel a name if it wasn't provided
            value.name = "%d" % key
        self._panels[key] = value

    def __iter__(self):
        """ Iterate through panels. """

        return iter(self._panels)

    def __len__(self):
        """ Return length of Panel list."""

        return len(self._panels)

    def get_panel_indices(self, idx):
        """ Get the indices of a panel for slicing a PanelList array """

        idx = self.get_panel_index_by_name(idx)

        i = 0  # Panel number
        n = 0  # Start index
        for p in self:
            np = p.n_pixels
            start = n
            stop = n + np
            n += np
            if i == idx:
                break
            i += 1

        return [start, stop]

    def get_panel_data(self, idx, dat):
        """ Slice a panel out of a concatentated array. """

        r = self.get_panel_indices(idx)
        return dat[r[0]:r[1]]
    
    def put_panel_data(self, idx, panel_dat, panel_list_dat):
        """ Put panel data into panellist array. """
        
        r = self.get_panel_indices(idx)
        panel_list_dat[r[0]:r[1]] = panel_dat.flat()
        
    def concatentate_panel_data(self,datalist):
        """ Make one contiguous array from a list of panel data arrays."""
        
        return np.concatenate([d.flat for d in datalist])

    def split_panel_data(self,datalist):
        """ Make one contiguous array from a list of panel data arrays."""
        
        dl = []
        for i in range(0,self.n_panels):
            dl.append(self.get_panel_data(i,datalist))
        return dl     

    @property
    def n_pixels(self):
        """ Total number of pixels in all panels."""

        return np.sum([p.n_pixels for p in self])

    @property
    def n_panels(self):
        """ Number of panels."""

        return len(self._panels)

    def append(self, p=None, name=""):
        """ Append a Panel, check that it is of the correct type."""

        if p is None:
            p = Panel()
        if not isinstance(p, Panel):
            raise TypeError("You may only append panels to a PanelList object")
        p.panellist = self

        # Create name if one doesn't exist
        if name != "":
            p.name = name
        elif p.name == "":
            p.name = "%d" % self.n_panels

        self._panels.append(p)

    def get_panel_index_by_name(self, name):
        """ Find the integer index of a Panel by it's unique name """

        if not isinstance(name, basestring):
            return name

        i = 0
        for p in self:
            if p.name == name:
                return i
            i += 1

        return None

    @property
    def V(self):
        """ Concatenated pixel position."""

        if self._v is None:

            self._v = np.concatenate(
                [p.V.reshape(np.product(p.V.shape) / 3, 3) for p in self])

        return self._v

    @property
    def K(self):
        """ Concatenated reciprocal-space vectors."""

        if self._k is None or self._k.shape[0] != self.n_pixels:

            self._k = np.concatenate(
                [p.K.reshape(np.product(p.K.shape) / 3, 3) for p in self])

        return self._k

    @property
    def Q(self):
        """ Concatenated reciprocal-space vectors."""

        return 2 * np.pi * self.K / self.beam.wavelength

    @property
    def Qmag(self):
        """ Concatentated reciprocal-space vector magnitudes."""

        return vec_mag(self.Q)

#     @property
#     def mcQ(self):
#         """ Monte Carlo q vectors; add jitter to wavelength, pixel position,
#         incident Beam direction for each pixel independently. """
#
#         return np.concatenate(
#             [p.mcQ.reshape(np.product(p.mcQ.shape) / 3, 3) for p in self])

    @property
    def stol(self):
        """ sin(theta)/lambda, where theta is the half angle """

        if self.beam.wavelength is None:

            raise ValueError("No wavelength is defined.  Cannot compute stol.")

        return 0.5 * vec_mag(self.K) / self.wavelength

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

        if not isinstance(
                data,
                np.ndarray) and data.ndim == 1 and data.size == self.n_pixels:
            raise ValueError("Must be flattened ndarray of size %d" %
                             self.n_pixels)

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

        if not isinstance(
                mask,
                np.ndarray) and mask.ndim == 1 and mask.size == self.n_pixels:
            raise ValueError("Must be flattened ndarray of size %d" %
                             self.n_pixels)

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

        if not isinstance(
                dark,
                np.ndarray) and dark.ndim == 1 and dark.size == self.n_pixels:
            raise ValueError("Must be flattened ndarray of size %d" %
                             self.n_pixels)

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

    @wavelength.setter
    def wavelength(self, val):

        if self.beam is None:
            self.beam = source.Beam()
        self.beam.wavelength = val

    @property
    def solid_angle(self):
        """ Concatenated solid angles."""

        if self._sa is None:

            self._sa = np.concatenate([
                p.solid_angle.reshape(np.product(p.solid_angle.shape))
                for p in self
            ])

        return self._sa

    @property
    def polarization_factor(self):
        """ Concatenated polarization factors."""

        if self._pf is None:

            self._pf = np.concatenate([
                p.polarization_factor.reshape(
                    np.product(p.polarization_factor.shape)) for p in self
            ])

        return self._pf

    def assemble_data(self, data=None):
        """ Project all intensity data along the Beam direction.  Nearest
        neighbor interpolation.  This is a crude way to display data... """

        if self._vll is None:
            pixelSize = self.pixel_size
            r = self.real_space_bounding_box
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
    def assembled_data(self):
        """ Project all intensity data along the Beam direction.  Nearest
        neighbor interpolation.  This is a crude way to display data... """

        if len(self) == 1:
            return self[0].data

        if self._vll is None:
            pixelSize = self.pixel_size
            r = self.real_space_bounding_box
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
    def real_space_bounding_box(self):
        """ This returns the coordinates of a box just big enough to fit all
        of the panels.  Useful for display purposes. """

        if self._rsbb is None:

            r = np.zeros([2, 3])
            v = np.zeros([2 * self.n_panels, 3])

            i = 0
            for p in self:
                v[i:(i + 2), :] = p.get_real_space_bounding_box()
                i += 2

            r[0, :] = np.min(v, axis=0)
            r[1, :] = np.max(v, axis=0)
            self._rsbb = r

        return self._rsbb.copy()

    @property
    def pixel_size(self):
        """ Return the average pixel size, if all pixels are identical."""

        if self._ps is None:

            pix = np.zeros(self.n_panels)
            i = 0
            for p in self:
                pix[i] = p.pixel_size
                i += 1
            mnpix = np.mean(pix)
            if all(np.absolute((pix - mnpix) / mnpix) < 1e-6):
                self._ps = mnpix
            else:
                raise ValueError(
                    "Pixel sizes in Panel list are not all the same.")

        return self._ps.copy()

    def get_center(self):
        """ Mean center of Panel list. """

        c = np.zeros((self.n_panels, 3))
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
        """ Delete derived data relevant to geometry.  Normally used
         internally. """

        self._ps = None  # Pixel size derived from F/S vectors
        # 3D vectors pointing from interaction region to pixel centers
        self._v = None
        self._sa = None  # Solid angles corresponding to each pixel
        self._k = None  # Reciprocal space vectors multiplied by wavelength
        self._rsbb = None  # Real-space bounding box information
        # Hash of the configured geometry parameters
        self._gh = None

    @property
    def geometry_hash(self):
        """ Hash all of the configured geometry values. """

        if self._gh is None:

            a = tuple([p.geometry_hash for p in self])

            if None in a:
                self._gh = None
                return self._gh

            self._gh = hash(a)

        return self._gh

    def simple_setup(self,
                     nF=None,
                     nS=None,
                     pixel_size=None,
                     distance=None,
                     wavelength=None,
                     T=None):
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
        p.simple_setup(nF, nS, pixel_size, distance, wavelength, T)
        self.beam = p.beam
        self.append(p)


class SimpleDetector(Panel):

    def __init__(self, n_pixels=1000, pixsize=0.00005,
                 detdist=0.05, wavelen=1, *args, **kwargs):

        Panel.__init__(self, *args, **kwargs)

        self.detector_distance = detdist
        self.wavelength = wavelen
        self.si_energy = units.hc / (wavelen * 1e-10)

#       make a single panel detector:
        self.simple_setup(
            n_pixels,
            n_pixels + 1,
            pixsize,
            detdist,
            wavelen,)

#       shape of the 2D det panel (2D image)
        self.img_sh = (self.nS, self.nF)

    def readout(self, amplitudes):
        self.intens = (np.abs(amplitudes)**2).reshape(self.img_sh)
        return self.intens

    def readout_finite(self, amplitudes, qmin, qmax, flux=1e20):
        struct_fact = (np.abs(amplitudes)**2).astype(np.float64)

        if qmin < self.Qmag.min():
            qmin = self.Qmag.min()
        if qmax > self.Qmag.max():
            qmax = self.Qmag.max()

        ilow = np.where(self.Qmag < qmin)[0]
        ihigh = np.where(self.Qmag > qmax)[0]

        if ilow.size:
            struct_fact[ilow] = 0
        if ihigh.size:
            struct_fact[ihigh] = 0

        rad_electron = 2.82e-13  # cm
        phot_per_pix = struct_fact * self.solidAngle * flux * rad_electron**2
        total_phot = int(phot_per_pix.sum())

        pvals = struct_fact / struct_fact.sum()

        self.intens = np.random.multinomial(total_phot, pvals)

        self.intens = self.intens.reshape(self.img_sh)

        return self.intens

    def display(self, use_log=True, vmax=None, **kwargs):
        if 'matplotlib' not in sys.modules:
            print("You need matplotlib to plot!")
            return
        #plt = matplotlib.pylab
        fig = plt.figure(**kwargs)
        ax = plt.gca()

        qx_min, qy_min = self.Q[:, :2].min(0)
        qx_max, qy_max = self.Q[:, :2].max(0)
        extent = (qx_min, qx_max, qy_min, qy_max)
        if use_log:
            ax_img = ax.imshow(
                np.log1p(
                    self.intens),
                extent=extent,
                cmap='viridis',
                interpolation='lanczos')
            cbar = fig.colorbar(ax_img)
            cbar.ax.set_ylabel('log(photon counts)', rotation=270, labelpad=12)
        else:
            assert(vmax is not None)
            ax_img = ax.imshow(
                self.intens,
                extent=extent,
                cmap='viridis',
                interpolation='lanczos',
                vmax=vmax)
            cbar = fig.colorbar(ax_img)
            cbar.ax.set_ylabel('photon counts', rotation=270, labelpad=12)

        ax.set_xlabel(r'$q_x\,\,\AA^{-1}$')
        ax.set_ylabel(r'$q_y\,\,\AA^{-1}$')

        plt.show()
