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

from .utils import vec_norm, vec_mag, vec_check
from . import source
from . import units


class PADGeometry(object):
    r"""
    This is a simplified version of the Panel class.  Hopefully it replaces Panel.  One main difference is that it does
    not include any information about the source, which makes a lot more sense and removes several headaches that I
    dealt with previously.  Another big difference is the emphasis on simplicity.  So far, there is no cache for
    derived arrays, but maybe that will be added later (but only on an as-needed basis...).
    As a result of simplifications, there are no checks; the programmer must think.
    """

    # These are the configurable parameters.  No defaults.  One must think.

    n_fs = None  #: The number of fast-scan pixels.
    n_ss = None  #: The number of slow-scan pixels.
    _fs_vec = None  #: The fast-scan basis vector.
    _ss_vec = None  #: The slow-scan basis vector.
    _t_vec = None  #: The overall translation vector.

    @property
    def fs_vec(self):
        r""" Fast-scan basis vector. """

        return self._fs_vec

    @property
    def ss_vec(self):
        r""" Slow-scan basis vector. """

        return self._ss_vec

    @property
    def t_vec(self):
        r""" Translation vector pointing from origin to center of corner pixel, which is first in memory. """

        return self._t_vec

    # The reason for these setters is that some assumptions are made about the shape of vectors used within bornagain.
    # TODO: Document assumptions made about vectors

    @fs_vec.setter
    def fs_vec(self, fs_vec):
        self._fs_vec = vec_check(fs_vec)

    @ss_vec.setter
    def ss_vec(self, ss_vec):
        self._ss_vec = vec_check(ss_vec)

    @t_vec.setter
    def t_vec(self, t_vec):
        self._t_vec = vec_check(t_vec)

    def simple_setup(self, n_pixels = 1000, pixel_size = 100e-6, distance = 0.1):
        r""" Make this a square PAD with beam at center.

        Returns:
            object:
        """

        self.n_fs = n_pixels
        self.n_ss = n_pixels
        self.fs_vec = [pixel_size, 0, 0]
        self.ss_vec = [0, pixel_size, 0]
        self.t_vec = [-pixel_size * (n_pixels / 2.0 - 0.5), -pixel_size * (n_pixels / 2.0 - 0.5), distance]

    def pixel_size(self):
        r""" Return pixel size, assuming square pixels. """

        return np.mean([vec_mag(self.fs_vec), vec_mag(self.ss_vec)])

    def shape(self):
        r""" Return tuple corresponding to the numpy shape of this PAD. """

        return (self.n_ss, self.n_fs)

    def indices_to_vectors(self, j, i):
        r"""
        Convert pixel indices to translation vectors pointing from origin to position on panel.
        The positions need not lie on the actual panel; this assums an infinite plane.

        Arguments:
            i (float) :
                Fast-scan index.
            j (float) :
                Slow-scan index.

        Returns:
            Nx3 numpy array
        """

        i = np.array(i)
        j = np.array(j)
        f = np.outer(i.ravel(), self.fs_vec)
        s = np.outer(j.ravel(), self.ss_vec)
        return vec_check(self.t_vec + f + s)

    def position_vecs(self):
        r"""
        Compute vectors pointing from origin to pixel centers.

        Returns: Nx3 numpy array
        """

        i = np.arange(self.n_fs)
        j = np.arange(self.n_ss)
        [i, j] = np.meshgrid(i, j)
        i.ravel()
        j.ravel()
        return self.indices_to_vectors(j, i)

    def norm_vec(self):
        r""" The vector that is normal to the PAD plane. """

        return vec_norm(np.cross(self.fs_vec, self.ss_vec))

    def ds_vecs(self, beam_vec=None):
        r""" Normalized scattering vectors s - s0 where s0 is the incident beam direction
        and s is the outgoing vector for a given pixel.  This does **not** have
        the 2*pi/lambda factor included."""

        return vec_norm(self.position_vecs()) - vec_check(beam_vec)

    def q_vecs(self, beam_vec=None, wavelength=None):
        r"""
        Calculate scattering vectors:

            :math:`\vec{q}_{ij}=\frac{2\pi}{\lambda}\left(\hat{v}_{ij} - \hat{b}\right)`

        Returns: numpy array
        """

        return (2 * np.pi / wavelength) * self.ds_vecs(beam_vec=beam_vec)

    def solid_angles(self):
        r"""
        Calculate solid angles of pixels.   Assuming the pixel is small, the approximation to the solid angle is:

            :math:`\Delta \Omega_{ij} \approx \frac{\text{Area}}{R^2}\cos(\theta) = \frac{|\vec{f}\times\vec{s}|}{|v|^2}\hat{n}\cdot \hat{v}_{ij}`.

        Returns: numpy array
        """

        v = self.position_vecs()
        n = self.norm_vec()

        A = vec_mag(np.cross(self.fs_vec, self.ss_vec))  # Area of the pixel
        R2 = vec_mag(v) ** 2  # Distance to the pixel, squared
        cs = np.dot(n, vec_norm(v).T)  # Inclination factor: cos(theta)
        sa = (A / R2) * cs  # Solid angle

        return sa.ravel()

    def polarization_factors(self, polarization_vec, beam_vec, weight=None):
        r"""
        The scattering polarization factors.

        Arguments:
            polarization_vec (numpy array) :
                First beam polarization vector (second is this one crossed with beam vector)
            beam_vec (numpy array) :
                Incident beam vector
            weight (float) :
                The weight of the first polarization component (second is one minus this weight)

        Returns:  numpy array
        """

        v = vec_norm(self.position_vecs())
        u = vec_norm(vec_check(polarization_vec))
        b = vec_norm(vec_check(beam_vec))
        up = np.cross(u, b)

        if weight is None:
            w1 = 1
            w2 = 0
        else:
            w1 = weight
            w2 = 1 - weight

        p1 = w1 * (1 - np.abs(np.dot(u, v.T)) ** 2)
        p2 = w2 * (1 - np.abs(np.dot(up, v.T)) ** 2)
        p = p1 + p2

        return p.ravel()

    def scattering_angles(self, beam_vec=None):
        """
        Scattering angles (i.e. half the Bragg angles).

        Arguments:
            beam_vec (numpy array) :
                Incident beam vector.

        Returns: numpy array
        """

        v = self.position_vecs()

        return np.arccos(vec_check(beam_vec), v.T)

    def reshape(self, dat):

        return dat.reshape(self.shape())


class PADAssembler(object):
    r"""
    Assemble PAD data into a fake single-panel PAD.  This is done in a lazy way.  The resulting image is not
    centered in any way; the fake detector is a snug fit to the individual PADs.

    A list of PADGeometry objects are required on initialization, as the first argument.  The data needed to
    "interpolate" are cached, hence the need for a class.  The geometry cannot change; there is no update method.
    """

    def __init__(self, pad_list):
        pixel_size = vec_mag(pad_list[0].fs_vec)
        v = np.concatenate([p.position_vecs() for p in pad_list])
        v -= np.min(v, axis=0)
        v /= pixel_size
        v = np.floor(v).astype(np.int)
        m = np.max(v, axis=0)
        a = np.zeros([m[0] + 1, m[1] + 1])
        self.v = v
        self.a = a

    def assemble_data(self, data):
        r"""
        Given a contiguous block of data, create the fake single-panel PAD.

        Arguments:
            data (numpy array):
                Image data

        Returns:
            assembled_data (numpy array):
                Assembled PAD image
        """
        a = self.a
        v = self.v
        a[v[:, 0], v[:, 1]] = data
        return a.copy()

    def assemble_data_list(self, data_list):
        r"""
        Same as assemble_data() method, but accepts a list of individual panels in the form of a list.

        Arguments:
            data_list (list of numpy arrays):
                Image data

        Returns:
            assembled_data (numpy array):
                Assembled PAD image
        """
        self.assemble_data(np.ravel(data_list))


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

    def shape(self):
        """ Return expected shape of data arrays. """

        return (self.nS, self.nF)

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
            self._k = vec_norm(self.V) - self.B
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
            V2 = np.sum(self.V ** 2, axis=-1)
            A = norm(np.cross(self.F, self.S))
            self._sa = A / V2 * np.dot(n, v.T)

        return self._sa

    @property
    def polarization_factor(self):
        """ The scattering polarization factor. """

        if self._pf is None:
            p = self.beam.polarization_vectors[0]
            u = vec_norm(self.V)
            self._pf = self.beam.polarization_weights[0] * (1.0 - np.abs(u.dot(p)) ** 2)

            p = self.beam.polarization_vectors[1]
            u = vec_norm(self.V)
            self._pf += self.beam.polarization_weights[1] * (1.0 - np.abs(u.dot(p)) ** 2)

            self._pf /= self.beam.polarization_weights[0] + self.beam.polarization_weights[1]

        return self._pf

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
        self._name = ""  # The name of this list (useful for rigid groups)
        self.beam = source.Beam()  # X-ray Beam information, common to all panels
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

    def zeros(self):
        """ Return an array of zeros of length n_pixels."""

        return np.zeros([self.n_pixels])

    def ones(self):
        """ Return array of ones of length n_pixels."""

        return np.ones([self.n_pixels])

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
        panel_list_dat[r[0]:r[1]] = panel_dat.ravel()

    def concatentate_panel_data(self, datalist):
        """ Make one contiguous array from a list of panel data arrays."""

        return np.concatenate([d.flat for d in datalist])

    def split_panel_data(self, datalist):
        """ Make one contiguous array from a list of panel data arrays."""

        dl = []
        for i in range(0, self.n_panels):
            dl.append(self.get_panel_data(i, datalist))
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

    def assemble_data(self, data):
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

        adat[Vll[:, 1], Vll[:, 0]] = data

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

    def clear_geometry_cache(self):
        """ Delete derived data relevant to geometry.  Normally used
         internally. """

        self._ps = None
        self._v = None
        self._pf = None
        self._sa = None
        self._k = None
        self._rsbb = None
        self._gh = None

    @property
    def geometry_hash(self):
        """ Hash all of the configured geometry values. """

        if self._gh is None:

            if self.n_panels == 0:
                self._gh = None
                return self._gh

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
    """
    A simple wrapper for folks who wish to use use the Panels
    class without learning the intricacies. This will return a detector object representing a
    square detector
    
    .. note:: 
        - One can readout pixel intensities using :func:`readout`
        - After reading out amplitudes, one can display pixels using :func:`display`

    Arguments
        - n_pixels (int)
            the number of pixels along one edge

        - pixsize (float)
            the edge length of the square pixels in meters

        - detdist (float)
            the distance from the interaction region to the point where 
            the forward beam intersects the detector (in meters)

        - wavelen (float)
            the wavelength of the photons (in Angstroms)

    """

    def __init__(self, n_pixels=1000, pixsize=0.00005, detdist=0.05, wavelen=1., *args, **kwargs):

        Panel.__init__(self, *args, **kwargs)

        self.detector_distance = detdist
        self.wavelength = wavelen
        self.si_energy = units.hc / (wavelen * 1e-10)

        self.fig = None

        #       make a single panel detector:
        self.simple_setup(
            n_pixels,
            n_pixels,
            pixsize,
            detdist,
            wavelen, )

        #       shape of the 2D det panel (2D image)
        self.img_sh = (self.nS, self.nF)
        self.center = map(lambda x: x / 2., self.img_sh)

        self.SOLID_ANG = np.cos(np.arcsin(self.Qmag * self.wavelength / 4 / np.pi)) ** 3

        self.rad2q = lambda rad: 4 * np.pi * np.sin(.5 * np.arctan(rad * pixsize / detdist)) / wavelen
        self.q2rad = lambda q: np.tan(np.arcsin(q * wavelen / 4 / np.pi) * 2) * detdist / pixsize

        self.intens = None

    def readout(self, amplitudes):
        """
        Given scattering amplitudes, this calculates the corresponding intensity values.

        Arguments
            amplitudes (complex np.ndarray) : Scattering amplitudes same shape as `self.Q`
            
        Returns
            np.ndarray : Scattering intensities as a 2-D image.
        """
        self.intens = (np.abs(amplitudes) ** 2).reshape(self.img_sh)
        return self.intens

    def readout_finite(self, amplitudes, qmin, qmax, flux=1e20):
        """
        Get scattering intensities as a 2D image considering 
        finite scattered photons

        Arguments:
            amplitudes (complex np.ndarray) : Scattering amplitudes same shape as `self.Q`.
            qmin (float) : Minimum q to generate intensities
            qmax (float) : Maximum q to generate intenities
            flux (float) : Forward beam flux in Photons per square centimeter

        Returns:
            np.ndarray : Scattering intensities as a 2-D image.
        """
        self.intens = (np.abs(amplitudes) ** 2).reshape(self.img_sh)
        struct_fact = (np.abs(amplitudes) ** 2).astype(np.float64)

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
        phot_per_pix = struct_fact * self.SOLID_ANG * flux * rad_electron ** 2
        total_phot = int(phot_per_pix.sum())

        pvals = struct_fact / struct_fact.sum()

        self.intens = np.random.multinomial(total_phot, pvals)

        self.intens = self.intens.reshape(self.img_sh)

        return self.intens

    def display(self, use_log=True, vmax=None, pause=None, **kwargs):
        """
        Displays a detector. Extra kwargs are passed
        to matplotlib.figure
        
        .. note::
            - Requires matplotlib.
            - Must first run :func:`readout` or :func:`readout_finite`
                at least one time
        
        Arguments
            - use_log (bool)
                whether to use log-scaling when displaying the intensity image.

            - vmax (float)
                colorbar scaling argument.
        """

        assert (self.intens is not None)

        if 'matplotlib' not in sys.modules:
            print("You need matplotlib to plot!")
            return
        # plt = matplotlib.pylab

        if self.fig is None:
            fig = plt.figure(**kwargs)
        else:
            fig = self.fig
        fig.clear()
        ax = plt.gca()
        qx_min, qy_min = self.Q[:, :2].min(0)
        qx_max, qy_max = self.Q[:, :2].max(0)
        extent = (qx_min, qx_max, qy_min, qy_max)
        if use_log:
            ax_img = ax.imshow(
                np.log1p(
                    self.intens),
                extent=extent,
                cmap='gnuplot',
                interpolation='lanczos')
            cbar = fig.colorbar(ax_img)
            cbar.ax.set_ylabel('log(photon counts)', rotation=270, labelpad=12)
        else:
            assert (vmax is not None)
            ax_img = ax.imshow(
                self.intens,
                extent=extent,
                cmap='gnuplot',
                interpolation='lanczos',
                vmax=vmax)
            cbar = fig.colorbar(ax_img)
            cbar.ax.set_ylabel('photon counts', rotation=270, labelpad=12)

        ax.set_xlabel(r'$q_x\,\,\AA^{-1}$')
        ax.set_ylabel(r'$q_y\,\,\AA^{-1}$')

        if pause is None:
            plt.show()
        elif pause is not None and self.fig is None:
            self.fig = fig
            plt.draw()
            plt.pause(pause)
        else:
            plt.draw()
            plt.pause(pause)

class IcosphereGeometry():
    """
    Experimental class for a spherical detector that follows the "icosphere" geometry.
    The Icosphere is generated by sub-dividing the vertices of an icosahedron.
    The following blog was helpful:
    http://sinestesia.co/blog/tutorials/python-icospheres/
    The code is quite slow; needs to be vectorized with numpy...
    """

    n_subdivisions = 1
    radius = 1

    def __init__(self, n_subdivisions=1, radius=1):

        self.n_subdivisions = n_subdivisions
        self.radius = radius

    def _vertex(self, x, y, z):
        """ Return vertex coordinates fixed to the unit sphere """

        length = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        return [(i * self.radius) / length for i in (x, y, z)]

    def _middle_point(self, point_1, point_2, verts, middle_point_cache):
        """ Find a middle point and project to the unit sphere """

        # We check if we have already cut this edge first
        # to avoid duplicated verts
        smaller_index = min(point_1, point_2)
        greater_index = max(point_1, point_2)

        key = '{0}-{1}'.format(smaller_index, greater_index)

        if key in middle_point_cache:
            return middle_point_cache[key]

        # If it's not in cache, then we can cut it
        vert_1 = verts[point_1]
        vert_2 = verts[point_2]
        middle = [sum(i) / 2 for i in zip(vert_1, vert_2)]

        verts.append(self._vertex(*middle))

        index = len(verts) - 1
        middle_point_cache[key] = index

        return index

    def compute_vertices_and_faces(self):

        # Make the base icosahedron

        vertex = self._vertex
        middle_point = self._middle_point
        middle_point_cache = {}

        # Golden ratio
        PHI = (1 + np.sqrt(5)) / 2

        verts = [
            vertex(-1, PHI, 0),
            vertex(1, PHI, 0),
            vertex(-1, -PHI, 0),
            vertex(1, -PHI, 0),

            vertex(0, -1, PHI),
            vertex(0, 1, PHI),
            vertex(0, -1, -PHI),
            vertex(0, 1, -PHI),

            vertex(PHI, 0, -1),
            vertex(PHI, 0, 1),
            vertex(-PHI, 0, -1),
            vertex(-PHI, 0, 1),
        ]

        faces = [
            # 5 faces around point 0
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],

            # Adjacent faces
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],

            # 5 faces around 3
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],

            # Adjacent faces
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]

        # -----------------------------------------------------------------------------
        # Subdivisions

        for i in range(self.n_subdivisions):
            faces_subdiv = []

            for tri in faces:
                v1 = middle_point(tri[0], tri[1], verts, middle_point_cache)
                v2 = middle_point(tri[1], tri[2], verts, middle_point_cache)
                v3 = middle_point(tri[2], tri[0], verts, middle_point_cache)

                faces_subdiv.append([tri[0], v1, v3])
                faces_subdiv.append([tri[1], v2, v1])
                faces_subdiv.append([tri[2], v3, v2])
                faces_subdiv.append([v1, v2, v3])

            faces = faces_subdiv

        faces = np.array(faces)
        verts = np.array(verts)
        n_faces = faces.shape[0]

        face_centers = np.zeros([n_faces, 3])
        for i in range(0, n_faces):
            face_centers[i, :] = (verts[faces[i, 0], :] + verts[faces[i, 1], :] + verts[faces[i, 2], :]) / 3

        return verts, faces, face_centers