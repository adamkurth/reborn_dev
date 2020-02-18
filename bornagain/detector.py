"""
Classes for analyzing/simulating diffraction data contained in pixel array
detectors (PADs).
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import h5py

from .utils import vec_norm, vec_mag, triangle_solid_angle, depreciate, warn


class PADGeometry(object):
    r"""
    A container for pixel-array detector (PAD) geometry specification, with hepful methods for generating:

    - Vectors from sample to pixel
    - Scattering vectors (i.e. "q" vectors... provided beam information)
    - Scattering vector magnitudes.
    - Scattering angles (twice the Bragg angle).
    - Polarization factors

    There are also a few methods for generating new arrays (zeros, ones, random) and re-shaping flattened arrays to 2D
    arrays.

    """

    # pylint: disable=too-many-public-methods
    # pylint: disable=too-many-instance-attributes
    # These are the configurable parameters.  No defaults.  One must think.

    def __init__(self, n_pixels=None, distance=None, pixel_size=None, shape=None):
        r"""
        High-level initialization.  Centers the detector in the x-y plane.

        Arguments:
            n_pixels (int or numpy array): Shape of the panels.  The first element is the slow-scan shape.  If there is
                                           only one element, or if it is an int, then it will be a square panel.
            distance (float): Sample-to-detector distance, where the beam is taken along the third ("Z") axis
            pixel_size (float): Size of the pixels in SI units.
        """

        self._n_fs = None
        self._n_ss = None
        self._fs_vec = None
        self._ss_vec = None
        self._t_vec = None

        if distance is not None and pixel_size is not None:

            self.simple_setup(n_pixels=n_pixels, distance=distance, pixel_size=pixel_size, shape=shape)

    def __str__(self):

        s = ''
        s += 'n_fs: %s\n' % self.n_fs.__str__()
        s += 'n_ss: %s\n' % self.n_ss.__str__()
        s += 'fs_vec: %s\n' % self.fs_vec.__str__()
        s += 'ss_vec: %s\n' % self.ss_vec.__str__()
        s += 't_vec: %s' % self.t_vec.__str__()
        return s

    @property
    def n_fs(self):
        r"""Number of fast-scan pixels."""
        if self._n_fs is None:
            raise ValueError('n_fs has not been defined for this PADGeometry!')
        return self._n_fs

    @n_fs.setter
    def n_fs(self, val):
        self._n_fs = val

    @property
    def n_ss(self):
        r"""Number of slow-scan pixels."""
        if self._n_ss is None:
            raise ValueError('n_ss has not been defined for this PADGeometry!')
        return self._n_ss

    @n_ss.setter
    def n_ss(self, val):
        self._n_ss = val

    @property
    def n_pixels(self):
        r"""Total number of pixels (:math:`n_{fs} \cdot n_{ss}`)"""
        return self.n_fs * self.n_ss

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

    @fs_vec.setter
    def fs_vec(self, fs_vec):
        self._fs_vec = np.array(fs_vec)

    @ss_vec.setter
    def ss_vec(self, ss_vec):
        self._ss_vec = np.array(ss_vec)

    @t_vec.setter
    def t_vec(self, t_vec):
        self._t_vec = np.array(t_vec)

    def save(self, save_fname):
        r"""Saves an hdf5 file with class attributes for later use"""
        with h5py.File(save_fname, "w") as h:
            for name, data in vars(self).items():
                h.create_dataset(name, data=data)

    @classmethod
    def load(cls, fname):
        r""" load a PAD object from fname"""
        pad = cls()
        with h5py.File(fname, "r") as h:
            for name in h.keys():
                data = h[name].value
                setattr(pad, name, data)
        return pad

    def simple_setup(self, n_pixels=None, pixel_size=None, distance=None, shape=None):
        r""" Make this a square PAD with beam at center.

        Arguments:
            shape : The shape of the panel, consistent with a numpy array shape.
            pixel_size : Pixel size in SI units.
            distance : Detector distance in SI units.
            n_pixels : (DEPRECIATED) Either the shape of the panel, or a single number if a square detector is desired.
                       Do not use this input as it will be removed in the future.

        Returns:
            object:
        """

        if n_pixels is not None:
            depreciate('The redundant "n_pixels" keyword argument in simple_setup is depreciated.  Use "shape" keyword '
                       'instead.')

        if pixel_size is None:
            warn('Setting pixel_size in simple_setup to 100e-6.  You should specify this value explicitly.')
            pixel_size = 100e-6

        if distance is None:
            warn('Setting distance in simple_setup to 0.1.  You should specify this value explicitly.')
            distance = 0.1

        if shape is not None:
            self.n_fs = shape[1]
            self.n_ss = shape[0]
        else:
            if n_pixels is None:
                warn('Setting n_pixels in simple_setup to 1000.  You should specify this value explicitly.')
                n_pixels = 1000
            try:
                self.n_fs = n_pixels[1]
                self.n_ss = n_pixels[0]
            except TypeError:
                self.n_fs = n_pixels
                self.n_ss = n_pixels

        self.fs_vec = np.array([pixel_size, 0, 0])
        self.ss_vec = np.array([0, pixel_size, 0])
        self.t_vec = np.array([pixel_size * -(self.n_fs / 2.0 - 0.5), pixel_size * -(self.n_ss / 2.0 - 0.5), distance])

    def pixel_size(self):
        r""" Return pixel size, assuming square pixels. """

        return np.mean([vec_mag(self.fs_vec), vec_mag(self.ss_vec)])

    def shape(self):
        r""" Return tuple corresponding to the numpy shape of this PAD. """

        return self.n_ss, self.n_fs

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
        return self.t_vec + f + s

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

    def ds_vecs(self, beam_vec=None, beam=None):
        r"""
        Normalized scattering vectors s - s0 where s0 is the incident beam direction
        (`beam_vec`) and  s is the outgoing vector for a given pixel.  This does **not** have
        the 2*pi/lambda factor included.

        Arguments:
            beam_vec (tuple or numpy array): specify the unit vector of the incident beam
            beam (source.Beam instance): specify incident beam properties.  If provided, you may omit the specification
                                         of beam_vec ect.

        Returns: numpy array

        """

        if beam is not None:
            beam_vec = beam.beam_vec

        return vec_norm(self.position_vecs()) - beam_vec

    def q_vecs(self, beam_vec=None, wavelength=None, beam=None):
        r"""
        Calculate scattering vectors:

            :math:`\vec{q}_{ij}=\frac{2\pi}{\lambda}\left(\hat{v}_{ij} - \hat{b}\right)`

        Arguments:
            beam_vec (tuple or numpy array): specify the unit vector of the incident beam
            wavelength (float): wavelength
            beam (source.Beam instance): specify incident beam properties.  If provided, you may omit the specification
                                         of beam_vec ect.

        Returns: numpy array
        """

        if beam is not None:
            beam_vec = beam.beam_vec
            wavelength = beam.wavelength

        return (2 * np.pi / wavelength) * self.ds_vecs(beam_vec=beam_vec)

    def q_mags(self, beam_vec=None, wavelength=None, beam=None):
        r"""
        Calculate scattering vector magnitudes:

        Arguments:
            beam_vec (tuple or numpy array): specify the unit vector of the incident beam
            wavelength (float): wavelength
            beam (source.Beam instance): specify incident beam properties.  If provided, you may omit the specification
                                        of beam_vec ect.

        Returns: numpy array
        """

        if beam is not None:
            beam_vec = beam.beam_vec
            wavelength = beam.wavelength

        return vec_mag(self.q_vecs(beam_vec=beam_vec, wavelength=wavelength))

    def solid_angles(self):
        r"""
        Calculate solid angles of pixels.  See solid_angles2 method.

        Returns: numpy array
        """

        return self.solid_angles2()

    def solid_angles1(self):
        r"""
        Calculate solid angles of pixels vectorally, assuming the pixels have small angular extent.

        Returns: numpy array
        """

        v = self.position_vecs()
        n = self.norm_vec()

        a = vec_mag(np.cross(self.fs_vec, self.ss_vec))  # Area of the pixel
        r2 = vec_mag(v) ** 2  # Distance to the pixel, squared
        cs = np.dot(n, vec_norm(v).T)  # Inclination factor: cos(theta)
        sa = (a / r2) * cs  # Solid angle

        return np.abs(sa.ravel())

    def solid_angles2(self):
        r"""
        Pixel solid angles calculated using the method of Van Oosterom, A. & Strackee, J. Biomed. Eng., IEEE T
        ransactions on BME-30, 125-126 (1983).  Divide each pixel up into two triangles with vertices R1,R2,R3 and
        R2,R3,R4. Then use analytical form to find the solid angle of each triangle. Sum them to get the solid angle of
        pixel.

        Thanks to Derek Mendez, who thanks Jonas Sellberg.

        Returns: numpy array
        """
        k = self.position_vecs()
        r1 = k - self.fs_vec * .5 - self.ss_vec * .5
        r2 = k + self.fs_vec * .5 - self.ss_vec * .5
        r3 = k - self.fs_vec * .5 + self.ss_vec * .5
        r4 = k + self.fs_vec * .5 + self.ss_vec * .5
        sa_1 = triangle_solid_angle(r1, r2, r3)
        sa_2 = triangle_solid_angle(r4, r2, r3)
        return sa_1 + sa_2

    def polarization_factors(self, polarization_vec=None, beam_vec=None, weight=None, beam=None):
        r"""
        The scattering polarization factors.

        Arguments:
            polarization_vec (numpy array) :
                First beam polarization vector (second is this one crossed with beam vector)
            beam_vec (numpy array) :
                Incident beam vector
            weight (float) :
                The weight of the first polarization component (second is one minus this weight)
            beam (source.Beam instance): specify incident beam properties.  If provided, you may omit the specification
                                         of beam_vec ect.

        Returns:  numpy array
        """

        if beam is not None:
            beam_vec = beam.beam_vec
            polarization_vec = beam.polarization_vec
            weight = beam.polarization_weight

        v = vec_norm(self.position_vecs())
        u = vec_norm(np.array(polarization_vec))
        b = vec_norm(np.array(beam_vec))
        up = np.cross(u, b)

        if weight is None:
            w1 = 1
            w2 = 0
        else:
            w1 = weight
            w2 = 1 - weight

        p1 = w1 * (1 - np.abs(np.dot(v, u)) ** 2)
        p2 = w2 * (1 - np.abs(np.dot(v, up)) ** 2)
        p = p1 + p2
        return p.ravel()

    def scattering_angles(self, beam_vec=None, beam=None):
        r"""
        Scattering angles (i.e. half the Bragg angles).

        Arguments:
            beam_vec (numpy array) :
                Incident beam vector.
            beam (source.Beam instance): specify incident beam properties.  If provided, you may omit the specification
                                         of beam_vec ect.

        Returns: numpy array
        """

        if beam is not None and beam_vec is None:
            beam_vec = beam.beam_vec
        elif beam_vec is not None and beam is None:
            pass
        else:
            raise ValueError('Scattering angles cannot be computed without knowing the incident beam direction')

        return np.arccos(vec_norm(self.position_vecs()).dot(beam_vec.ravel()))

    def beamstop_mask(self, beam=None, q_min=None, min_angle=None):
        r"""

        Arguments:
            beam: Instance of the Beam class (for wavelength)
            wavelength: Specify wavelength (needed for q_mags)
            q_min: Minimum q magnitude
            min_angle: Minimum scattering angle

        Returns:

        """

        if beam is None:
            raise ValueError("A beam must be provided")
        mask = self.ones().ravel()
        if q_min is not None:
            mask[self.q_mags(beam=beam) < q_min] = 0
        elif min_angle is not None:
            mask[self.scattering_angles(beam=beam) < min_angle] = 0
        else:
            raise ValueError("Specify either q_min (and wavelength) or min_angle")
        return self.reshape(mask)

    def reshape(self, dat):
        r"""

        Re-shape a flattened array to a 2D array.

        Arguments:
            dat (numpy array): The flattened data array

        Returns: a 2D numpy array

        """

        return dat.reshape(self.shape())

    def zeros(self):
        r"""
        For convenience: np.zeros((self.n_ss, self.n_fs))
        """
        return np.zeros((self.n_ss, self.n_fs))

    def ones(self):
        r"""
        For convenience: np.ones((self.n_ss, self.n_fs))
        """
        return np.ones((self.n_ss, self.n_fs))

    def random(self):
        r"""
        For convenience: np.random.random((self.n_ss, self.n_fs))
        """
        return np.random.random((self.n_ss, self.n_fs))  # pylint: disable=no-member

    def max_resolution(self, beam=None):
        r"""
        Maximum resolution over all pixels: 2*pi/q

        Arguments:
            beam: A Beam class instance.

        Returns:
            float
        """
        return 2 * np.pi / np.max(self.q_mags(beam=beam))


def tiled_pad_geometry_list(pad_shape=(512, 1024), pixel_size=100e-6, distance=0.1, tiling_shape=(4, 2), pad_gap=50):
    r"""
    Make a list of PADGeometry instances with identical panel sizes, tiled in a regular grid.

    Arguments:
        pad_shape (tuple): Shape of the pads (slow scan, fast scan)
        pixel_size (float): Pixel size in SI units
        distance (float): Detector distance in SI units
        tiling_shape (tuple): Shape of tiling (n tiles along slow scan, n tiles along fast scan)
        pad_gap (float): Gap between pad tiles in SI units

    Returns:
        list of PADGeometry instances
    """

    pads = []

    tilefs_sep = pad_shape[1] + pad_gap/pixel_size / 2 #+ 0.5
    tilefs_pos = (np.arange(tiling_shape[1]) - (tiling_shape[1] - 1) / 2) * tilefs_sep
    tiless_sep = pad_shape[0] + pad_gap/pixel_size / 2 #+ 0.5
    tiless_pos = (np.arange(tiling_shape[0]) - (tiling_shape[0] - 1) / 2) * tiless_sep

    for fs_cent in tilefs_pos:  # fast scan
        for ss_cent in tiless_pos:  # slow scan
            pad = PADGeometry(shape=pad_shape, pixel_size=pixel_size, distance=distance)
            pad.t_vec += pad.fs_vec * fs_cent - 0.5*pixel_size
            pad.t_vec += pad.ss_vec * ss_cent - 0.5*pixel_size
            pads.append(pad)

    return pads


def split_pad_data(pad_list, data):
    r"""

    Given a contiguous block of data, split it up into individual PAD panels

    Arguments:
        pad_list: A list of PADGeometry instances
        data: A contiguous array with data values (total pixels to add up to sum of pixels in all PADs)

    Returns:
        A list of 2D PAD data arrays

    """

    data_list = []

    offset = 0
    for pad in pad_list:
        data_list.append(pad.reshape(data[offset:(offset + pad.n_pixels)]))
        offset += pad.n_pixels

    return data_list


def edge_mask(data, n):
    r"""
    Make an "edge mask"; an array of ones with zeros around the edges.
    The mask will be the same type as the data (e.g. double).

    Arguments:
        data (2D numpy array): a data array (for shape reference)
        n (int): number of pixels to mask around edges

    Returns: numpy array
    """
    n = int(n)
    mask = np.ones_like(data)
    ns, nf = data.shape
    mask[0:n, :] = 0
    mask[(ns - n):ns, :] = 0
    mask[:, 0:n] = 0
    mask[:, (nf - n):nf] = 0

    return mask


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
        self.shape = (m[0] + 1, m[1] + 1)

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

        data = np.ravel(data)

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

        return self.assemble_data(np.ravel(data_list))


class IcosphereGeometry(object):
    r"""

    Experimental class for a spherical detector that follows the "icosphere" geometry. The Icosphere is generated by
    sub-dividing the vertices of an icosahedron.  The following blog was helpful:
    http://sinestesia.co/blog/tutorials/python-icospheres/

    The code is quite slow; needs to be vectorized with numpy.  There are definitely better spherical detectors - the
    solid angles of these pixels are not very uniform.

    """

    n_subdivisions = 1
    radius = 1

    def __init__(self, n_subdivisions=1, radius=1):

        self.n_subdivisions = n_subdivisions
        self.radius = radius

    def _vertex(self, x, y, z):
        r""" Return vertex coordinates fixed to the unit sphere """

        length = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        return [(i * self.radius) / length for i in (x, y, z)]

    def _middle_point(self, point_1, point_2, verts, middle_point_cache):
        r""" Find a middle point and project to the unit sphere """

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
        r"""
        Compute vertex and face coordinates.  FIXME: Needs documentation.
        """

        # Make the base icosahedron

        vertex = self._vertex
        middle_point = self._middle_point
        middle_point_cache = {}

        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2

        verts = [
            vertex(-1, phi, 0),
            vertex(1, phi, 0),
            vertex(-1, -phi, 0),
            vertex(1, -phi, 0),

            vertex(0, -1, phi),
            vertex(0, 1, phi),
            vertex(0, -1, -phi),
            vertex(0, 1, -phi),

            vertex(phi, 0, -1),
            vertex(phi, 0, 1),
            vertex(-phi, 0, -1),
            vertex(-phi, 0, 1),
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
        n_faces = faces.shape[0]  # pylint:disable=unsubscriptable-object

        face_centers = np.zeros([n_faces, 3])
        for i in range(0, n_faces):
            face_centers[i, :] = (
                verts[faces[i, 0], :] + verts[faces[i, 1], :] + verts[faces[i, 2], :]) / 3

        return verts, faces, face_centers


class RadialProfiler(object):

    r"""
    Helper class to create radial profiles.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, q_mags=None, mask=None, n_bins=None, q_range=None):

        self.n_bins = None
        self.bins = None
        self.bin_edges = None
        self.bin_size = None
        self.bin_indices = None
        self.q_mags = None
        self.mask = None
        self.counts = None
        self.q_range = None
        self.counts_non_zero = None

        if (n_bins or q_range is not None) and q_mags is not None:
            self.make_plan(q_mags=q_mags, mask=mask, n_bins=n_bins, q_range=q_range)

    def make_plan(self, q_mags, mask=None, n_bins=None, q_range=None):
        r"""
        Setup the binning indices for the creation of radial profiles.

        Arguments:
            q_mags (numpy array) : Scattering vector magnitudes.
            mask (numpy array) : Pixel mask.  Should be ones and zeros, where one means "good" and zero means "bad".
            n_bins (int) : Number of bins.
            q_range (list-like) : The minimum and maximum of the scattering vector magnitudes.  The bin size will be
                                  equal to (max_q - min_q) / n_bins
        """
        q_mags = q_mags.ravel()
        if mask is None:
            mask = np.ones([len(q_mags)])
        else:
            mask = mask.copy().ravel()
        self.setup_bin_indices(q_mags, q_range=q_range, n_bins=n_bins)
        self.set_mask(mask)

    def setup_bin_indices(self, q_mags, n_bins=None, q_range=None):
        r"""
        Assign radial bin values to each of the q magnitudes.  We do this only once and store the values.  The binning
        convention is as described in the docs.

        Args:
            q_mags:
            n_bins:
            q_range:

        Returns:

        """
        q_mags = q_mags.ravel()

        if q_range is None:
            min_q = np.min(q_mags)
            max_q = np.max(q_mags)
            q_range = np.array([min_q, max_q])
        else:
            q_range = q_range.copy()
            min_q = q_range[0]
            max_q = q_range[1]

        bin_size = (max_q - min_q) / float(n_bins - 1)
        bins = min_q + np.arange(n_bins)*bin_size
        bin_edges = min_q - bin_size/2 + np.arange(n_bins+1)*bin_size
        bin_indices = np.int64(np.floor((q_mags - (min_q - bin_size/2)) / bin_size))
        # bin_index_groups = []  # A list of length n_bins, each of which contains a numpy array with the
        # for i in range(n_bins):


        # Overflow ends up in the min and max bins
        bin_indices[bin_indices < 0] = 0
        bin_indices[bin_indices >= n_bins] = n_bins - 1



        self.q_mags = q_mags
        self.n_bins = n_bins
        self.bins = bins
        self.bin_edges = bin_edges
        self.bin_size = bin_size
        self.bin_indices = bin_indices
        self.q_range = q_range

    def set_mask(self, mask):
        r"""
        Configure the mask.  This affects how averaging is done since masked pixels are ignored.

        Args:
            mask (numpy array) : The mask array.  Zero means ignore.
        """
        counts = np.bincount(self.bin_indices, mask.ravel(), self.n_bins)
        counts_non_zero = counts > 0
        self.mask = mask.copy().ravel()
        self.counts = counts
        self.counts_non_zero = counts_non_zero

    def get_profile(self, data, average=True):
        r"""
        Create a radial profile for a particular dataframe.

        Arguments:
            data (numpy array) : Intensity data.
            average (bool) : If true, divide the sum in each bin by the counts, else return the sum.  Default: True.

        Returns:
            numpy array : The requested radial profile.
        """
        profile = np.bincount(self.bin_indices, data.ravel() * self.mask, self.n_bins)
        if average:
            profile.flat[self.counts_non_zero] /= self.counts.flat[self.counts_non_zero]  # pylint:disable=no-member
        return profile

    # def get_profile_sum(self):
    #
    #
    #
    # def get_profile_mean(self):
    #
    # def get_profile_median(self):



