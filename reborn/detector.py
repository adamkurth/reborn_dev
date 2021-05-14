"""
Classes for analyzing/simulating diffraction data contained in pixel array
detectors (PADs).
"""

import os
import json
import numpy as np
from scipy.stats import binned_statistic
from . import utils
from time import time

class PADGeometry():
    r"""
    A container for pixel-array detector (PAD) geometry specification, with helpful methods for generating:

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
        On initialization, optional arguments may be provided (you must provide *all* of them):

        Arguments:
            shape (int or numpy array): Shape of the panels.  The first element is the slow-scan shape.  If there is
                                           only one element, or if it is an int, then it will be a square panel.
            distance (float): Sample-to-detector distance, where the beam is taken along the third ("Z") axis
            pixel_size (float): Size of the pixels in SI units.
        """

        if n_pixels is not None:
            utils.depreciate("When initializing PADGeometry, use the 'shape' argument not 'n_pixels'")

        self._n_fs = None
        self._n_ss = None
        self._fs_vec = None
        self._ss_vec = None
        self._t_vec = None
        self.name = None

        if distance is not None and pixel_size is not None:

            self.simple_setup(n_pixels=n_pixels, distance=distance, pixel_size=pixel_size, shape=shape)

    def __str__(self):
        out = ''
        out += 'n_fs: %s\n' % self.n_fs.__str__()
        out += 'n_ss: %s\n' % self.n_ss.__str__()
        out += 'fs_vec: %s\n' % self.fs_vec.__str__()
        out += 'ss_vec: %s\n' % self.ss_vec.__str__()
        out += 't_vec: %s' % self.t_vec.__str__()
        return out

    @property
    def hash(self):
        r"""Return a hash of the geometry parameters.  Useful if you want to avoid re-computing things like q_mags."""
        return hash(self.__str__())

    def __eq__(self, other):
        if not isinstance(other, PADGeometry):
            return False
        if not self.n_fs == other.n_fs:
            return False
        if not self.n_ss == other.n_ss:
            return False
        if np.max(np.abs(self.ss_vec - other.ss_vec)) > 0:
            return False
        if np.max(np.abs(self.fs_vec - other.fs_vec)) > 0:
            return False
        if np.max(np.abs(self.t_vec - other.t_vec)) > 0:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

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

    def to_dict(self):
        r""" Convert geometry to a dictionary.  It contains n_fs, n_ss, fs_vec, ss_vec and t_vec keys."""
        return {'n_fs': self.n_fs, 'n_ss': self.n_ss, 'fs_vec': tuple(self.fs_vec), 'ss_vec': tuple(self.ss_vec), 't_vec': tuple(self.t_vec)}

    def from_dict(self, dictionary):
        r""" Loads geometry from dictionary.  This goes along with the to_dict method."""
        self.n_fs = dictionary['n_fs']
        self.n_ss = dictionary['n_ss']
        self.fs_vec = dictionary['fs_vec']
        self.ss_vec = dictionary['ss_vec']
        self.t_vec = dictionary['t_vec']

    def copy(self):
        r""" Make a copy of this class instance. """
        p = PADGeometry()
        p.from_dict(self.to_dict())
        return p

    def save_json(self, file_name):
        r""" Save the geometry as a json file. """
        with open(file_name, 'w') as f:
            json.dump(self.to_dict(), f)

    def load_json(self, file_name):
        r""" Save the geometry as a json file. """
        with open(file_name, 'r') as f:
            d = json.load(f)
        self.from_dict(d)

    def simple_setup(self, n_pixels=None, pixel_size=None, distance=None, shape=None):
        r""" Make this a square PAD with beam at center.

        Arguments:
            shape : The shape of the panel, consistent with a numpy array shape.
            pixel_size : Pixel size in SI units.
            distance : Detector distance in SI units.
            n_pixels : (utils.depreciateD) Either the shape of the panel, or a single number if a square detector is desired.
                       Do not use this input as it will be removed in the future.

        Returns:
            object:
        """

        if n_pixels is not None:
            utils.depreciate('The redundant "n_pixels" keyword argument in simple_setup is utils.depreciated.  Use "shape" keyword '
                             'instead.')

        if pixel_size is None:
            utils.warn('Setting pixel_size in simple_setup to 100e-6.  You should specify this value explicitly.')
            pixel_size = 100e-6

        if distance is None:
            utils.warn('Setting distance in simple_setup to 0.1.  You should specify this value explicitly.')
            distance = 0.1

        if shape is not None:
            self.n_fs = shape[1]
            self.n_ss = shape[0]
        else:
            if n_pixels is None:
                utils.warn('Setting n_pixels in simple_setup to 1000.  You should specify this value explicitly.')
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

        return np.mean([utils.vec_mag(self.fs_vec), utils.vec_mag(self.ss_vec)])

    def shape(self):
        r""" Return tuple corresponding to the numpy shape of this PAD. """

        return self.n_ss, self.n_fs

    def indices_to_vectors(self, idx_ss, idx_fs):
        r"""
        Convert pixel indices to translation vectors pointing from origin to position on panel.
        The positions need not lie on the actual panel; this assums an infinite plane.

        Arguments:
            idx_fs (float) :
                Fast-scan index.
            idx_ss (float) :
                Slow-scan index.

        Returns:
            Nx3 numpy array
        """

        idx_fs = np.array(idx_fs)
        idx_ss = np.array(idx_ss)
        f_vec = np.outer(idx_fs.ravel(), self.fs_vec)
        s_vec = np.outer(idx_ss.ravel(), self.ss_vec)
        return self.t_vec + f_vec + s_vec

    def vectors_to_indices(self, vecs, insist_in_pad=True, round=False):
        r""" Suppose you have a vector pointing away from the origin and you want to know which pixel the vector
        will intercept (if scaled by an appropriate constant).  This function will do that calculation for you.  It will
        return the indices corresponding to the point where the vector intercepts the PAD.  Note that the indices are
        floating points, so you might need to convert to integers if you use them for indexing.

        Arguments:
            vecs (|ndarray|): An array of vectors, with shape (N, 3) or shape (3)
            insist_in_pad (bool): If you want to allow out-of-range indices, set this to True.  Otherwise, out-of-range
                                  values will be set to nan.

        Returns:
            slow-scan indices, fast-scan indices.
        """
        vecs = np.atleast_2d(vecs)
        fxs = np.dot(vecs, np.cross(self.ss_vec, self.fs_vec))
        i = np.dot(np.cross(self.ss_vec, vecs), self.t_vec)/fxs
        j = -np.dot(np.cross(self.fs_vec, vecs), self.t_vec)/fxs
        if round:
            i = np.round(i)
            j = np.round(j)
        if insist_in_pad:
            ii = i + 0.5
            jj = j + 0.5
            m = np.zeros(ii.shape, dtype=np.int64)
            m[ii < 0] = 1
            m[jj < 0] = 1
            m[ii > self.n_fs] = 1
            m[jj > self.n_ss] = 1
            i[m == 1] = np.nan
            j[m == 1] = np.nan
        return j, i

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

    def center_pos_vec(self):
        r"""
        The vector that points from the origin to the center of the PAD

        Returns: |ndarray|
        """
        return self.t_vec + (self.n_fs - 1) * self.fs_vec / 2.0 + (self.n_ss - 1) * self.ss_vec / 2.0

    def norm_vec(self):
        r"""
        The vector that is normal to the PAD plane.

        Returns: |ndarray|
        """

        return utils.vec_norm(np.cross(self.fs_vec, self.ss_vec))

    def s_vecs(self):
        r"""
        Outgoing unit-vectors (length 1) pointing from sample to pixel.

        Returns: |ndarray|
        """

        return utils.vec_norm(self.position_vecs())

    def ds_vecs(self, beam_vec=None, beam=None):
        r"""
        Scattering vectors :math:`\hat{s} - \hat{s}_0` where :math:`\hat{s}_0` is the incident beam direction
        and :math:`\hat{s}` is the outgoing vector pointing from sample to pixel.  This does **not** have
        the :math:`2\pi/\lambda` factor that is included in :meth:`q_mags <reborn.detector.PADGeometry.q_mags>`.

        Arguments:
            beam_vec (tuple or numpy array): specify the unit vector of the incident beam
            beam (|Beam| instance): specify incident beam properties.  If provided, you may omit the specification
                                         of beam_vec ect.

        Returns: numpy array
        """

        if beam is not None:
            beam_vec = beam.beam_vec

        return self.s_vecs() - beam_vec

    def q_vecs(self, beam_vec=None, wavelength=None, beam=None):
        r"""
        Calculate scattering vectors :math:`\frac{2\pi}{\lambda}(\hat{s} - \hat{s}_0)`

        .. math::

            \vec{q}_{ij}=\frac{2\pi}{\lambda}\left(\hat{v}_{ij} - \hat{b}\right)

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

    def ds_mags(self, beam_vec=None, beam=None):
        r"""
        These are the magnitudes that correspond to

        Arguments:
            beam_vec:
            beam:

        Returns:
        """
        return utils.vec_mag(self.ds_vecs(beam_vec=beam_vec, beam=beam))

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

        return utils.vec_mag(self.q_vecs(beam_vec=beam_vec, wavelength=wavelength))

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

        v_vec = self.position_vecs()
        n_vec = self.norm_vec()

        area = utils.vec_mag(np.cross(self.fs_vec, self.ss_vec))  # Area of the pixel
        dist2 = utils.vec_mag(v_vec) ** 2  # Distance to the pixel, squared
        inc = np.dot(n_vec, utils.vec_norm(v_vec).T)  # Inclination factor: cos(theta)
        solid_ang = (area / dist2) * inc  # Solid angle

        return np.abs(solid_ang.ravel())

    def solid_angles2(self):
        r"""
        Pixel solid angles calculated using the method of Van Oosterom, A. & Strackee, J. Biomed. Eng., IEEE T
        ransactions on BME-30, 125-126 (1983).  Divide each pixel up into two triangles with vertices R1,R2,R3 and
        R2,R3,R4. Then use analytical form to find the solid angle of each triangle. Sum them to get the solid angle of
        pixel.

        Thanks to Derek Mendez, who thanks Jonas Sellberg.

        Returns: numpy array
        """
        pixel_center = self.position_vecs()
        corner1 = pixel_center - self.fs_vec * .5 - self.ss_vec * .5
        corner2 = pixel_center + self.fs_vec * .5 - self.ss_vec * .5
        corner3 = pixel_center - self.fs_vec * .5 + self.ss_vec * .5
        corner4 = pixel_center + self.fs_vec * .5 + self.ss_vec * .5
        solid_angle_1 = utils.triangle_solid_angle(corner1, corner2, corner3)
        solid_angle_2 = utils.triangle_solid_angle(corner4, corner2, corner3)
        return solid_angle_1 + solid_angle_2

    def polarization_factors(self, polarization_vec_1=None, beam_vec=None, weight=None, beam=None):
        r"""
        The scattering polarization factors.

        Arguments:
            polarization_vec_1 (numpy array) :
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
            polarization_vec_1 = beam.polarization_vec
            weight = beam.polarization_weight
        pix_vec = utils.vec_norm(self.position_vecs())
        polarization_vec_1 = utils.vec_norm(np.array(polarization_vec_1))
        beam_vec = utils.vec_norm(np.array(beam_vec))
        polarization_vec_2 = np.cross(polarization_vec_1, beam_vec)
        if weight is None:
            weight1 = 1
            weight2 = 0
        else:
            weight1 = weight
            weight2 = 1 - weight
        polarization_factor = 0
        if weight1 > 0:
            polarization_factor += weight1 * (1 - np.abs(np.dot(pix_vec, polarization_vec_1)) ** 2)
        if weight2 > 0:
            polarization_factor += weight2 * (1 - np.abs(np.dot(pix_vec, polarization_vec_2)) ** 2)
        return polarization_factor.ravel()

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

        return np.arccos(utils.vec_norm(self.position_vecs()).dot(beam_vec.ravel()))

    def beamstop_mask(self, beam=None, q_min=None, min_angle=None):
        r"""

        Arguments:
            beam: Instance of the Beam class (for wavelength)
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

    def corner_position_vectors(self):
        t, f, s, nf, ns = self.t_vec, self.fs_vec, self.ss_vec, self.n_fs, self.n_ss
        return np.array([t, t+nf*f, t+nf*f+ns*s, t+ns*s])

def save_pad_geometry_list(file_name, geom_list):
    r""" Save a list of PADGeometry instances as a json file. """
    if not isinstance(geom_list, list):
        geom_list = [geom_list]
    with open(file_name, 'w') as f:
        json.dump([g.to_dict() for g in geom_list], f, sort_keys=True, indent=0)


def load_pad_geometry_list(file_name):
    r""" Load a list of PADGeometry instances stored in json format. """
    with open(file_name, 'r') as f:
        dicts = json.load(f)
    out = []
    for d in dicts:
        pad = PADGeometry()
        pad.from_dict(d)
        out.append(pad)
    return out


def tiled_pad_geometry_list(pad_shape=(512, 1024), pixel_size=100e-6, distance=0.1, tiling_shape=(4, 2), pad_gap=0):
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

    tilefs_sep = pad_shape[1] + pad_gap / pixel_size
    tilefs_pos = (np.arange(tiling_shape[1]) - (tiling_shape[1] - 1) / 2) * tilefs_sep
    tiless_sep = pad_shape[0] + pad_gap / pixel_size
    tiless_pos = (np.arange(tiling_shape[0]) - (tiling_shape[0] - 1) / 2) * tiless_sep
    for fs_cent in tilefs_pos:  # fast scan
        for ss_cent in tiless_pos:  # slow scan
            pad = PADGeometry(shape=pad_shape, pixel_size=pixel_size, distance=distance)
            pad.t_vec += pad.fs_vec * fs_cent
            pad.t_vec += pad.ss_vec * ss_cent
            # pad.t_vec[0:2] += 0.5 * pixel_size
            pads.append(pad)

    return pads


def concat_pad_data(data):
    r"""
    Given a list of numpy arrays, concatenate them into a single 1D array.  This is a very simple command:

    .. code-block:: python

        return np.concatenate([d.ravel() for d in data])

    This should exist in numpy but I couldn't find it.

    Arguments:
        data (list or numpy array): A list of 2D numpy arrays.  If data is a numpy array, then data.ravel() is returned

    Returns: 1D numpy array
    """

    if isinstance(data, np.ndarray):
        return data.ravel()

    return np.concatenate([d.ravel() for d in data])


def split_pad_data(pad_list, data):
    r"""
    Given a contiguous block of data produced by the function :func:`concat_pad_data <reborn.detector.concat_pad_data>`,
    split the data into individual 2D PAD panels.

    Arguments:
        pad_list: A list of PADGeometry instances
        data: A contiguous array with data values (total pixels to add up to sum of pixels in all PADs)

    Returns:
        A list of 2D numpy arrays

    """

    data_list = []

    offset = 0
    for pad in pad_list:
        data_list.append(pad.reshape(data[offset:(offset + pad.n_pixels)]))
        offset += pad.n_pixels

    return data_list


def edge_mask(data, n_edge):
    r"""
    Make an "edge mask"; an array of ones with zeros around the edges.
    The mask will be the same type as the data (e.g. double).

    Arguments:
        data (2D numpy array): a data array (for shape reference)
        n_edge (int): number of pixels to mask around edges

    Returns: numpy array
    """
    n_edge = int(n_edge)
    mask = np.ones_like(data)
    n_ss, n_fs = data.shape
    mask[0:n_edge, :] = 0
    mask[(n_ss - n_edge):n_ss, :] = 0
    mask[:, 0:n_edge] = 0
    mask[:, (n_fs - n_edge):n_fs] = 0

    return mask


def subtract_pad_friedel_mate(data, mask, pads):
    r""" Subtract the intensities related by Fridel symmetry"""
    data = utils.ensure_list(data)
    mask = utils.ensure_list(mask)
    pads = utils.ensure_list(pads)
    n_pads = len(pads)
    for i in range(n_pads):
        data[i] = (data[i].copy() * mask[i]).astype(np.float32)
    data_diff = [d.copy() for d in data]
    mask_diff = [p.zeros().astype(np.float32) for p in pads]
    for i in range(n_pads):
        vecs = pads[i].position_vecs()
        for j in range(n_pads):
            v = vecs.copy().astype(np.float32)
            v[:, 0:2] *= -1  # Invert vectors
            #del vecs
            x, y = pads[j].vectors_to_indices(v, insist_in_pad=True, round=True)
            del v
            w = np.where(np.isfinite(x))
            x = x[w]
            y = y[w]
            data_diff[i].flat[w] -= data[j][x.astype(int), y.astype(int)]
            mask_diff[i].flat[w] += np.abs(data[j][x.astype(int), y.astype(int)])
        mask_diff[i] *= mask[i]
    for i in range(n_pads):
        m = mask_diff[i]
        m[m > 0] = 1
        data_diff[i] *= m
    return data_diff

# class PADData(list):
#     r"""
#     A class for dealing with lists of PAD data.  Contains information about geometry.
#     """
#     _beam = None
#     _pad_data = None
#     _pad_geometry = None
#     _masks = None
#     def __init__(self, pad_data, pad_geometry, beam, masks=None):
#         self._pad_geometry = utils.ensure_list(pad_geometry)
#         if type(pad_data) == np.ndarray:
#             self._pad_data = split_pad_data(self._pad_geometry, pad_data)
#         else:
#             self._pad_data = utils.ensure_list(pad_data)
#         for d in self._pad_data:
#             self.append(d)
#         if masks is None:
#             mask = [p.ones() for p in self.pad_geometry]
#
#     def correct_polarization(self):
#         if not self._polarization_corrected:
#
#         self._polarization_corrected = True


class PADAssembler():

    r"""
    Assemble PAD data into a fake single-panel PAD.  This is done in a lazy way.  The resulting image is not
    centered in any way; the fake detector is a snug fit to the individual PADs.

    A list of PADGeometry objects are required on initialization, as the first argument.  The data needed to
    "interpolate" are cached, hence the need for a class.  The geometry cannot change; there is no update method.
    """

    def __init__(self, pad_list):
        pixel_size = utils.vec_mag(pad_list[0].fs_vec)
        position_vecs_concat = np.concatenate([p.position_vecs() for p in pad_list])
        position_vecs_concat -= np.min(position_vecs_concat, axis=0)
        position_vecs_concat /= pixel_size
        position_vecs_concat = np.floor(position_vecs_concat).astype(np.int64)
        maxval = np.max(position_vecs_concat, axis=0)
        assembled = np.zeros([maxval[0] + 1, maxval[1] + 1])
        self.position_vecs_concat = position_vecs_concat
        self.assembled = assembled
        self.shape = (maxval[0] + 1, maxval[1] + 1)

    def assemble_data(self, data):
        r"""
        Given a contiguous block of data, create the fake single-panel PAD.

        Arguments:
            data (numpy array): Image data

        Returns:
            assembled_data (numpy array): Assembled PAD image
        """
        data = np.ravel(data)
        assembled = self.assembled
        position_vecs_concat = self.position_vecs_concat
        assembled[position_vecs_concat[:, 0], position_vecs_concat[:, 1]] = data

        return assembled.copy()

    def assemble_data_list(self, data_list):
        r"""
        Same as assemble_data() method, but accepts a list of individual panels in the form of a list.

        Arguments:
            data_list (list of numpy arrays): Image data

        Returns:
            assembled_data (numpy array): Assembled PAD image
        """

        return self.assemble_data(np.ravel(data_list))


class IcosphereGeometry():
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

    def _vertex(self, x_coords, y_coords, z_coords):
        r""" Return vertex coordinates fixed to the unit sphere """

        length = np.sqrt(x_coords ** 2 + y_coords ** 2 + z_coords ** 2)

        return [(i * self.radius) / length for i in (x_coords, y_coords, z_coords)]

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
                pt1 = middle_point(tri[0], tri[1], verts, middle_point_cache)
                pt2 = middle_point(tri[1], tri[2], verts, middle_point_cache)
                pt3 = middle_point(tri[2], tri[0], verts, middle_point_cache)

                faces_subdiv.append([tri[0], pt1, pt3])
                faces_subdiv.append([tri[1], pt2, pt1])
                faces_subdiv.append([tri[2], pt3, pt2])
                faces_subdiv.append([pt1, pt2, pt3])

            faces = faces_subdiv

        faces = np.array(faces)
        verts = np.array(verts)
        n_faces = faces.shape[0]  # pylint:disable=unsubscriptable-object

        face_centers = np.zeros([n_faces, 3])
        for i in range(0, n_faces):
            face_centers[i, :] = (
                verts[faces[i, 0], :] + verts[faces[i, 1], :] + verts[faces[i, 2], :]) / 3

        return verts, faces, face_centers


class RadialProfiler():

    # pylint: disable=too-many-instance-attributes

    def __init__(self, q_mags=None, mask=None, n_bins=None, q_range=None, pad_geometry=None, beam=None):
        r"""
        A class for creating radial profiles from image data.  You must provide the number of bins and the q range that
        you desire for your radial profiles.  The q magnitudes that correspond to your diffraction patterns may be
        derived from a list of |PADGeometry|'s along with a |Beam|, or you may supply the q magnitudes directly.

        Arguments:
            q_mags (|ndarray|): Optional.  Array of q magnitudes.
            mask (|ndarray|): Optional.  The arrays will be multiplied by this mask, and the counts per radial bin
                                will come from this (e.g. use values of 0 and 1 if you want a normal average, otherwise
                                you get a weighted average).
            n_bins (int): Number of radial bins you desire.
            q_range (tuple): The minimum and maximum of the *centers* of the q bins.
            pad_geometry (list of |PADGeometry| instances):  Optional.  Will be used to generate q magnitudes.  You must
                                                             provide beam if you provide this.
            beam (|Beam| instance): Optional, unless pad_geometry is provided.  Wavelength and beam direction are
                                     needed in order to calculate q magnitudes.
        """
        self.n_bins = None  # Number of bins in radial profile
        self.q_range = None  # The range of q magnitudes in the 1D profile.  These correspond to bin centers
        self.q_edge_range = None  # Same as above, but corresponds to bin edges not centers
        self.bin_centers = None  # q magnitudes corresponding to 1D profile bin centers
        self.bin_edges = None  # q magnitudes corresponding to 1D profile bin edges (length is n_bins+1)
        self.bin_size = None  # The size of the 1D profile bin in q space
        self.q_mags = None  # q magnitudes corresponding to diffraction pattern intensities
        self._mask = None  # The default mask, in case no mask is provided upon requesting profiles
        self._counts_profile = None  # For speed, we cache the counts corresponding to the default _mask
        self.pad_geometry = None  # List of PADGeometry instances
        self.beam = None  # Beam instance for creating q magnitudes
        self._indices = None  # A list of indices for each radial bin
        self.make_plan(q_mags=q_mags, mask=mask, n_bins=n_bins, q_range=q_range, pad_geometry=pad_geometry, beam=beam)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is not None:
            mask = concat_pad_data(mask)
            if self._mask is not None:  # Check if we already have an identical mask
                if np.sum(np.abs(mask - self.mask)) == 0:
                    return
            self._mask = mask.copy()  # Ensure a properly flattened array
            self._counts_profile = None  # Wipe out this profile
            self._indices = None

    def set_mask(self, mask):
        self.mask = mask

    @property
    def counts_profile(self):
        r""" The number of pixels in each radial bin. """
        if self._counts_profile is None:
            if self.mask is None:
                self._counts_profile, _ = np.histogram(self.q_mags, bins=self.n_bins, range=self.q_edge_range)
            else:
                self._counts_profile, _ = np.histogram(self.q_mags, weights=self.mask, bins=self.n_bins,
                                                       range=self.q_edge_range)
        return self._counts_profile

    def make_plan(self, q_mags=None, mask=None, n_bins=None, q_range=None, pad_geometry=None, beam=None):
        r"""
        Setup the binning indices for the creation of radial profiles.

        Arguments:
            q_mags (|ndarray|): Optional.  Array of q magnitudes.
            mask (|ndarray|): Optional.  The arrays will be multiplied by this mask, and the counts per radial bin
                                will come from this (e.g. use values of 0 and 1 if you want a normal average, otherwise
                                you get a weighted average).
            n_bins (int): Number of radial bins you desire.
            q_range (tuple): The minimum and maximum of the *centers* of the q bins.
            pad_geometry (list of |PADGeometry| instances):  Optional.  Will be used to generate q magnitudes.  You must
                                                             provide beam if you provide this.
            beam (|Beam| instance): Optional, unless pad_geometry is provided.  Wavelength and beam direction are
                                     needed in order to calculate q magnitudes.
        """

        if q_mags is None:
            if pad_geometry is None:
                raise ValueError("You must provide a |PADGeometry| if q_mags are not provided in RadialProfiler")
            if beam is None:
                raise ValueError("You must provide a |Beam| if q_mags are not provided in RadialProfiler")
            pad_geometry = utils.ensure_list(pad_geometry)
            q_mags = [p.q_mags(beam=beam) for p in pad_geometry]
        q_mags = concat_pad_data(q_mags)
        if q_range is None:
            q_range = (0, np.max(q_mags))
        if n_bins is None:
            n_bins = int(np.sqrt(q_mags.size)/4.0)
        q_range = np.array(q_range)
        bin_size = (q_range[1] - q_range[0]) / float(n_bins - 1)
        bin_centers = np.linspace(q_range[0], q_range[1], n_bins)
        bin_edges = np.linspace(q_range[0] - bin_size / 2, q_range[1] + bin_size / 2, n_bins + 1)
        q_edge_range = np.array([q_range[0] - bin_size / 2, q_range[1] + bin_size / 2])
        self.q_mags = q_mags
        self.n_bins = n_bins
        self.bin_centers = bin_centers
        self.bin_edges = bin_edges
        self.bin_size = bin_size
        self.q_range = q_range
        self.q_edge_range = q_edge_range
        self.mask = mask
        self.pad_geometry = pad_geometry
        self._indices = None
        self.zeros = np.zeros(self.n_bins)

    def get_counts_profile(self, mask=None):
        if mask is not None:
            mask = concat_pad_data(mask)
            cntdat, _ = np.histogram(self.q_mags, weights=mask, bins=self.n_bins, range=self.q_edge_range)
            return cntdat
        return self.counts_profile

    def get_sum_profile(self, data, mask=None):
        r"""
        Calculate the radial profile of summed intensities.  This is divided by counts to get an average.

        Arguments:
            data |ndarray|:  The intensity data from which the radial profile is formed.
            mask |ndarray|:  Optional.  A mask to indicate bad pixels.  Zero is bad, one is good.

        Returns:  |ndarray|
        """
        data = concat_pad_data(data)
        if mask is not None:
            data *= concat_pad_data(mask)
        cntdat, _ = np.histogram(self.q_mags, weights=data, bins=self.n_bins, range=self.q_edge_range)
        return cntdat

    def get_mean_profile(self, data, mask=None):
        r"""
        Calculate the radial profile of averaged intensities.

        Arguments:
            data (|ndarray|):  The intensity data from which the radial profile is formed.
            mask (|ndarray|):  Optional.  A mask to indicate bad pixels.  Zero is bad, one is good.  If no mask is
                                 provided here, the mask configured with :meth:`set_mask` will be used.

        Returns: |ndarray|
        """

        if mask is None:
            mask = self.mask  # Use the default mask
        sumdat = self.get_sum_profile(data, mask=mask)
        if mask is not None:
            cntdat = self.get_sum_profile(mask)
        else:
            cntdat = self.counts_profile
        return np.divide(sumdat, cntdat, where=(cntdat > 0), out=self.zeros)

    def get_median_profile(self, data, mask=None):
        r"""
        Calculate the radial profile of averaged intensities.

        Arguments:
            data (|ndarray|):  The intensity data from which the radial profile is formed.
            mask (|ndarray|):  Optional.  A mask to indicate bad pixels.  Zero is bad, one is good.  If no mask is
                                 provided here, the mask configured with :meth:`set_mask` will be used.

        Returns:  |ndarray|
        """
        # t = time()
        data = concat_pad_data(data)
        q_mags = self.q_mags
        if mask is not None:
            self.mask = mask
        if self.mask is not None:
            w = np.where(self.mask)
            data = data[w]
            q_mags = q_mags[w]
        med = utils.binned_statistic(q_mags, data, np.median, self.n_bins, (self.bin_edges[0], self.bin_edges[-1]))
        # print(time()-t)
        return med

    def subtract_profile(self, data, mask=None, statistic='mean'):
        r"""
        Given some PAD data, subtract a radially averaged profile.

        Arguments:
            data:
            mask:
            type:

        Returns:

        """
        as_list = False
        if type(data) == list:
            as_list = True
        if statistic == 'mean':
            mprof = self.get_mean_profile(data, mask=mask)
        elif statistic == 'median':
            mprof = self.get_median_profile(data, mask=mask)
        else:
            raise ValueError('Statistic %s not recognized' % (statistic,))
        mprofq = self.bin_centers
        mpat = np.interp(self.q_mags, mprofq, mprof)
        mpat = concat_pad_data(mpat)
        data = concat_pad_data(data)
        data = data.copy()
        data -= mpat
        if as_list:
            data = split_pad_data(self.pad_geometry, data)
        return data

    def subtract_median_profile(self, data, mask=None):
        r"""
        Given some PAD data, calculate the radial median and subtract it from the data.

        Arguments:
            data:
            mask:

        Returns:

        """
        return self.subtract_profile(data, mask=mask, statistic='median')

    def get_profile(self, data, mask=None, average=True):
        r"""
        This method is depreciated.  Use get_mean_profile or get_sum_profile instead.
        """
        utils.depreciate("RadialProfiler.get_profile() is depreciated.  Use RadialProfiler.get_mean_profile().")
        if average is True:
            return self.get_mean_profile(data, mask=mask)
        return self.get_sum_profile(data, mask=mask)


def save_pad_masks(file_name, mask_list, packbits=True):
    r"""
    Save list of 2D mask arrays in a compressed format.  It is assumed that masks consist of values of zero or one.
    We presently use the :func:`numpy.packbits` function along with :func:`numpy.savez_compressed` function.

    .. note::
        The file name extension will be '.mask'.  If you provide a name without an extension, or with a different
        extension, *the extension will be changed*.  It is recommended that you explicitly provide the extension.

    Arguments:
        file_name (str): Path to file that will be saved.
        mask_list (list): A list of |ndarray| masks.  Will be converted to bool type before saving.
        packbits (bool): Specify if :func:`numpy.packbits` should be used to reduce file size.  (Default: True).

    Returns: str: File name
    """
    if not file_name.endswith('.mask'):
        file_name += '.mask'
    mask_list = utils.ensure_list(mask_list)
    if packbits:
        shapes = [np.array(m.shape).astype(int) for m in mask_list]
        masks = [np.packbits(m.ravel().astype(bool).astype(np.uint8)) for m in mask_list]
        np.savez_compressed(file_name, *shapes, *masks, format=1)
    else:
        np.savez_compressed(file_name, *mask_list, format=0)
    os.rename(file_name+'.npz', file_name)
    return file_name


def load_pad_masks(file_name):
    r"""
    Load a mask created using the save_pad_masks function.

    Arguments:
        file_name (str): The path to the file you want to open.

    Returns: List of |ndarray| objects with int type.
    """
    out = np.load(file_name)
    keys = list(out.keys())
    n = int(len(out) - 1)
    file_format = out['format']
    def _range(x): return np.arange(x, dtype=int)
    if file_format == 1:
        shapes = [out[keys[i]] for i in _range(n/2) + 1]
        masks = [out[keys[i]] for i in _range(n/2) + int(n/2) + 1]
        masks = [np.unpackbits(masks[i])[0:np.prod(shapes[i])].astype(int).reshape(shapes[i]) for i in _range(n/2)]
    else:
        masks = [out[keys[i]] for i in _range(n) + 1]
    return masks
