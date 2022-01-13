# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

"""
Classes for analyzing/simulating diffraction data contained in pixel array detectors (PADs).
"""

import os
import json
import numpy as np
import pkg_resources
from . import utils, source


pnccd_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/pnccd_front_geometry.json')
cspad_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/cspad_geometry.json')
cspad_2x2_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/cspad_2x2_geometry.json')
epix10k_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/epix10k_geometry.json')
mpccd_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/mpccd_geometry.json')
jungfrau4m_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/jungfrau4m_geometry.json')
rayonix_mx340_xfel_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/rayonix_mx340_xfel_geometry.json')


class PADGeometry:
    r"""
    A container for pixel-array detector (PAD) geometry specification.  By definition, a PAD consists of a single 2D
    grid of pixels; see the extended description in the :ref:`documentation <doc_detectors>`.

    .. note::

        A common point of confusion is that XFEL detectors typically consist of *multiple* PADs, in which case your
        code must handle *multiple* |PADGeometry| instances.  If that is the case for your data, then you should look to
        the |PADGeometryList| documentation as it extends Python's built-in list class with useful methods for
        PADGeometry instances.  Before you look to |PADGeometryList|, you should finish reading this documentation.

    The complete specification of an individual PAD geometry is understood to be the following 5 parameters, which must
    be defined for a proper instance of |PADGeometry|:

        * **n_fs**: The number of pixels along the fast-scan direction.
        * **n_ss**: The number of pixels along the slow-scan direction.
        * **t_vec**: The vector that points from the origin (interaction point) to the center of the first pixel in
          memory.
        * **fs_vec**: The vector that points from the first pixel in memory, to the next pixel in the fast-scan
          direction.
        * **ss_vec**: The vector that points from the first pixel in memory, to the next pixel in the slow-scan
          direction.

    In the above:

        * The lengths of the **fs_vec** and **ss_vec** vectors encode the size of the (possibly rectangular) pixel. They
          moreover form the *basis* of the 2D grid that maps the pixel positions in the 3D space of the measurement.
        * The term "fast-scan" corresponds to the right-most index of a 2D numpy |ndarray| containing PAD data.
        * The term "slow-scan" corresponds to the left-most index of a 2D |ndarray| containing PAD data.
        * In the default memory buffer layout of an |ndarray|, the fast-scan direction corresponds to pixels that are
          contiguous in memory, and which therefore have the smallest stride.  If the phrase "contiguous in memory" and
          the term "stride" does not mean anything to you, then you need to read the |numpy| documentation for
          |ndarray|.

    In addition to providing a standard way to specify PAD geometry, the PADGeometry class also provides methods
    that make it easy to generate:

        * Vectors from sample to pixel.
        * Scattering vectors (i.e. "q" vectors... provided beam information).
        * Scattering vector magnitudes.
        * Scattering angles (twice the Bragg angle).
        * Polarization factors.
        * Pixel solid angles.
        * Maximum resolution.
        * etc.

    Some of the above parameters require more than a PADGeometry instance -- they also require information about the
    x-ray beam.  The |Beam| class in reborn provides a standard way to specify the properties of an x-ray beam.

    Although PADGeometry is primarily meant to deal with *geometry*, you may also include the information needed to
    |slice| the PAD data from a parent data array (as of January 2022).  For example, data from the |CSPAD| detector is
    presented as a 3D |ndarray| when accessed using the LCLS |psana| python package.  In order to specify slicing,
    you must add the following parameters:

        * **parent_data_shape**: The shape of the parent data array (example: (32, 185, 392) ).
        * **parent_data_slice**: The slice of the parent data array (example: np.s_[4, :, 196:]).
    """
    # pylint: disable=too-many-public-methods
    # pylint: disable=too-many-instance-attributes
    # These are the configurable parameters.  No defaults.  One must think.
    _n_fs = None
    _n_ss = None
    _fs_vec = None
    _ss_vec = None
    _t_vec = None
    _name = ''
    _parent_data_slice = None  # Slice of parent data block
    _parent_data_shape = None  # Shape of parent data block

    def __init__(self, distance=None, pixel_size=None, shape=None, **kwargs):
        r"""
        On initialization, optional arguments may be provided (you must provide *all* of them):

        Arguments:
            shape (tuple): (optional) Shape of the PAD.
            distance (float): (optional) Sample-to-detector distance.
            pixel_size (float): (optional) Size of the pixels in SI units.
        """
        if [a for a in [distance, pixel_size, shape] if a is not None]:
            self.simple_setup(distance=distance, pixel_size=pixel_size, shape=shape, **kwargs)

    def __str__(self):
        out = ''
        out += '{\n'
        out += 'name: %s\n' % self.name.__str__()
        out += 'n_fs: %s\n' % self._n_fs.__str__()
        out += 'n_ss: %s\n' % self._n_ss.__str__()
        out += 'fs_vec: %s\n' % self._fs_vec.__str__()
        out += 'ss_vec: %s\n' % self._ss_vec.__str__()
        out += 't_vec: %s\n' % self._t_vec.__str__()
        out += 'parent_data_slice: %s\n' % self._parent_data_slice.__str__()
        out += 'parent_data_shape: %s\n' % self._parent_data_shape.__str__()
        out += '}\n'
        return out

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
        if self._parent_data_shape != other._parent_data_shape:
            return False
        if self._parent_data_slice != other._parent_data_slice:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def hash(self):
        r"""Return a hash of the geometry parameters.  Useful if you want to avoid re-computing things like q_mags."""
        return hash(self.__str__())

    def validate(self):
        r""" Determine if this instance has all the needed parameters defined.

        Returns:
            bool: True if validation passes.

        Raises:
            ValueError: If any of n_fs, n_ss, fs_vec, ss_vec, t_vec are not defined properly.
        """
        if not isinstance(self._n_fs, int):
            raise ValueError("The n_fs parameter is undefined in your PADGeometry instance.")
        if not isinstance(self._n_ss, int):
            raise ValueError("The n_ss parameter is undefined in your PADGeometry instance.")
        if not isinstance(self._fs_vec, np.ndarray):
            raise ValueError("The fs_vec parameter is undefined in your PADGeometry instance.")
        if not isinstance(self._ss_vec, np.ndarray):
            raise ValueError("The ss_vec parameter is undefined in your PADGeometry instance.")
        if not isinstance(self._t_vec, np.ndarray):
            raise ValueError("The t_vec parameter is undefined in your PADGeometry instance.")
        if (self._parent_data_slice is not None) and (self._parent_data_shape is None):
            raise ValueError("The parent data slice is defined but the parent data shape is undefined.")
        if (self._parent_data_shape is not None) and (self._parent_data_slice is None):
            raise ValueError("The parent data shape is defined but the parent data slice is undefined.")
        return True

    @property
    def name(self):
        r"""(*str*) The unique name of this panel. """
        return self._name

    @name.setter
    def name(self, val):
        self._name = str(val)

    @property
    def n_fs(self):
        r"""(*int*) Number of fast-scan pixels."""
        if self._n_fs is None:
            raise ValueError('n_fs has not been defined for this PADGeometry!')
        return self._n_fs

    @n_fs.setter
    def n_fs(self, val):
        self._n_fs = int(val)

    @property
    def n_ss(self):
        r"""Number of slow-scan pixels."""
        if self._n_ss is None:
            raise ValueError('n_ss has not been defined for this PADGeometry!')
        return self._n_ss

    @n_ss.setter
    def n_ss(self, val):
        self._n_ss = int(val)

    @property
    def n_pixels(self):
        r"""Total number of pixels (:math:`n_{fs} \cdot n_{ss}`)"""
        return self.n_fs * self.n_ss

    @property
    def fs_vec(self):
        r""" (|ndarray|) Fast-scan basis vector. """
        return self._fs_vec

    @property
    def ss_vec(self):
        r""" (|ndarray|) Slow-scan basis vector. """
        return self._ss_vec

    @property
    def t_vec(self):
        r""" (|ndarray|) Translation vector pointing from origin to center of corner pixel, which is first in memory."""
        return self._t_vec

    @fs_vec.setter
    def fs_vec(self, fs_vec):
        self._fs_vec = np.array(fs_vec).reshape((3,))
        if self._fs_vec.size != 3:
            raise ValueError('PADGeometry vectors should have a length of 3 (it is a vector)')

    @ss_vec.setter
    def ss_vec(self, ss_vec):
        self._ss_vec = np.array(ss_vec).reshape((3,))
        if self._ss_vec.size != 3:
            raise ValueError('PADGeometry vectors should have a length of 3 (it is a vector)')

    @t_vec.setter
    def t_vec(self, t_vec):
        self._t_vec = np.array(t_vec).reshape((3,))
        if self._t_vec.size != 3:
            raise ValueError('PADGeometry vectors should have a length of 3 (it is a vector)')

    @property
    def parent_data_slice(self):
        r""" Optionally, this defines the slice of an |ndarray| that this geometry corresponds to.  This is helpful
        if you wish to work with the 3D arrays in psana, for example. """
        return _tuple_to_slice(self._parent_data_slice)

    @parent_data_slice.setter
    def parent_data_slice(self, slc):
        self._parent_data_slice = _slice_to_tuple(slc)

    @property
    def parent_data_shape(self):
        r""" Optionally, this defines the shape of the |ndarray| from which this PAD is sliced. """
        return self._parent_data_shape

    @parent_data_shape.setter
    def parent_data_shape(self, shape):
        if isinstance(shape, list):
            shape = tuple(shape)
        if not (isinstance(shape, tuple) or (shape is None)):
            raise ValueError('parent_data_shape must be tuple or None')
        self._parent_data_shape = shape

    def slice_from_parent(self, data):
        r""" Slice this 2D array from the parent data array. """
        data = np.reshape(data, self._parent_data_shape)
        return self.reshape(data[self.parent_data_slice])

    def to_dict(self):
        r""" Convert geometry to a dictionary.

        Returns: (dict): Dictionary containing the keys **name**, **n_fs**, **n_ss**, **fs_vec**, **ss_vec**, **t_vec**,
                         **parent_data_shape**, and **parent_data_slice**.
        """
        return {'name': self.name, 'n_fs': self.n_fs, 'n_ss': self.n_ss, 'fs_vec': tuple(self.fs_vec),
                'ss_vec': tuple(self.ss_vec), 't_vec': tuple(self.t_vec), 'parent_data_shape': self.parent_data_shape,
                'parent_data_slice': _slice_to_tuple(self._parent_data_slice)}

    def from_dict(self, dictionary):
        r""" Loads geometry from dictionary.  This goes along with the to_dict method."""
        self.name = dict_default(dictionary, 'name', None)
        self.n_fs = dictionary['n_fs']
        self.n_ss = dictionary['n_ss']
        self.fs_vec = dictionary['fs_vec']
        self.ss_vec = dictionary['ss_vec']
        self.t_vec = dictionary['t_vec']
        self.parent_data_slice = dict_default(dictionary, 'parent_data_slice', None)
        self.parent_data_shape = dict_default(dictionary, 'parent_data_shape', None)

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

    def simple_setup(self, pixel_size=None, distance=None, shape=None, **kwargs):
        r""" Make this a square PAD with beam at center.

        Arguments:
            shape (tuple): The shape of the 2D panel.
            pixel_size (float): Pixel size in SI units.
            distance (float): Detector distance in SI units.
        """
        if len(kwargs) > 0:  # Deal with depreciated keywords
            if 'n_pixels' in kwargs:
                utils.depreciate('Use the "shape" keyword instead of "n_pixels" keyword.')
                n = int(kwargs['n_pixels'])
                del kwargs['n_pixels']
                shape = (n, n)
        if len(kwargs) > 0:
            raise ValueError('Keywords not recognized:' + '%s '*len(kwargs) % kwargs)
        if distance is None:
            utils.warn('Setting distance in simple_setup to 0.1.  You should specify this value explicitly.')
            distance = 0.1
        if pixel_size is None:
            utils.warn('Setting pixel_size in simple_setup to 100e-6.  You should specify this value explicitly.')
            pixel_size = 100e-6
        if shape is None:
            utils.warn('Setting shape to (1000, 1000).  You should specify this value explicitly.')
            shape = (1000, 1000)
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError('A PADGeometry shape must have exactly two elements.')
        self.n_fs = shape[1]
        self.n_ss = shape[0]
        self.fs_vec = np.array([pixel_size, 0, 0])
        self.ss_vec = np.array([0, pixel_size, 0])
        self.t_vec = np.array([pixel_size * -(self.n_fs / 2.0 - 0.5), pixel_size * -(self.n_ss / 2.0 - 0.5), distance])

    def pixel_size(self):
        r""" Return pixel size, assuming square pixels. """
        return np.mean([utils.vec_mag(self.fs_vec), utils.vec_mag(self.ss_vec)])

    def shape(self):
        r""" Return tuple corresponding to the |ndarray| shape of this PAD. """
        return self.n_ss, self.n_fs

    def indices_to_vectors(self, idx_ss, idx_fs):
        r"""
        Convert pixel indices to translation vectors pointing from origin to position on panel.
        The positions need not lie on the actual panel; this assums an infinite plane.

        Arguments:
            idx_fs (float) : Fast-scan index.
            idx_ss (float) : Slow-scan index.

        Returns:
            |ndarray| : Nx3 vector array.
        """
        idx_fs = np.array(idx_fs)
        idx_ss = np.array(idx_ss)
        f_vec = np.outer(idx_fs.ravel(), self.fs_vec)
        s_vec = np.outer(idx_ss.ravel(), self.ss_vec)
        return self.t_vec + f_vec + s_vec

    def vectors_to_indices(self, vecs, insist_in_pad=True, round_to_nearest=False, **kwargs):
        r""" Suppose you have a vector pointing away from the origin and you want to know which pixel the vector will
        intercept.  This function will do that calculation for you.  It will return the indices corresponding to the
        point where the vector intercepts the PAD.  Note that the indices are floating points, so you might need to
        convert to integers if you use them for indexing.

        Arguments:
            vecs (|ndarray|): An array of vectors, with shape (N, 3) or shape (3)
            insist_in_pad (bool): If you want to allow out-of-range indices, set this to True.  Otherwise, out-of-range
                                  values will be set to nan.
            round_to_nearest (bool): Round to nearest pixel position.  Default: False.

        Returns:
            (tuple) : Slow-scan indices, Fast-scan indices.
        """
        if "round" in kwargs.keys():  # Legacy keyword agument
            round_to_nearest = kwargs["round"]
            del kwargs["round"]
        vecs = np.atleast_2d(vecs)
        fxs = np.dot(vecs, np.cross(self.ss_vec, self.fs_vec))
        i = np.dot(np.cross(self.ss_vec, vecs), self.t_vec)/fxs
        j = -np.dot(np.cross(self.fs_vec, vecs), self.t_vec)/fxs
        if round_to_nearest:
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

        Returns: |ndarray| of shape (N, 3)
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

    def ds_vecs(self, beam=None, **kwargs):
        r"""
        Scattering vectors :math:`\hat{s} - \hat{s}_0` where :math:`\hat{s}_0` is the incident beam direction
        and :math:`\hat{s}` is the outgoing vector pointing from sample to pixel.  This does **not** have
        the :math:`2\pi/\lambda` factor that is included in :meth:`q_mags <reborn.detector.PADGeometry.q_mags>`.

        Arguments:
            beam (|Beam|): specify incident beam properties.  If provided, you may omit the specification
                                         of beam_vec ect.

        Returns: |ndarray|
        """
        if beam is None:
            utils.warn('You need to define the beam.')
            beam_vec = dict_default(kwargs, 'beam_vec', None)
            beam = source.Beam(beam_vec=beam_vec)
        return self.s_vecs() - beam.beam_vec

    def q_vecs(self, beam=None, **kwargs):
        r"""
        Calculate scattering vectors :math:`\frac{2\pi}{\lambda}(\hat{s} - \hat{s}_0)`

        .. math::

            \vec{q}_{ij}=\frac{2\pi}{\lambda}\left(\hat{v}_{ij} - \hat{b}\right)

        Arguments:
            beam (source.Beam instance): specify incident beam properties.  If provided, you may omit the specification
                                         of beam_vec ect.

        Returns: |ndarray|
        """
        if beam is None:
            utils.warn('You need to define the beam.')
            beam_vec = dict_default(kwargs, 'beam_vec', None)
            wavelength = dict_default(kwargs, 'wavelength', None)
            beam = source.Beam(beam_vec=beam_vec, wavelength=wavelength)
        return (2 * np.pi / beam.wavelength) * self.ds_vecs(beam=beam)

    def ds_mags(self, beam=None, **kwargs):
        r"""
        These are the magnitudes that correspond to

        Arguments:
            beam (|Beam|):

        Returns: |ndarray|
        """
        if beam is None:
            utils.warn('You need to define the beam.')
            beam_vec = dict_default(kwargs, 'beam_vec', None)
            beam = source.Beam(beam_vec=beam_vec)
        return utils.vec_mag(self.ds_vecs(beam=beam))

    def q_mags(self, beam=None, **kwargs):
        r"""
        Calculate scattering vector magnitudes:

        Arguments:
            beam (source.Beam instance): specify incident beam properties.  If provided, you may omit the specification
                                        of beam_vec ect.

        Returns: |ndarray|
        """
        if beam is None:
            utils.warn('You need to define the beam.')
            beam_vec = dict_default(kwargs, 'beam_vec', None)
            wavelength = dict_default(kwargs, 'wavelength', None)
            beam = source.Beam(beam_vec=beam_vec, wavelength=wavelength)
        return utils.vec_mag(self.q_vecs(beam=beam))

    def solid_angles(self):
        r"""
        Calculate solid angles of pixels.  See solid_angles2 method.

        Returns: |ndarray|
        """
        return self.solid_angles1()

    def solid_angles1(self):
        r"""
        Calculate solid angles of pixels vectorally, assuming the pixels have small angular extent.

        Returns: |ndarray|
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

        Returns: |ndarray|
        """
        pixel_center = self.position_vecs()
        corner1 = pixel_center - self.fs_vec * .5 - self.ss_vec * .5
        corner2 = pixel_center + self.fs_vec * .5 - self.ss_vec * .5
        corner3 = pixel_center - self.fs_vec * .5 + self.ss_vec * .5
        corner4 = pixel_center + self.fs_vec * .5 + self.ss_vec * .5
        solid_angle_1 = utils.triangle_solid_angle(corner1, corner2, corner3)
        solid_angle_2 = utils.triangle_solid_angle(corner4, corner2, corner3)
        return solid_angle_1 + solid_angle_2

    def polarization_factors(self, beam=None, e1=None, b=None, a=None):
        r"""
        The scattering polarization factors.

        Arguments:
            beam (|Beam|): Incident beam.
            e1 (|ndarray|) : Optional: Principle polarization vector.
            b (|ndarray|) : Optional: Incident beam vector.
            a (float) : Optional: The weight of the first polarization component.

        Returns: |ndarray|
        """
        if beam is not None:
            b = beam.beam_vec
            e1 = beam.e1_vec
            a = beam.polarization_weight
        pix_vec = utils.vec_norm(self.position_vecs())
        e1 = utils.vec_norm(np.array(e1))
        b = utils.vec_norm(np.array(b))
        polarization_vec_2 = np.cross(e1, b)
        if a is None:
            weight1 = 1
            weight2 = 0
        else:
            weight1 = a
            weight2 = 1 - a
        polarization_factor = np.zeros(self.n_pixels)
        if weight1 > 0:
            polarization_factor += weight1 * (1 - np.abs(np.dot(pix_vec, e1)) ** 2)
        if weight2 > 0:
            polarization_factor += weight2 * (1 - np.abs(np.dot(pix_vec, polarization_vec_2)) ** 2)
        return polarization_factor.ravel()

    def scattering_angles(self, beam=None, **kwargs):
        r"""
        Scattering angles (i.e. twice the Bragg angles).

        Arguments:
            beam (source.Beam instance): specify incident beam properties.  If provided, you may omit the specification
                                         of beam_vec ect.

        Returns: |ndarray|
        """
        if beam is None:
            utils.warn('You need to define the beam.')
            beam_vec = dict_default(kwargs, 'beam_vec', None)
            beam = source.Beam(beam_vec=beam_vec)
        return np.arccos(utils.vec_norm(self.position_vecs()).dot(beam.beam_vec.ravel()))

    def azimuthal_angles(self, beam):
        r"""
        The azimuthal angles of pixels in |spherical_coordinates|.  In the physics convention, the incident beam points
        along the zenith :math:`\hat{z}`, the outgoing wavevector points to the pixel at position :math:`\vec{r}`, the
        "polar angle" :math:`\theta` is the scattering angle, and the "azimuthal angle" is :math:`\phi = \arctan(y/x)`.

        Since reborn does not enforce any particular coordinate convention or beam direction, we define the azimuthal
        angles according to the definition of the incident |Beam| :

        .. math::

            \phi = \arctan(\hat{e}_2 \cdot \hat{r} / \hat{e}_1 \cdot \hat{r})

        where :math:`\hat{e}_1` is the principle polarization component of the incident x-ray beam, and
        :math:`\hat{e}_2` is the complementary polarization component.

        Arguments:
            beam (source.Beam instance): specify incident beam properties.

        Returns: |ndarray|
        """
        q_vecs = self.q_vecs(beam=beam)
        q1 = np.dot(q_vecs, beam.e1_vec)
        q2 = np.dot(q_vecs, beam.e2_vec)
        return np.arctan2(q2, q1)

    def beamstop_mask(self, beam=None, q_min=None, min_angle=None):
        r"""

        Arguments:
            beam (|Beam|): Instance of the Beam class (for wavelength)
            q_min: Minimum q magnitude
            min_angle: Minimum scattering angle

        Returns: |ndarray|
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
            dat (|ndarray|): The flattened data array

        Returns: |ndarray|
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
        r"""
        Returns the coordinates of all four corners of the detector.  The output is an |ndarray| with shape (5, 3) and
        the following entries [t, t+nf*f, t+nf*f+ns*s, t+ns*s] .

        Returns:
            |ndarray| : The corner positions of the PAD.
        """
        t, f, s, nf, ns = self.t_vec, self.fs_vec, self.ss_vec, self.n_fs, self.n_ss
        return np.array([t, t+nf*f, t+nf*f+ns*s, t+ns*s])

    def binned(self, binning=2):
        r"""
        Make the pixel size bigger by an integer multiple, while keeping the array size approximately the same.

        Note:
            This may result in loss of information.  Example: with binning set to 2, a 5x7 PAD results in a 2x3 PAD
            with pixels twice the size.  There is no way to recover the initial 5x7 shape from the binned PAD.

        Note:
            This operation is not inplace.  It does not affect the current instance of PADGeometry.  It returns a new
            PADGeometry.

        Args:
            binning (int): An integer value of 1,2,3,etc.  The pixel size will be increased by this factor.

        Returns:
            |PADGeometry| : The new, binned, PAD geometry.
        """
        if not isinstance(binning, int):
            raise ValueError('binning should be an integer')
        p = self.copy()
        p.n_fs = np.int(p.n_fs / binning)
        p.n_ss = np.int(p.n_ss / binning)
        p.t_vec += (p.fs_vec + p.ss_vec) * (binning - 1) / 2
        p.fs_vec *= binning
        p.ss_vec *= binning
        return p

    def translate(self, vec):
        r""" Translate the geometry.  Equivalent to self.t_vec += vec. """
        self.t_vec += vec

    def rotate(self, matrix=None):
        r""" Apply a rotation matrix to t_vec, fs_vec, ss_vec.
        Equivalent to self.t_vec = np.dot(self.t_vec, matrix.T)"""
        self.t_vec = np.dot(self.t_vec, matrix.T)
        self.fs_vec = np.dot(self.fs_vec, matrix.T)
        self.ss_vec = np.dot(self.ss_vec, matrix.T)
        

class PADGeometryList(list):
    r""" A subclass of list that does operations on lists of |PADGeometry| instances.  Is helpful, for example.
    when getting q vectors for many separate PADs.
    """

    _name = ''
    _q_mags = None
    _groups = []

    def __init__(self, pad_geometry=None, filepath=None):
        r"""
        Arguments:
            pad_geometry (|PADGeometry| or list of): The PAD geometry that will form the PADGeometryList.
        """
        super().__init__()
        if pad_geometry is not None:
            pad_geometry = utils.ensure_list(pad_geometry)
            for p in pad_geometry:
                self.append(p)
        if filepath is not None:
            self.load(filepath)

    def append(self, item):
        r""" Override append method.  Check the type, name the panel if it has no name. """
        if not isinstance(item, PADGeometry):
            raise ValueError('Not a PADGeometry instance.')
        if not item.name:
            item.name = len(self).__str__()
        super().append(item)

    def copy(self):
        r""" Same as the the matching method in |PADGeometry|."""
        return PADGeometryList([p.copy() for p in self])

    def __str__(self):
        s = ''
        for item in self:
            s += '\n'+item.__str__()
        return s

    @property
    def hash(self):
        r"""Return a hash of the geometry parameters.  Useful if you want to avoid re-computing things like q_mags."""
        s = ''
        for p in self:
            s += p.__str__()
        return hash(s)

    def validate(self):
        r""" Same as the matching method in |PADGeometry|."""
        status = True
        if self.defines_slicing():
            p0 = self[0]
            for p in self:
                if p.parent_data_shape != p0.parent_data_shape:
                    raise ValueError('Mismatched parent data shape')
        for p in self:
            status *= p.validate()
        if status:
            return True
        return False

    def save(self, filename):
        r""" Save this PADGeometryList in default json format. """
        self.save_json(filename)

    def load(self, filename):
        r""" Load the data from saved PADGeometryList. """
        try:
            pads = load_pad_geometry_list(filename)
        except:
            try:
                from .external import crystfel
                pads = crystfel.geometry_file_to_pad_geometry_list(filename)
            except:
                raise ValueError("Cannot figure out what kind of geometry file this is.")
        if len(self) == 0:
            for p in pads:
                self.append(p)
            return
        elif len(self) == len(pads):
            for (n, p) in enumerate(pads):
                self[n] = p
            return
        raise ValueError("It is not clear what you are trying to do.  The PADGeometryList should be empty, or should"
                         "have the same length as the geometry file.")

    def add_group(self, pads, group_name=None):
        r""" Add a group of PADGeometry instances.  Helpful if you have multiple "detectors" that have different
        properties as compared with others.  Or perhaps there are PADs that should translate as a group."""
        if group_name is None:
            group_name = str(len(self._groups))
        if group_name in self.get_group_names():
            raise ValueError('Group name', group_name, 'already exists.')
        indices = []
        for p in pads:
            indices.append(len(self))
            self.append(p)
        self._groups.append({'name': group_name, 'indices': indices})

    def get_group_indices(self, group_name):
        r""" Get the list indices for a named group. """
        indices = None
        for g in self._groups:
            if group_name == g['name']:
                indices = g['indices']
        if indices is None:
            raise ValueError('No group named', group_name)
        return indices

    def get_group(self, group_name):
        r""" Return a named group in the form of a |PADGeometryList| ."""
        for g in self._groups:
            if g['name'] == group_name:
                return PADGeometryList([self[i] for i in g['indices']])
        raise ValueError('No group named', group_name)

    def get_all_groups(self):
        r""" Equivalent to get_group, but sets the argument to all group names.  Beware: you may have redundancies!"""
        groups = []
        for g in self._groups:
            groups.append(self.get_group(g['name']))
        return groups

    def get_group_names(self):
        r""" Get a list of all group names.  Will be empty list if there are no groups.  """
        names = []
        for g in self._groups:
            names.append(g['name'])
        return names

    def get_by_name(self, name):
        r""" Return a |PADGeometry| with a given name. """
        pad = None
        for p in self:
            if p.name == name:
                if pad is not None:
                    raise ValueError('Ambiguous; more than one pad with the same name!')
                pad = p
        if pad is None:
            raise ValueError('No PAD named', name)
        return pad

    def defines_slicing(self):
        r""" False if any of the |PADGeometry| instances does not have a parent_data_slice or parent_data_shape
        defined.  True otherwise. """
        if (None in [p.parent_data_slice for p in self]) or (None in [p.parent_data_shape for p in self]):
            return False
        return True

    @property
    def parent_data_shape(self):
        r""" Return parent data shape, or None if undefined.  Raise ValueError if mis-matched parent data shapes."""
        if False in [(self[0].parent_data_shape == s.parent_data_shape) for s in self]:
            raise ValueError("Your PADGeometry instances have different parent data shapes!")
        return self[0].parent_data_shape

    def reshape(self, data):
        r""" If parent_data_shape is defined, then reshape the data to that shape. """
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be an ndarray')
        shape = self.parent_data_shape
        if shape is None:
            dr = data.ravel()
            if len(dr) != self.n_pixels:
                raise ValueError('Data length does not match this PADGeometryList.n_pixels')
            return data.ravel()
        return np.reshape(data, shape)

    def _split_data(self, data):
        r""" Split a contiguous 1D |ndarray| into list of 2D |ndarray| instances."""
        if self.defines_slicing():
            self.validate()
            datalist = []
            d = np.reshape(data, self.parent_data_shape)
            for p in self:
                datalist.append(d[p.parent_data_slice])
            return datalist
        return split_pad_data(self, data)

    def concat_data(self, data):
        r""" Concatenate a list of |ndarray| instances into a single concatenated 1D |ndarray| ."""
        if isinstance(data, list):
            if len(data) != len(self):
                raise ValueError("Length of data list is not the same length as the PADGeometryList")
        if self.defines_slicing():
            self.validate()
            if isinstance(data, np.ndarray):
                return data.ravel()
            if isinstance(data, list):
                datacat = np.zeros(self[0].parent_data_shape, dtype=data[0].dtype)
                for (p, d) in zip(self, data):
                    datacat[p.parent_data_slice] = d
                return datacat.ravel()
        return concat_pad_data(data)

    def split_data(self, data):
        r""" Slice this PAD data from a parent data array. """
        if not self.defines_slicing():
            return self._split_data(data)
        return [p.slice_from_parent(data) for p in self]

    @property
    def n_pixels(self):
        r""" Sums the output of the matching method in |PADGeometry|"""
        return np.sum(np.array([p.n_pixels for p in self]))

    def save_json(self, file_name):
        r""" Same as the matching method in |PADGeometry|."""
        save_pad_geometry_list(file_name, self)

    def position_vecs(self):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.position_vecs().ravel() for p in self]).reshape([self.n_pixels, 3])

    def s_vecs(self):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.s_vecs().ravel() for p in self]).reshape([self.n_pixels, 3])

    def ds_vecs(self, beam):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.ds_vecs(beam=beam).ravel() for p in self]).reshape([self.n_pixels, 3])

    def q_vecs(self, beam):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.q_vecs(beam=beam).ravel() for p in self]).reshape([self.n_pixels, 3])

    def q_mags(self, beam):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.q_mags(beam=beam).ravel() for p in self])

    def solid_angles(self):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.solid_angles().ravel() for p in self])

    def solid_angles1(self):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.solid_angles1().ravel() for p in self])

    def solid_angles2(self):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.solid_angles2().ravel() for p in self])

    def polarization_factors(self, beam):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.polarization_factors(beam=beam).ravel() for p in self])

    def scattering_angles(self, beam):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.scattering_angles(beam=beam).ravel() for p in self])

    def azimuthal_angles(self, beam):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.azimuthal_angles(beam).ravel() for p in self])

    def beamstop_mask(self, beam=None, q_min=None, min_angle=None):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.beamstop_mask(beam, q_min, min_angle).ravel() for p in self])

    def zeros(self):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.zeros().ravel() for p in self])

    def ones(self):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.ones().ravel() for p in self])

    def random(self):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.concatenate([p.random().ravel() for p in self])

    def max_resolution(self, beam):
        r""" Concatenates the output of the matching method in |PADGeometry|"""
        return np.max(np.array([p.max_resolution(beam=beam) for p in self]))

    def binned(self, binning=2):
        r""" See corresponding method in |PADGeometry|. """
        binned = [p.binned(binning) for p in self]
        return PADGeometryList(binned)

    def translate(self, vec):
        r""" See corresponding method in |PADGeometry|. """
        for p in self:
            p.translate(vec)

    def rotate(self, matrix):
        r""" See corresponding method in |PADGeometry|. """
        for p in self:
            p.rotate(matrix)


def f2_to_photon_counts(f_squared, beam=None, pad_geometry=None):
    r"""
    Convert computed scattering factors :math:`F(\vec{q})^2` into photon counts.  This multiplies :math:`F(\vec{q})^2`
    by the incident beam fluence, the classical electron area, the pixel solid angles, and the beam polarization
    factor.

    Args:
        f_squared:
        beam:
        pad_geometry:

    Returns:

    """
    SA = pad_geometry.solid_angles()
    P = pad_geometry.polarization_factors(beam=beam)
    return f_squared * 2.18e-15 ** 2 * SA * P * beam.photon_number_fluence


def save_pad_geometry_list(file_name, geom_list):
    r""" Save a list of PADGeometry instances as a json file. """
    if not isinstance(geom_list, list):
        geom_list = [geom_list]
    with open(file_name, 'w') as f:
        json.dump([g.to_dict() for g in geom_list], f, sort_keys=True, indent=0)


def load_pad_geometry_list(file_name):
    r""" Load a list of PADGeometry instances stored in json format. """
    if file_name == '_pnccd':
        return None
    with open(file_name, 'r') as f:
        dicts = json.load(f)
    out = []
    for d in dicts:
        pad = PADGeometry()
        pad.from_dict(d)
        out.append(pad)
    out = PADGeometryList(out)
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

    Returns: |PADGeometryList|
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
    return PADGeometryList(pads)


def concat_pad_data(data):
    r"""
    Given a list of numpy arrays, concatenate them into a single 1D array.  This is a very simple command:

    .. code-block:: python

        return np.concatenate([d.ravel() for d in data])

    This should exist in numpy but I couldn't find it.

    Arguments:
        data (list or |ndarray|): A list of 2D |ndarray| s.  If data is an |ndarray|, then data.ravel() is returned

    Returns: 1D |ndarray|
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
        A list of 2D |ndarray| s
    """
    if isinstance(data, list):
        return data
    data_list = []
    offset = 0
    data = data.ravel()
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

    Returns: |ndarray|
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


class PADAssembler:
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
            data (|ndarray|): Image data

        Returns: |ndarray|
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

        Returns: |ndarray|
        """
        return self.assemble_data(np.ravel(data_list))


class IcosphereGeometry:
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


class PolarPADAssembler:
    r""" A class for converting PAD data to polar coordinates. """
    def __init__(self, pad_geometry=None, beam=None, n_q_bins=50, q_range=None, n_phi_bins=None, phi_range=None):
        r"""
        Arguments:
            pad_geometry (|PADGeometryList|): PAD Geometry.
            beam (|Beam|): Beam information.
            n_q_bins (int): Number of q bins.
            q_range (tuple): Minimum and maximum q bin centers.  If None, the range is [0, maximum q in PAD].
            n_phi_bins (int): Number of phi bins.
            phi_range (tuple): Minimum and maximum phi bin centers.  If None, the full 2*pi ring is assumed.
        """
        q_mags = pad_geometry.q_mags(beam=beam)
        if q_range is None:
            q_range = [0, np.max(q_mags)]
        q_bin_size = (q_range[1] - q_range[0]) / float(n_q_bins - 1)
        q_centers = np.linspace(q_range[0], q_range[1], n_q_bins)
        q_edges = np.linspace(q_range[0] - q_bin_size / 2, q_range[1] + q_bin_size / 2, n_q_bins + 1)
        q_min = q_edges[0]
        if phi_range is None:  # Then we go from 0 to 2pi...
            phi_bin_size = 2 * np.pi / n_phi_bins
            phi_range = [phi_bin_size / 2, 2 * np.pi - phi_bin_size / 2]
        else:
            phi_bin_size = (phi_range[1] - phi_range[0]) / float(n_phi_bins - 1)
        phi_centers = np.linspace(phi_range[0], phi_range[1], n_phi_bins)
        phi_edges = np.linspace(phi_range[0] - phi_bin_size / 2, phi_range[1] + phi_bin_size / 2, n_phi_bins + 1)
        phi_min = phi_edges[0]
        self.q_bin_size = q_bin_size
        self.q_bin_centers = q_centers
        self.q_bin_edges = q_edges
        self.n_q_bins = n_q_bins
        self.q_min = q_min
        self.phi_bin_size = phi_bin_size
        self.phi_bin_centers = phi_centers
        self.phi_bin_edges = phi_edges
        self.n_phi_bins = n_phi_bins
        self.phi_min = phi_min
        self.q_mags = q_mags
        self.phis = pad_geometry.azimuthal_angles(beam=beam)

    def get_mean(self, data, mask=None):
        r""" Create the mean polar-binned average intensities.
        Arguments:
            data (list or |ndarray|): The PAD data to be binned.
            mask (list or |ndarray|): A mask to indicate ignored pixels.
        """
        # TODO: Implement mask
        # TODO: Speed up the algorithm.  Fortran?
        data = concat_pad_data(data)
        n_q = self.n_q_bins
        q_size = self.q_bin_size
        n_phi = self.n_phi_bins
        phi_size = self.phi_bin_size
        q_min = self.q_min
        phi_min = self.phi_min
        q = self.q_mags
        phi = self.phis
        sum_ = np.zeros([n_q, n_phi])
        cnt = np.zeros([n_q, n_phi], dtype=int)
        for i in range(len(data)):
            qi = q[i]
            pi = phi[i] % (2 * np.pi)
            vi = data[i]
            q_ind = int(np.floor((qi - q_min) / q_size))
            if q_ind >= n_q:
                continue
            if q_ind < 0:
                continue
            p_ind = int(np.floor((pi - phi_min) / phi_size))
            if p_ind >= n_phi:
                continue
            if p_ind < 0:
                continue
            cnt[q_ind, p_ind] += 1
            sum_[q_ind, p_ind] += vi
        mean_ = np.divide(sum_, cnt, out=np.zeros_like(sum_), where=cnt != 0)
        return mean_

    def get_sdev(self, data, mask=None):
        r""" Create polar-binned standard deviation.  Not implemented yet."""
        # TODO: Implement this once it is needed.
        raise NotImplementedError('Time to implement this method!')


class RadialProfiler:
    r"""
    A class for creating radial profiles from image data.  You must provide the number of bins and the q range that
    you desire for your radial profiles.  The q magnitudes that correspond to your diffraction patterns may be
    derived from a list of |PADGeometry|'s along with a |Beam|, or you may supply the q magnitudes directly.
    """
    # pylint: disable=too-many-instance-attributes
    n_bins = None  # Number of bins in radial profile
    q_range = None  # The range of q magnitudes in the 1D profile.  These correspond to bin centers
    q_edge_range = None  # Same as above, but corresponds to bin edges not centers
    bin_centers = None  # q magnitudes corresponding to 1D profile bin centers
    bin_edges = None  # q magnitudes corresponding to 1D profile bin edges (length is n_bins+1)
    bin_size = None  # The size of the 1D profile bin in q space
    _q_mags = None  # q magnitudes corresponding to diffraction pattern intensities
    _mask = None  # The default mask, in case no mask is provided upon requesting profiles
    _counts_profile = None  # For speed, we cache the counts corresponding to the default _mask
    pad_geometry = None  # List of PADGeometry instances
    beam = None  # Beam instance for creating q magnitudes

    def __init__(self, q_mags=None, mask=None, n_bins=None, q_range=None, pad_geometry=None, beam=None, _plan=True):
        r"""
        Arguments:
            q_mags (|ndarray|): Optional.  Array of q magnitudes.
            mask (|ndarray|): Optional.  The arrays will be multiplied by this mask, and the counts per radial bin
                                will come from this (e.g. use values of 0 and 1 if you want a normal average, otherwise
                                you get a weighted average).
            n_bins (int): Number of radial bins you desire.
            q_range (list-like): The minimum and maximum of the *centers* of the q bins.
            pad_geometry (list of |PADGeometry| instances):  Optional.  Will be used to generate q magnitudes.  You must
                                                             provide beam if you provide this.
            beam (|Beam| instance): Optional, unless pad_geometry is provided.  Wavelength and beam direction are
                                     needed in order to calculate q magnitudes.
        """
        if _plan:
            self.make_plan(q_mags=q_mags, mask=mask, n_bins=n_bins, q_range=q_range,
                           pad_geometry=pad_geometry, beam=beam)

    def copy(self):
        r""" Make a copy of this profiler.  Copy all internal data. """
        rp = RadialProfiler(_plan=False)
        if self.n_bins is not None:
            rp.n_bins = self.n_bins
            rp.q_range = self.q_range
            rp.q_edge_range = self.q_edge_range.copy()
            rp.bin_centers = self.bin_centers.copy()
            rp.bin_edges = self.bin_edges.copy()
            rp.bin_size = self.bin_size
            rp._q_mags = self._q_mags.copy()
            rp.pad_geometry = self.pad_geometry.copy()
            rp.beam = self.beam.copy()
        if self._mask is not None:
            rp._mask = self._mask.copy()
        if self._counts_profile is not None:
            rp._counts_profile = self._counts_profile.copy()
        return rp

    @property
    def mask(self):
        r""" PAD masks. """
        if self._mask is not None:
            return self._mask.copy()
        return None

    @mask.setter
    def mask(self, mask):
        if mask is not None:
            mask = concat_pad_data(mask)
            # if self._mask is not None:  # Check if we already have an identical mask
            #     if np.sum(np.abs(mask - self.mask)) == 0:
            #         return
            self._mask = mask.copy()  # Ensure a properly flattened array
            self._counts_profile = None  # Wipe out this profile

    def set_mask(self, mask):
        r""" Update the mask.

        Arguments:
            mask (|ndarray| or list of |ndarray|): Mask
        """
        self.mask = mask

    @property
    def q_bin_centers(self):
        r""" The centers of the q bins. """
        return self.bin_centers.copy()

    @property
    def counts_profile(self):
        r""" The number of pixels in each radial bin. """
        if self._counts_profile is None:
            if self.mask is None:
                self._counts_profile, _ = np.histogram(self._q_mags, bins=self.n_bins, range=self.q_edge_range)
            else:
                self._counts_profile, _ = np.histogram(self._q_mags, weights=self.mask, bins=self.n_bins,
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
            pad_geometry = PADGeometryList(pad_geometry)
            q_mags = pad_geometry.q_mags(beam=beam)
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
        self._q_mags = q_mags
        self.n_bins = n_bins
        self.bin_centers = bin_centers
        self.bin_edges = bin_edges
        self.bin_size = bin_size
        self.q_range = q_range
        self.q_edge_range = q_edge_range
        self.mask = mask
        self.pad_geometry = pad_geometry
        self.beam = beam

    def get_profile_statistic(self, data, mask=None, statistic=None):
        r"""
        Calculate the radial profile of averaged intensities.

        Arguments:
            data (|ndarray|): The intensity data from which the radial profile is formed.
            mask (|ndarray|): Optional.  A mask to indicate bad pixels.  Zero is bad, one is good.  If no mask is
                                 provided here, the mask configured with :meth:`set_mask` will be used.
            statistic (function or list of functions): Provide a function of your choice that runs on each radial bin.


        Returns: |ndarray|
        """
        data = concat_pad_data(data)
        q_mags = self._q_mags
        if mask is not None:
            self.mask = mask
        if self.mask is not None:
            w = np.where(self.mask)
            data = data[w]
            q_mags = q_mags[w]
        list_type = False
        if isinstance(statistic, list):
            list_type = True
        statistic = utils.ensure_list(statistic)
        stat = []
        for s in statistic:
            stat.append(utils.binned_statistic(q_mags, data, s, self.n_bins, (self.bin_edges[0], self.bin_edges[-1])))
        if not list_type:
            stat = stat[0]
        return stat

    def get_counts_profile(self, mask=None):
        r""" Calculate the radial profile of counts that fall in each bin.

        Arguments:
            mask (|ndarray|): Optional mask (one means "good")

        Returns:
            |ndarray| : The counts profile
        """
        data = np.ones(self.pad_geometry.n_pixels)
        return self.get_profile_statistic(data, mask=mask, statistic=np.sum)

    def get_sum_profile(self, data, mask=None):
        r"""
        Calculate the radial profile of summed intensities.  This is divided by counts to get an average.

        Arguments:
            data |ndarray|:  The intensity data from which the radial profile is formed.
            mask |ndarray|:  Optional.  A mask to indicate bad pixels.  Zero is bad, one is good.

        Returns:  |ndarray|
        """
        data = concat_pad_data(data)
        return self.get_profile_statistic(data, mask=mask, statistic=np.sum)

    def get_mean_profile(self, data, mask=None):
        r"""
        Calculate the radial profile of averaged intensities.

        Arguments:
            data (|ndarray|):  The intensity data from which the radial profile is formed.
            mask (|ndarray|):  Optional.  A mask to indicate bad pixels.  Zero is bad, one is good.  If no mask is
                                 provided here, the mask configured with :meth:`set_mask` will be used.

        Returns: |ndarray|
        """
        data = concat_pad_data(data)
        return self.get_profile_statistic(data, mask=mask, statistic=np.mean)

    def get_median_profile(self, data, mask=None):
        r"""
        Calculate the radial profile of averaged intensities.

        Arguments:
            data (|ndarray|):  The intensity data from which the radial profile is formed.
            mask (|ndarray|):  Optional.  A mask to indicate bad pixels.  Zero is bad, one is good.  If no mask is
                                 provided here, the mask configured with :meth:`set_mask` will be used.

        Returns:  |ndarray|
        """
        data = concat_pad_data(data)
        return self.get_profile_statistic(data, mask=mask, statistic=np.median)

    def get_sdev_profile(self, data, mask=None):
        r"""
        Calculate the standard deviations of radial bin.

        Arguments:
            data (|ndarray|):  The intensity data from which the radial profile is formed.
            mask (|ndarray|):  Optional.  A mask to indicate bad pixels.  Zero is bad, one is good.  If no mask is
                                 provided here, the mask configured with :meth:`set_mask` will be used.

        Returns:  |ndarray|
        """
        data = concat_pad_data(data)
        return self.get_profile_statistic(data, mask=mask, statistic=np.std)

    def subtract_profile(self, data, mask=None, statistic=np.median):
        r"""
        Given some PAD data, subtract a radial profile (mean or median).

        Arguments:
            data (|ndarray|):  The intensity data from which the radial profile is formed.
            mask (|ndarray|):  Optional.  A mask to indicate bad pixels.  Zero is bad, one is good.  If no mask is
                                 provided here, the mask configured with :meth:`set_mask` will be used.
            statistic (function): Provide a function of your choice that runs on each radial bin.

        Returns:

        """
        as_list = False
        if isinstance(data, list):
            as_list = True
        if statistic == 'mean':
            mprof = self.get_mean_profile(data, mask=mask)
        elif statistic == 'median':
            mprof = self.get_median_profile(data, mask=mask)
        else:
            mprof = self.get_profile_statistic(data, mask=mask, statistic=statistic)
            # raise ValueError('Statistic %s not recognized' % (statistic,))
        mprofq = self.bin_centers
        mpat = np.interp(self._q_mags, mprofq, mprof)
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
        return self.subtract_profile(data, mask=mask, statistic=np.median)

    def get_profile(self, data, mask=None, average=True):
        r"""
        This method is depreciated.  Use get_mean_profile or get_sum_profile instead.
        """
        utils.depreciate("RadialProfiler.get_profile() is depreciated.  Use RadialProfiler.get_mean_profile().")
        if average is True:
            return self.get_mean_profile(data, mask=mask)
        return self.get_sum_profile(data, mask=mask)


def get_radial_profile(data, beam, pad_geometry, mask=None, n_bins=None, q_range=None, statistic=np.mean):
    r"""
    Compute a radial profile from a PAD (or list of pads).  Groups pixels according to bins in q-space, and then applies
    whatever function you desire to the data values in each bin.  Calculates the mean by default.

    Arguments:
        data (|ndarray| or list of |ndarray|): Data to get profiles from.
        beam (|Beam|): Beam info.
        pad_geometry (|PADGeometryList|): PAD geometry info.
        mask (|ndarray| or list of |ndarray|): Mask (one is good, zero is bad).
        n_bins (int): Number of radial bins.
        q_range (tuple of floats): Centers of the min and max q bins.
        statistic (function): The function you want to apply to each bin (default: :func:`np.mean`).

    Returns:
        (tuple):
            - **statistic** (|ndarray|) -- Radial statistic.
            - **bins** (|ndarray|) -- The values of q at the bin centers.
    """
    rp = RadialProfiler(beam=beam, pad_geometry=pad_geometry, mask=mask, n_bins=n_bins, q_range=q_range)
    return rp.get_profile_statistic(data, mask=None, statistic=statistic), rp.bin_centers


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

    Returns: str: File name.
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

    def _range(x):
        return np.arange(x, dtype=int)

    if file_format == 1:
        shapes = [out[keys[i]] for i in _range(n/2) + 1]
        masks = [out[keys[i]] for i in _range(n/2) + int(n/2) + 1]
        masks = [np.unpackbits(masks[i])[0:np.prod(shapes[i])].astype(int).reshape(shapes[i]) for i in _range(n/2)]
    else:
        masks = [out[keys[i]] for i in _range(n) + 1]
    return masks


def pnccd_pad_geometry_list(detector_distance=0.1):
    r"""
    Generate a list of :class:`PADGeometry <reborn.detector.PADGeometry>` instances that are inspired by
    the `pnCCD <https://doi.org/10.1016/j.nima.2009.12.053>`_ detector.

    Returns: List of |PADGeometry| instances
    """
    pads = load_pad_geometry_list(pnccd_geom_file)
    for p in pads:
        p.t_vec[2] = detector_distance
    return pads


def cspad_pad_geometry_list(detector_distance=0.1):
    r"""
    Generate a list of |PADGeometry| instances that are inspired by the |CSPAD| detector.

    Arguments:
        detector_distance (float): Detector distance in SI units

    Returns: List of |PADGeometry| instances
    """
    pads = load_pad_geometry_list(cspad_geom_file)
    for p in pads:
        p.t_vec[2] = detector_distance
    return pads


def cspad_2x2_pad_geometry_list(detector_distance=2.4):
    r"""
    Generate a list of |PADGeometry| instances that are inspired by the |CSPAD| detector.

    Arguments:
        detector_distance (float): Detector distance in SI units

    Returns: List of |PADGeometry| instances
    """
    pads = load_pad_geometry_list(cspad_2x2_geom_file)
    for p in pads:
        p.t_vec[2] = detector_distance
    return pads


def jungfrau4m_pad_geometry_list(detector_distance=0.1):
    r"""
    Generate a list of |PADGeometry| instances that are inspired by the |Jungfrau| 4M detector.

    Arguments:
        detector_distance (float): Detector distance in SI units

    Returns: List of |PADGeometry| instances
    """
    pads = load_pad_geometry_list(jungfrau4m_geom_file)
    for p in pads:
        p.t_vec[2] = detector_distance
    return pads


def epix10k_pad_geometry_list(detector_distance=0.1):
    r"""
    Generate a list of |PADGeometry| instances that are inspired by the epix10k detector.

    Arguments:
        detector_distance (float): Detector distance in SI units.

    Returns:
        (list): List of |PADGeometry| instances.
    """
    pads = load_pad_geometry_list(epix10k_geom_file)
    for p in pads:
        p.t_vec[2] = detector_distance
    return pads


def mpccd_pad_geometry_list(detector_distance=0.1):
    r"""
    Generate a list of |PADGeometry| instances that are inspired by SACLA's MPCCD detector.

    Arguments:
        detector_distance (float): Detector distance in SI units.

    Returns:
        (list): List of |PADGeometry| instances.
    """
    pads = load_pad_geometry_list(mpccd_geom_file)
    for p in pads:
        p.t_vec[2] = detector_distance
    return pads


def rayonix_mx340_xfel_pad_geometry_list(detector_distance=0.1, return_mask=False):
    r"""
    Generate a list of |PADGeometry| instances that are inspired by the Rayonix MX340-XFEL detector.

    Arguments:
        detector_distance (float): Detector distance in SI units.
        return_mask (bool): The Rayonix has a hole in the center; setting this to True will return the corresponding
                            mask along with .

    Returns: |PADGeometryList|
    """
    pads = load_pad_geometry_list(rayonix_mx340_xfel_geom_file)
    for p in pads:
        p.t_vec[2] = detector_distance
    if return_mask:
        mask = pads.ones()
        xyz = pads[0].position_vecs()
        xyz[:, 2] = 0
        mask[utils.vec_mag(xyz) < 0.0025] = 0
        return pads, mask
    return PADGeometryList(pads)


def dict_default(dictionary, key, default):
    r""" Sometimes we want to fetch a dictionary value for a given key, but the key might be absent in which case we
    accept a default value.  This function does that. """
    if key in dictionary.keys():
        return dictionary[key]
    return default


def _slice_to_tuple(slc):
    r""" Special conversion of slice type to tuple. """
    if slc is None:
        return None
    if isinstance(slc, list):
        return _slice_to_tuple(tuple(slc))
    if isinstance(slc, slice):
        return slc.start, slc.stop, slc.step
    if isinstance(slc, tuple):
        return tuple(_slice_to_tuple(s) for s in slc)
    if isinstance(slc, int):
        return slc
    raise ValueError('Cannot convert slice to tuple:', slc.__str__())


def _tuple_to_slice(slc):
    r""" Special conversion of tuple to slice. """
    if slc is None:
        return None
    if isinstance(slc, slice):
        return slc
    if isinstance(slc, tuple) or isinstance(slc, list):
        if False not in [isinstance(s, int) for s in slc]:
            return slice(slc[0], slc[1], slc[2])
        out = []
        for s in slc:
            if isinstance(s, tuple) or isinstance(s, list):
                out.append(slice(s[0], s[1], s[2]))
            else:
                out.append(s)
        return tuple(out)
    return slc
