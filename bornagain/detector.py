"""
Classes for analyzing/simulating diffraction data contained in pixel array
detectors (PADs).
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys

import numpy as np
import h5py

try:
    import matplotlib
    import pylab as plt
except ImportError:
    pass

from .utils import vec_norm, vec_mag, vec_check
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
    def n_pixels(self):

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

    def save(self, save_fname):
        """Saves an hdf5 file with class attributes for later use"""
        with h5py.File(save_fname, "w") as h:
            for name, data in vars(self).items():
                h.create_dataset(name, data=data)

    @classmethod
    def load(cls, fname):
        """ load a PAD object from fname"""
        pad = cls()
        with h5py.File(fname, "r") as h:
            for name in h.keys():
                data = h[name].value
                setattr(pad, name, data)
        return pad

    def simple_setup(self, n_pixels=1000, pixel_size=100e-6, distance=0.1):
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

    def ds_vecs(self, beam_vec):
        r""" Normalized scattering vectors s - s0 where s0 is the incident beam direction
        (`beam_vec`) and  s is the outgoing vector for a given pixel.  This does **not** have
        the 2*pi/lambda factor included."""

        return vec_norm(self.position_vecs()) - vec_check(beam_vec)

    def q_vecs(self, beam_vec, wavelength):
        r"""
        Calculate scattering vectors:

            :math:`\vec{q}_{ij}=\frac{2\pi}{\lambda}\left(\hat{v}_{ij} - \hat{b}\right)`

        Returns: numpy array
        """

        return (2 * np.pi / wavelength) * self.ds_vecs(beam_vec=beam_vec)

    def solid_angles2(self):
        """
        this should be sped up by vectorizing, but its more readable for now
        and only has to be done once per PAD geometry... 
        
        Divide each pixel up into two triangles with vertices R1,R2,R3
        and R2,R3,R4. Then use analytical form to find the solid angle of
        each triangle. Sum them to get the solid angle of pixel.
        """
        k = self.position_vecs()
        R1 = k - self.fs_vec * .5 - self.ss_vec * .5
        R2 = k + self.fs_vec * .5 - self.ss_vec * .5
        R3 = k - self.fs_vec * .5 + self.ss_vec * .5
        R4 = k + self.fs_vec * .5 + self.ss_vec * .5
        sa_1 = np.array([self._comp_solid_ang(r1, r2, r3)
                         for r1, r2, r3 in zip(R1, R2, R3)])
        sa_2 = np.array([self._comp_solid_ang(r4, r2, r3)
                         for r4, r2, r3 in zip(R4, R2, R3)])
        return sa_1 + sa_2

    def _comp_solid_ang(self, r1, r2, r3):

        r"""
        compute solid angle of a triangle whose vertices are r1,r2,r3
        Ref:thanks Jonas ...  
        Van Oosterom, A. & Strackee, J. 
        The Solid Angle of a Plane Triangle. Biomedical Engineering, 
        IEEE Transactions on BME-30, 125-126 (1983).
        """

        numer = np.abs(np.dot(r1, np.cross(r2, r3)))

        r1_n = np.linalg.norm(r1)
        r2_n = np.linalg.norm(r2)
        r3_n = np.linalg.norm(r3)
        denom = r1_n * r2_n * r2_n
        denom += np.dot(r1, r2) * r3_n
        denom += np.dot(r2, r3) * r1_n
        denom += np.dot(r3, r1) * r2_n
        s_ang = np.arctan2(numer, denom) * 2

        return s_ang

    def solid_angles(self):

        r"""
        Calculate solid angles of pixels, assuming the pixels have small angular extent.

        Returns: numpy array
        """

        v = self.position_vecs()
        n = self.norm_vec()

        a = vec_mag(np.cross(self.fs_vec, self.ss_vec))  # Area of the pixel
        r2 = vec_mag(v) ** 2  # Distance to the pixel, squared
        cs = np.dot(n, vec_norm(v).T)  # Inclination factor: cos(theta)
        sa = (a / r2) * cs  # Solid angle

        return np.abs(sa.ravel())

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

    def scattering_angles(self, beam_vec):
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

    def zeros(self):

        return np.zeros((self.n_ss, self.n_fs))

    def ones(self):

        return np.ones((self.n_ss, self.n_fs))


def split_pad_data(pad_list, data):

    r"""

    Given a contiguous block of data, split it up into individual PAD panels

    Args:
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


class SimplePAD(PADGeometry):

    """
    A simple child class to PADGeometry with some higher level functionality
    
    This will return a detector object representing a
    square pixel array detector
    
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

        - center (tuple)
            the fast-scan center coordinate and the slow-scan center coordinate

    """

    def __init__(self, n_pixels=1000, pixsize=0.00005, detdist=0.05, wavelen=1., center=None,
                 *args, **kwargs):

        PADGeometry.__init__(self, *args, **kwargs)

        self.detector_distance = detdist
        self.wavelength = wavelen
        self.si_energy = units.hc / (wavelen * 1e-10)

        self.simple_setup(n_pixels=n_pixels,
                          pixel_size=pixsize,
                          distance=detdist)

        self.fig = None

        # shape of the 2D det panel (2D image)
        self.img_sh = self.shape()

        if center is not None:
            assert (len(center) == 2)
            assert (center[0] < self.n_fs)
            assert (center[1] < self.n_ss)
            self.center = center
        else:
            self.center = map(lambda x: x / 2., self.img_sh)

        self.SOLID_ANG = self.solid_angles()

        self._make_qmag()

        # useful functions fr converting between pixel radii and momentum transfer
        self.rad2q = lambda rad: 4 * np.pi * np.sin(.5 * np.arctan(rad * pixsize / detdist)) / wavelen
        self.q2rad = lambda q: np.tan(np.arcsin(q * wavelen / 4 / np.pi) * 2) * detdist / pixsize

        self.intens = None

    def _make_qmag(self):

        r"""
        Makes the momentum transfer of each Q
        """

        self.Q_vectors = self.q_vecs(
            beam_vec=np.array([0, 0, 1]),
            wavelength=self.wavelength)
        self.Qmag = np.sqrt(np.sum(self.Q_vectors ** 2, axis=1))

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
        qx_min, qy_min = self.Q_vectors[:, :2].min(0)
        qx_max, qy_max = self.Q_vectors[:, :2].max(0)
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


class IcosphereGeometry(object):
    r"""

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
        n_faces = faces.shape[0]

        face_centers = np.zeros([n_faces, 3])
        for i in range(0, n_faces):
            face_centers[i, :] = (verts[faces[i, 0], :] + verts[faces[i, 1], :] + verts[faces[i, 2], :]) / 3

        return verts, faces, face_centers

