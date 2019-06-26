from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from bornagain.utils import vec_mag
from scipy.stats import binned_statistic_dd
try:
    from numba import jit
except ImportError:
    from bornagain.utils import passthrough_decorator as jit


class DensityMap(object):

    shape = None  # : Shape of the array, numpy style
    corner_min = None  # : 3-vector with coordinates of the center of the voxel with smaller values
    corner_max = None  # : 3-vector with coordinates of the center of the voxel with larger values
    deltas = None
    n_voxels = None  # : Total number of voxels
    dx = None  # : 3-vecto with increments beteen voxels
    strides = None  # : Array strides

    def __init__(self, shape=None, corner_min=None, corner_max=None, deltas=None, molecule=None):
        r"""

        Args:
            shape: shape of the 3d map
            corner_min:  a 3-vector with the coordinate of the center of the corner pixel
            corner_max:  opposit corner to corner_min
        """

        self.shape = np.array(shape)
        self.corner_min = corner_min
        if deltas is None:
            self.deltas = (corner_max - corner_min)/(self.shape - 1)
        else:
            self.corner_max = self.corner_min + (self.shape - 1)*self.deltas
        self.limits = np.zeros((3, 2))
        self.limits[:, 0] = corner_min
        self.limits[:, 1] = corner_max
        self.n_voxels = int(np.prod(shape.ravel()))
        self.dx = 1.0 / shape
        self.strides = np.array([self.shape[0]*self.shape[1], self.shape[0], 1])

    @property
    def n_vecs(self):

        r"""

        Get an Nx3 array of vectors corresponding to the indices of the map voxels.  The array looks like this:

        [[0,0,0],[0,0,1],[0,0,2], ... ,[N-1,N-1,N-1]]

        Returns: numpy array

        """

        shp = self.shape
        ind = np.arange(0, self.n_voxels)
        n_vecs = np.zeros([self.n_voxels, 3])
        n_vecs[:, 0] = np.floor(ind / (shp[1]*shp[2]))
        n_vecs[:, 1] = np.floor(ind / shp[2]) % shp[1]
        n_vecs[:, 2] = ind % shp[2]
        return n_vecs


# class CrystalDensityMap(object):
#     r'''
#     A helper class for working with 3D crystal density maps.  Most importantly, it helps with spacegroup symmetry
#     transformations.  Once initialized with a crystal spacegroup and lattice, along with desired resolution and
#     oversampling ratio (this is one for normal crystallography), this tool chooses the shape of the map
#     and creates lookup tables for the symmetry transformations.
#
#     It is generally assumed that you are working in the crystal basis (fractional coordinates) -- the symmetry
#     transforms provided apply only to the crystal basis because that is the only way that we may avoid interpolation
#     artifacts for an arbitrary spacegroup/lattice.
#
#     This class does not maintain the data array as the name might suggest; it provides methods needed to work on the
#     data arrays.
#     '''
#
#     sym_luts = None
#     cryst = None  # :  CrystalStructure class used to initiate the map
#     oversampling = None  # : Oversampling ratio
#     dx = None  # : Length increments for fractional coordinates
#     cshape = None  # :  Number of samples along edges of unit cell within density map
#     shape = None  # :  Number of samples along edge of full density map (includes oversampling)
#     size = None  # :  Total number of elements in density map (np.prod(self.shape)
#     strides = None # :  The stride vector (mostly for internal use)
#
#     def __init__(self, cryst, resolution, oversampling):
#         r'''
#         On initialization, you provide a CrystalStructure class instance, along with your desired resolution
#         and oversampling.
#
#         Arguments:
#             cryst (CrystalStructure class instance) : A crystal structure that contains the spacegroup and lattice
#                                                       information.
#             resolution (float) : The desired resolution of the map (will be modified to suit integer samples and a
#                                   square 3D mesh)
#             oversampling (int) : An oversampling of 2 gives a real-space map that is twice as large as the unit cell. In
#                                   Fourier space, there will be one sample between Bragg samples.  And so on for 3,4,...
#         '''
#
#         # Given desired resolution and unit cell, these are the number of voxels along each edge of unit cell.
#         cshape = np.ceil((1/resolution) * (1/vec_mag(cryst.unitcell.a_mat.T)))
#
#         # The number of cells along an edge must be a multple of the shortest translation.  E.g., if an operation
#         # consists of a translation of 1/3 distance along the cell, we must have a multiple of 3.
#         multiples = np.ones(3, dtype=np.int)
#         for vec in cryst.spacegroup.sym_translations:
#             for j in range(0, 3):
#                 comp = vec[j] % 1
#                 comp = min(comp, 1-comp)  # This takes care of the fact that e.g. a 2/3 translation is the same as -1/3
#                 if comp == 0:
#                     comp = 1  # Avoid divide-by-zero problem
#                 comp = int(np.round(1/comp))
#                 if comp > multiples[j]:
#                     multiples[j] = comp
#         multiples = np.max(multiples)*np.ones(3)
#         cshape = np.ceil(cshape / multiples) * multiples
#
#         self.cryst = cryst
#         self.oversampling = np.int(np.ceil(oversampling))
#         self.dx = 1.0 / cshape
#         self.cshape = cshape.astype(int)
#         self.shape = (cshape * self.oversampling).astype(int)
#         self.size = np.prod(self.shape)
#         self.strides = np.array([self.shape[2]*self.shape[1], self.shape[2], 1])  # :  The stride vector (mostly for internal use)
#
#     @property
#     def n_vecs(self):
#         r"""
#         Get an Nx3 array of vectors corresponding to the indices of the map voxels.  The array looks like this:
#
#         [[0,  0,  0  ],
#          [0,  0,  1  ],
#          [0,  0,  2  ],
#          ...          ,
#          [N-1,N-1,N-1]]
#
#         Note that it is the third index, which we might associate with "z", that increments most rapidly.
#
#         Returns: numpy array
#         """
#
#         shp = self.shape
#         ind = np.arange(0, self.size)
#         n_vecs = np.zeros([self.size, 3])
#         n_vecs[:, 0] = np.floor(ind / (shp[1]*shp[2]))
#         n_vecs[:, 1] = np.floor(ind / shp[2]) % shp[1]
#         n_vecs[:, 2] = ind % shp[2]
#         return n_vecs
#
#     @property
#     def x_vecs(self):
#         r"""
#         Get an Nx3 array that contains the fractional coordinates.  For example, if there were four samples per unit
#         cell, the array looks like this:
#
#         [[    0,    0,    0],
#          [    0,    0, 0.25],
#          [    0,    0,  0.5],
#          [    0,    0, 0.75],
#          [    0, 0.25,    0],
#          ...                ,
#          [ 0.75, 0.75, 0.75]]
#
#         Returns: numpy array
#         """
#
#         return self.n_vecs * self.dx
#
#     @property
#     def x_limits(self):
#         r"""
#         Return a 3x2 array with the limits of the density map.  These limits correspond to the centers of the voxels.
#
#         Returns:
#         """
#
#         shp = self.shape
#         dx = self.dx
#         return np.array([[0, dx[0]*(shp[0]-1)], [0, dx[1]*(shp[1]-1)], [0, dx[2]*(shp[2]-1)]])
#
#     @property
#     def h_vecs(self):
#         r"""
#         This provides an Nx3 array of Fourier-space vectors "h".  These coordinates can be understood as "fractional
#         Miller indices" that coorespond to the density samples upon taking an FFT of the real-space map.  With atomic
#         coordinates x (in the crystal basis) one can take the Fourier transform F(h) = sum_n f_n exp(i h*x)
#
#         Returns: numpy array
#         """
#
#         h0 = np.fft.fftshift(np.fft.fftfreq(self.shape[0], d=self.oversampling/self.shape[0]))
#         h1 = np.fft.fftshift(np.fft.fftfreq(self.shape[1], d=self.oversampling/self.shape[1]))
#         h2 = np.fft.fftshift(np.fft.fftfreq(self.shape[2], d=self.oversampling/self.shape[2]))
#         hh0, hh1, hh2 = np.meshgrid(h0, h1, h2, indexing='ij')
#         print(hh0.shape)
#         h_vecs = np.empty((self.size, 3))
#         h_vecs[:, 0] = hh0.ravel()
#         h_vecs[:, 1] = hh1.ravel()
#         h_vecs[:, 2] = hh2.ravel()
#         return h_vecs
#
#     @property
#     def h_limits(self):
#
#         limits = np.zeros((3, 2))
#         limits[:, 0] = -np.floor(self.shape/2)/self.oversampling
#         limits[:, 1] = np.floor((self.shape-1) / 2)/self.oversampling
#         return limits
#
#     def get_sym_luts(self):
#         r"""
#         This provides a list of "symmetry transform lookup tables".  These are the linearized array indices. For a
#         transformation that consists of an identity matrix along with zero translation, the lut is just an array
#         p = [0,1,2,3,...,N^3-1].  Other transforms are like a "scrambling" of that ordering, such that a remapping of
#         density samples is done with an operation like this: newmap.flat[p2] = oldmap.flat[p1].   Note that the luts are
#         kept in memory for future use - beware of the memory requirement.
#
#         Returns: list of numpy arrays
#         """
#
#         if self.sym_luts is None:
#
#             sym_luts = []
#             x0 = self.x_vecs
#
#             for (R, T) in zip(self.cryst.spacegroup.sym_rotations, self.cryst.spacegroup.sym_translations):
#                 lut = np.dot(R, x0.T).T + T          # transform x vectors in 3D grid
#                 lut = np.round(lut / self.dx)        # switch from x to n vectors
#                 lut = lut % self.shape               # wrap around
#                 lut = np.dot(self.strides, lut.T)    # in p space
#                 sym_luts.append(lut.astype(np.int))
#             self.sym_luts = sym_luts
#
#         return self.sym_luts
#
#     def symmetry_transform(self, i, j, data):
#         r"""
#         Apply crystallographic symmetry transformation to a density map (3D numpy array).  This applies the mapping from
#         symmetry element i to symmetry element j, where i=0,1,...,N-1 for a spacegroup with N symmetry operations.
#
#         Arguments:
#             i (int) : The "from" index; symmetry transforms are performed from this index to the j index
#             j (int) : The "to" index; symmetry transforms are performed from the i index to this index
#
#         Returns: Numpy array with transformed densities
#         """
#
#         luts = self.get_sym_luts()
#         data_trans = np.zeros_like(data)
#         data_trans.flat[luts[j]] = data.flat[luts[i]]
#
#         return data_trans
#
#     def place_atoms_in_map(self, atom_x_vecs, atom_fs, mode='gaussian', fixed_atom_sigma=0.5e-10):
#
#         r"""
#
#         This will take a list of atom position vectors and densities and place them in a 3D map.  The position vectors
#         should be in the crystal basis, and the densities must be real (because the scipy function that we use does
#         not allow for complex numbers...).  This is done in a lazy way - the density samples are placed in the nearest
#         voxel.  There are no Gaussian shapes asigned to the atomic form.  Nothing fancy...
#
#         Args:
#             atom_x_vecs (numpy array):  An nx3 array of position vectors
#             atom_fs (numpy array):  An n-length array of densities (must be real)
#             mode (str): Either 'gaussian' or 'nearest'
#             fixed_atom_sigma (float): Standard deviation of
#
#         Returns: An NxNxN numpy array containing the sum of densities that were provided as input.
#
#         """
#         if mode == 'gaussian':
#             sigma = fixed_atom_sigma # Gaussian sigma (i.e. atom "size"); this is a fudge factor and needs to be updated
#             # n_atoms = atom_x_vecs.shape[0]
#             orth_mat = self.cryst.unitcell.o_mat.copy()
#             map_x_vecs = self.x_vecs
#             n_map_voxels = map_x_vecs.shape[0]
#             f_map = np.zeros([n_map_voxels], dtype=np.complex)
#             f_map_tmp = np.zeros([n_map_voxels], dtype=np.double)
#             s = self.oversampling
#             place_atoms_in_map(atom_x_vecs, atom_fs, sigma, s, orth_mat, map_x_vecs, f_map, f_map_tmp)
#             return np.reshape(f_map, self.shape)
#         elif mode == 'nearest':
#             mm = [0, self.oversampling]
#             rng = [mm, mm, mm]
#             a, _, _ = binned_statistic_dd(atom_x_vecs, atom_fs, statistic='sum', bins=[self.shape] * 3, range=rng)
#             return a


# @jit(nopython=True)
# def place_atoms_in_map(x_vecs, atom_fs, sigma, s, orth_mat, map_x_vecs, f_map, f_map_tmp):
#
#         r"""
#
#         Needs documentation...
#
#         """
#
#         n_atoms = x_vecs.shape[0]
#         n_map_voxels = map_x_vecs.shape[0]
#         # f_map = np.empty([n_map_voxels], dtype=atom_fs.dtype)
#         # f_map_tmp = np.empty([n_map_voxels], dtype=x_vecs.dtype)
#         for n in range(n_atoms):
#             x = x_vecs[n, 0] % s
#             y = x_vecs[n, 1] % s
#             z = x_vecs[n, 2] % s
#             w_tot = 0
#             for i in range(n_map_voxels):
#                 mx = map_x_vecs[i, 0]
#                 my = map_x_vecs[i, 1]
#                 mz = map_x_vecs[i, 2]
#                 dx = np.abs(x - mx)
#                 dy = np.abs(y - my)
#                 dz = np.abs(z - mz)
#                 dx = min(dx, s - dx)
#                 dy = min(dy, s - dy)
#                 dz = min(dz, s - dz)
#                 dr2 = (orth_mat[0, 0] * dx + orth_mat[0, 1] * dy + orth_mat[0, 2] * dz)**2 + \
#                       (orth_mat[1, 0] * dx + orth_mat[1, 1] * dy + orth_mat[1, 2] * dz)**2 + \
#                       (orth_mat[2, 0] * dx + orth_mat[2, 1] * dy + orth_mat[2, 2] * dz)**2
#                 w = np.exp(-dr2/(2*sigma**2))
#                 f_map_tmp[i] = w
#                 w_tot += w
#             f_map += atom_fs[n] * f_map_tmp/w_tot


try:
    from bornagain.target import density_f
except ImportError:
    density_f = None


def trilinear_interpolation_fortran(densities, vectors, corners, deltas, out):

    float_t = np.float64
    assert densities.dtype == float_t
    assert vectors.dtype == float_t
    assert corners.dtype == float_t
    assert deltas.dtype == float_t
    assert out.dtype == float_t
    assert densities.flags.c_contiguous
    assert vectors.flags.c_contiguous
    assert corners.flags.c_contiguous
    assert deltas.flags.c_contiguous
    assert out.flags.c_contiguous
    assert np.min(deltas) > 0
    density_f.trilinear_interpolation(densities.T, vectors.T, corners.T, deltas.T, out.T)


def trilinear_interpolation(densities, vectors, corners, deltas, out=None):

    if out is None:
        out = np.zeros(vectors.shape[0], dtype=densities.dtype)
    if density_f is not None:
        trilinear_interpolation_fortran(densities, vectors, corners, deltas, out)
#    else:
#        trilinear_interpolation_numba(densities=None, vectors=None, limits=None, out=None)
    return out


@jit(nopython=True)
def trilinear_interpolation_numba(densities=None, vectors=None, corners=None, deltas=None, out=None):
    r"""
    Trilinear interpolation of a 3D map.

    Args:
        densities: A 3D array of shape AxBxC
        vectors: An Nx3 array of 3-vectors
        limits: A 3x2 array specifying the limits of the density map samples.  These values specify the voxel centers.

    Returns: Array of intensities with length N.
    """

    nx = int(densities.shape[0])
    ny = int(densities.shape[1])
    nz = int(densities.shape[2])

    for ii in range(vectors.shape[0]):

        # Floating point coordinates
        i_f = float(vectors[ii, 0] - corners[0, 0]) / deltas[0]
        j_f = float(vectors[ii, 1] - corners[1, 0]) / deltas[1]
        k_f = float(vectors[ii, 2] - corners[2, 0]) / deltas[2]

        # Integer coordinates
        i = int(np.floor(i_f)) % nx
        j = int(np.floor(j_f)) % ny
        k = int(np.floor(k_f)) % nz

        # Trilinear interpolation formula specified in e.g. paulbourke.net/miscellaneous/interpolation
        k0 = k
        j0 = j
        i0 = i
        k1 = k+1
        j1 = j+1
        i1 = i+1
        x0 = i_f - np.floor(i_f)
        y0 = j_f - np.floor(j_f)
        z0 = k_f - np.floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        out[ii] = densities[i0, j0, k0] * x1 * y1 * z1 + \
                 densities[i1, j0, k0] * x0 * y1 * z1 + \
                 densities[i0, j1, k0] * x1 * y0 * z1 + \
                 densities[i0, j0, k1] * x1 * y1 * z0 + \
                 densities[i1, j0, k1] * x0 * y1 * z0 + \
                 densities[i0, j1, k1] * x1 * y0 * z0 + \
                 densities[i1, j1, k0] * x0 * y0 * z1 + \
                 densities[i1, j1, k1] * x0 * y0 * z0

    return out


def trilinear_insertion(densities, weights, vectors, vals, corners, deltas, weight=1):

    float_t = np.float64
    assert densities.dtype == float_t
    assert weights.dtype == float_t
    assert vectors.dtype == float_t
    assert vals.dtype == float_t
    assert corners.dtype == float_t
    assert deltas.dtype == float_t
    assert densities.flags.c_contiguous
    assert weights.flags.c_contiguous
    assert vectors.flags.c_contiguous
    assert vals.flags.c_contiguous
    assert corners.flags.c_contiguous
    assert deltas.flags.c_contiguous
    weight = float_t(weight)
    density_f.trilinear_insertion(densities.T, weights.T, vectors.T, vals.T, corners.T, deltas.T, weight)


# @jit(['void(float64[:], float64[:], float64[:], float64[:], float64[:])'], nopython=True)
# def trilinear_insertion(densities=None, weights=None, vectors=None, input_densities=None, limits=None):
#     r"""
#     Trilinear "insertion" -- basically the opposite of trilinear interpolation.  This places densities into a grid
#     using the same weights as in trilinear interpolation.
#
#     Args:
#         densities (NxMxP array):
#         weights (NxMxP array):
#         vectors (Qx3 array):
#         input_densities (length-Q array):
#         limits (3x2 array): A 3x2 array specifying the limits of the density map samples.  These values specify the
#                             voxel centers.
#
#     Returns: None -- the inputs densities and weights are modified by this function
#     """
#
#     nx = int(densities.shape[0])
#     ny = int(densities.shape[1])
#     nz = int(densities.shape[2])
#
#     dx = (limits[0, 1] - limits[0, 0]) / nx
#     dy = (limits[1, 1] - limits[1, 0]) / ny
#     dz = (limits[2, 1] - limits[2, 0]) / nz
#
#     for ii in range(vectors.shape[0]):
#
#         # Floating point coordinates
#         i_f = float(vectors[ii, 0] - limits[0, 0]) / dx
#         j_f = float(vectors[ii, 1] - limits[1, 0]) / dy
#         k_f = float(vectors[ii, 2] - limits[2, 0]) / dz
#
#         # Integer coordinates
#         i = int(np.floor(i_f))
#         j = int(np.floor(j_f))
#         k = int(np.floor(k_f))
#
#         # Trilinear interpolation formula specified in e.g. paulbourke.net/miscellaneous/interpolation
#         k0 = k
#         j0 = j
#         i0 = i
#         k1 = k+1
#         j1 = j+1
#         i1 = i+1
#         x0 = i_f - np.floor(i_f)
#         y0 = j_f - np.floor(j_f)
#         z0 = k_f - np.floor(k_f)
#         x1 = 1.0 - x0
#         y1 = 1.0 - y0
#         z1 = 1.0 - z0
#         if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
#             val = input_densities[ii]
#             densities[i0, j0, k0] += val
#             densities[i1, j0, k0] += val
#             densities[i0, j1, k0] += val
#             densities[i0, j0, k1] += val
#             densities[i1, j0, k1] += val
#             densities[i0, j1, k1] += val
#             densities[i1, j1, k0] += val
#             densities[i1, j1, k1] += val
#             weights[i0, j0, k0] += x1 * y1 * z1
#             weights[i1, j0, k0] += x0 * y1 * z1
#             weights[i0, j1, k0] += x1 * y0 * z1
#             weights[i0, j0, k1] += x1 * y1 * z0
#             weights[i1, j0, k1] += x0 * y1 * z0
#             weights[i0, j1, k1] += x1 * y0 * z0
#             weights[i1, j1, k0] += x0 * y0 * z1
#             weights[i1, j1, k1] += x0 * y0 * z0
