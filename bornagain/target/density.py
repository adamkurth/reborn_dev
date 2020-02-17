from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
from numba import jit
from bornagain.fortran import density_f


class DensityMap(object):

    shape = None    # : Shape of the array, numpy style
    corner_min = None  # : 3-vector with coordinates of the center of the voxel with smaller values
    corner_max = None  # : 3-vector with coordinates of the center of the voxel with larger values
    spacings = None
    n_voxels = None  # : Total number of voxels
    dx = None       # : 3-vecto with increments beteen voxels
    strides = None  # : Array strides

    def __init__(self, shape=None, corner_min=None, corner_max=None, deltas=None):
        r"""

        Arguments:
            shape: shape of the 3d map
            corner_min:  a 3-vector with the coordinate of the center of the corner pixel
            corner_max:  opposit corner to corner_min
            deltas: step size along each of the 3 dimensions
        """

        self.corner_min = np.array(corner_min)

        if shape is None:
            self.shape = np.array(shape)

        self.corner_max = np.array(corner_max)
        self.deltas = (corner_max - corner_min)/(self.shape - 1)
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


@jit(nopython=True)
def build_atomic_scattering_density_map(x_vecs, f, sigma, x_min, x_max, shape, orth_mat, n_sigma=4):
    r"""
    Construct an atomic scattering density by summing Gaussians.  Sampling is assumed to be a rectangular grid, but axes
    need no be orthogonal (orrhogonalization matrix may be provided).  Normalization is taken care of by ensuring that
    the sum over Gaussian samples is equal to the provided scattering factors.

    Args:
        x_vecs (Mx3 numpy array) : Atom position vectors
        f (numpy array) : Atom structure factors
        sigma (float) : Standard deviation of Gaussian (all atoms are the same)
        x_min (numpy array) : Min position of corner voxel (center of voxel)
        x_max (numpy array) : Max position of corner voxel (center of voxel)
        shape (numpy array) : Shape of the 3D array
        orth_mat (3x3 numpy array) : Matrix that acts on distance vectors before calculating distance scalar
        n_sigma : Not implemented yet (how many sigmas to extend the atomic densities).

    Returns:
        numpy array : The density map
    """
    # Note that we must deal with wrap-around when calculating distances from atoms to grid points.
    #
    #
    # bins   |_____*_____|_x___*_____|_____*_____|
    #
    # index        0           1           2           3           4           5           6           7           8
    #
    # wrapped idx  0           1           2           0           1           2           0           1           2
    #
    # The above schematic is for a map with 3 bins.  The grid samples that correspond to x_min and x_max are in the
    # centers of the bins, indicated by the * symbol.  Supposing we want to place a Gaussian centered at the x position,
    # we need to calculate distances to sample points indexed with 0, 1, 2 but with wrap-around factored in.
    #
    #

    n_atoms = f.ravel().shape[0]  # Number of atoms
    dx = (x_max - x_min)/(shape - 1)  # Bin width
    b_tot = x_max - x_min + dx  # Total width of bins, including half-bins that extend beyond bin center points
    sum_map = np.zeros(shape.astype(np.int), dtype=f.dtype)
    sum_map_temp = np.zeros(shape.astype(np.int), dtype=f.dtype)

    count = 0
    for n in range(n_atoms):
        x_atom = x_vecs[n, :]
        sum_val = 0
        for i in range(int(shape[0])):
            xg = x_min[0] + i * dx[0]
            for j in range(int(shape[1])):
                yg = x_min[1] + j * dx[1]
                for k in range(int(shape[2])):
                    count += 1
                    zg = x_min[2] + k * dx[2]
                    x_grid = np.array([xg, yg, zg])
                    diff1 = x_grid - x_atom
                    diff = ((diff1 + b_tot/2) % b_tot) - b_tot/2
                    diff = np.dot(diff, orth_mat.T)
                    val = np.exp(-np.sum(diff**2)/(2*sigma**2))
                    sum_val += val
                    sum_map_temp[i, j, k] = val
        sum_map_temp /= sum_val
        sum_map_temp *= f[n]
        sum_map += sum_map_temp

    return sum_map


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

    Arguments:
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


# @jit(['void(float64[:], float64[:], float64[:], float64[:], float64[:])'], nopython=True)
# def trilinear_insertion(densities=None, weights=None, vectors=None, input_densities=None, limits=None):
#     r"""
#     Trilinear "insertion" -- basically the opposite of trilinear interpolation.  This places densities into a grid
#     using the same weights as in trilinear interpolation.
#
#     Arguments:
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
