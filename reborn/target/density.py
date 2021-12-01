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

import numpy as np
try:
    from numba import jit
except ImportError:
    from ..utils import __fake_numba_jit as jit
from ..utils import depreciate


class DensityMap(object):

    shape = None    # : Shape of the array, numpy style
    corner_min = None  # : 3-vector with coordinates of the center of the voxel with smaller values
    corner_max = None  # : 3-vector with coordinates of the center of the voxel with larger values
    spacings = None
    n_voxels = None  # : Total number of voxels
    dx = None       # : 3-vector with increments between voxels
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

        Returns: |ndarray|
        """

        shp = self.shape
        ind = np.arange(0, self.n_voxels)
        n_vecs = np.zeros([self.n_voxels, 3])
        n_vecs[:, 0] = np.floor(ind / (shp[1]*shp[2]))
        n_vecs[:, 1] = np.floor(ind / shp[2]) % shp[1]
        n_vecs[:, 2] = ind % shp[2]
        return n_vecs


def build_atomic_scattering_density_map(x_vecs, f, sigma, x_min, x_max, shape, orth_mat, max_radius=1e10):
    r"""
    Construct an atomic scattering density by summing Gaussians.  Sampling is assumed to be a rectangular grid, but axes
    need not be orthogonal (the orthogonalization matrix may be provided).  Normalization is preformed such that the
    sum over the whole map should equal the sum over the input scattering factors f.

    In order to increase speed, you may specify the maximum atom size, in which case only grid points within that
    distance are sampled.

    Arguments:
        x_vecs (Mx3 numpy array) : Atom position vectors
        f (numpy array) : Atom structure factors
        sigma (float) : Standard deviation of Gaussian (all atoms are the same)
        x_min (numpy array) : Min position of corner voxel (center of voxel)
        x_max (numpy array) : Max position of corner voxel (center of voxel)
        shape (numpy array) : Shape of the 3D array
        orth_mat (3x3 numpy array) : Matrix that acts on distance vectors before calculating distance scalar
        max_atom_size : Maximum atom size (saves time by not computing tails of Gaussians)

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

    shape = shape.astype(np.int64)
    sum_map = np.zeros(shape.astype(np.int64), dtype=f.dtype)
    tmp = np.zeros(shape.astype(np.int64), dtype=f.dtype)
    _build_atomic_scattering_density_map_numba(x_vecs=x_vecs, f=f, sigma=sigma, x_min=x_min, x_max=x_max, shape=shape,
                                               orth_mat=orth_mat, max_radius=max_radius, sum_map=sum_map, tmp=tmp)

    return sum_map


@jit(nopython=True)
def _build_atomic_scattering_density_map_numba(x_vecs, f, sigma, x_min, x_max, shape, orth_mat, max_radius, sum_map,
                                               tmp):

    n_atoms = len(f)  # Number of atoms
    dx = (x_max - x_min)/(shape - 1)  # Bin widths in fractional coordinates
    dr = np.dot(dx, orth_mat.T)  # Bin widths in cartesian coordinates
    nn = np.ceil(max_radius/dr)  # Radii in voxel units
    for i in np.arange(3):
        #i = int(i)  # Why doesn't declarying int(i) here work?  Below I must do int(i) again, else there is an error.
        nn[int(i)] = int(min(nn[int(i)], np.floor((shape[int(i)]-1)/2.0)))  # Cap the radius so it is not larger than the map itself
    b_tot = x_max - x_min + dx  # Total width of bins, including half-bins that extend beyond bin center points

    for n in np.arange(n_atoms):
        n = int(n)
        x_atom = x_vecs[n, :]
        sum_val = 0
        idx = np.floor((x_atom - x_min)/dx + 0.5)  # Nominal grid point
        idx_min = idx - nn
        idx_max = idx + nn
        for i in np.arange(idx_min[0], idx_max[0]+1):
            imod = int(i % shape[0])
            xg = x_min[0] + imod * dx[0]
            for j in np.arange(idx_min[1], idx_max[1]+1):
                jmod = int(j % shape[1])
                yg = x_min[1] + jmod * dx[1]
                for k in np.arange(idx_min[2], idx_max[2]+1):
                    kmod = int(k % shape[2])
                    zg = x_min[2] + kmod * dx[2]
                    x_grid = np.array([xg, yg, zg])
                    diff1 = x_grid - x_atom
                    diff = ((diff1 + b_tot/2) % b_tot) - b_tot/2
                    diff = np.dot(diff, orth_mat.T)
                    val = np.exp(-np.sum(diff**2)/(2*sigma**2))
                    sum_val += val
                    tmp[imod, jmod, kmod] = val
        for i in np.arange(idx_min[0], idx_max[0]+1):
            imod = int(i % shape[0])
            for j in np.arange(idx_min[1], idx_max[1]+1):
                jmod = int(j % shape[1])
                for k in np.arange(idx_min[2], idx_max[2]+1):
                    kmod = int(k % shape[2])
                    sum_map[imod, jmod, kmod] += tmp[imod, jmod, kmod] * f[n] / sum_val

    return sum_map


def trilinear_interpolation(*args, **kwargs):
    r""" Depreciated duplicate of :func:`reborn.misc.interpolate.trilinear_interpolation` """
    depreciate("Don't use reborn.target.density.trilinear_interpolation.  Use "
               "reborn.misc.interpolate.trilinear_interpolation instead.")
    from ..misc.interpolate import trilinear_interpolation
    return trilinear_interpolation(*args, **kwargs)


def trilinear_insertion(*args, **kwargs):
    r""" Depreciated duplicate of :func:`reborn.misc.interpolate.trilinear_insertion` """
    depreciate("Don't use reborn.target.density.trilinear_insertion.  Use "
               "reborn.misc.interpolate.trilinear_insertion instead.")
    from ..misc.interpolate import trilinear_insertion
    return trilinear_insertion(*args, **kwargs)


def trilinear_insertions(*args, **kwargs):
    r""" Depreciated duplicate of :func:`reborn.misc.interpolate.trilinear_insertions` """
    depreciate("Don't use reborn.target.density.trilinear_insertions.  Use "
               "reborn.misc.interpolate.trilinear_insertions instead.")
    from ..misc.interpolate import trilinear_insertions
    return trilinear_insertions(*args, **kwargs)


def trilinear_insertion_factor(*args, **kwargs):
    r""" Depreciated duplicate of :func:`reborn.misc.interpolate.trilinear_insertion_factor` """
    depreciate("Don't use reborn.target.density.trilinear_insertion_factor.  Use "
               "reborn.misc.interpolate.trilinear_insertion_factor instead.")
    from ..misc.interpolate import trilinear_insertion_factor
    return trilinear_insertion_factor(*args, **kwargs)
