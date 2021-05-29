import numpy as np
from ..fortran import density_f
try:
    from numba import jit
except ImportError:
    from ..utils import __fake_numba_jit as jit

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


def build_atomic_scattering_density_map(x_vecs, f, sigma, x_min, x_max, shape, orth_mat, max_radius=1e10):
    r"""
    Construct an atomic scattering density by summing Gaussians.  Sampling is assumed to be a rectangular grid, but axes
    need not be orthogonal (the orthogonalization matrix may be provided).  Normalization is preformed such that the
    sum over the whole map should equal the sum over the input scattering factors f.

    In order to increase speed, you may specify the maximum atom size, in which case only grid points within that
    distance are sampled.

    Args:
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


def _trilinear_interpolation_fortran(densities, vectors, corners, deltas, out):
    r"""
    This is the wrapper to the corresponding fortran function.  It is not meant to be used directly.  See the
    ``trilinear_interpolation`` function.

    Arguments:
        densities:
        vectors:
        corners:
        deltas:
        out:
    """
    assert vectors.dtype == np.float64
    assert corners.dtype == np.float64
    assert deltas.dtype == np.float64
    assert densities.flags.c_contiguous
    assert vectors.flags.c_contiguous
    assert corners.flags.c_contiguous
    assert deltas.flags.c_contiguous
    assert out.flags.c_contiguous
    assert np.min(deltas) > 0
    if np.iscomplexobj(densities) or np.iscomplexobj(out):
        assert densities.dtype == np.complex128
        assert out.dtype == np.complex128
        density_f.trilinear_interpolation_complex(densities.T, vectors.T, corners.T, deltas.T, out.T)
    else:
        assert densities.dtype == np.float64
        assert out.dtype == np.float64
        density_f.trilinear_interpolation(densities.T, vectors.T, corners.T, deltas.T, out.T)


def trilinear_interpolation(densities, vectors, corners=None, deltas=None, x_min=None, x_max=None, out=None,
                            strict_types=True):
    r"""
    Perform a `trilinear interpolation <https://en.wikipedia.org/wiki/Trilinear_interpolation>`__ on a 3D array.  An
    arbitrary set of sample points in the form of an :math:`N\times 3` array may be specified.

    Notes:
        * This function behaves as if the density is periodic; points that lie out of bounds will wrap around.  This
          might change in the future, in which case a keyword argument will be added so that you may explicitly decide
          what to do in the case of points that lie outside of the grid.  Note that periodic boundaries avoid the
          need for conditional statements within a for loop, which probably makes the function faster.  For now, if you
          think you have points that lie outside of the grid, consider handling them separately.
        * You may specify the output array, which is useful if you wish to simply add to an existing array that you
          have already allocated.  This can make your code faster and reduce memory.  Beware: the out array is not
          over-written -- the underlying fortran function will *add* to the existing *out* array.
        * Only double precision arrays (both real and complex are allowed) at the fortran level.  You may pass in
          other types, but they will be converted to double (or double complex) before passing to the fortran function.
        * Make sure that all your arrays are c-contiguous.
        * An older version of this code allowed the arguments *corners* and *deltas*.  They are discouraged because
          we aim to standardize on the *x_min* and *x_max* arguments documented below.  They may be removed in the
          future.
        * The shape of the 3D array is inferred from the *densities* argument.

    Arguments:
        densities (numpy array): A 3D density array.
        vectors (numpy array): An Nx3 array of vectors that specify the points to be interpolated.
        x_min (float or numpy array): A 3-element vector specifying the *center* of the corner voxel of the 3D array.
                                      If a float is passed instead, it will be replicated to make a 3D array.
        x_max (float or numpy array): Same as x_min, but specifies the opposite corner, with larger values than x_min.
        out (numpy array): If you don't want the output array to be created (e.g for speed), provide it here.
        strict_types (bool): Set this to False if you don't mind your code being slow due to the need to convert
                             datatypes (i.e. copy arrays) on every function call.  Default: True.

    Returns:
        numpy array
    """
    if (corners is not None) and (deltas is not None):
        corners = np.array(corners).copy()
        deltas = np.array(deltas).copy()
    else:
        if (x_min is None) or (x_max is None):
            raise ValueError('trilinear_interpolation requires the x_min and x_max arguments')
        shape = np.array(densities.shape)
        if len(shape) != 3:
            raise ValueError('trilinear_interpolation requires a 3D densities argument')
        x_min = np.atleast_1d(np.array(x_min))
        x_max = np.atleast_1d(np.array(x_max))
        if len(x_min) == 1:
            x_min = np.squeeze(np.array([x_min, x_min, x_min]))
        if len(x_max) == 1:
            x_max = np.squeeze(np.array([x_max, x_max, x_max]))
        deltas = (x_max - x_min)/(shape - 1)
        corners = x_min
    corners = corners.astype(np.float64)
    deltas = deltas.astype(np.float64)
    if not strict_types:
        if np.iscomplexobj(densities):
            densities = densities.astype(np.complex128)
        else:
            densities = densities.astype(np.float64)
        if out is not None:
            if np.iscomplexobj(out):
                out = out.astype(np.complex128)
            else:
                out = out.astype(np.float64)
    if out is None:
        out = np.zeros(vectors.shape[0], dtype=densities.dtype)
    _trilinear_interpolation_fortran(densities, vectors, corners, deltas, out)
    return out


def _trilinear_insertion_fortran(densities, weights, vectors, insert_vals, corners, deltas):
    r"""
    This is the wrapper to the corresponding fortran function.  It is not meant to be used directly.  See the
    ``trilinear_insertion`` function.

    Args:
        densities:
        weights:
        vectors:
        insert_vals:
        corners:
        deltas:
    """
    assert weights.dtype == np.float64
    assert vectors.dtype == np.float64
    assert corners.dtype == np.float64
    assert deltas.dtype == np.float64
    assert densities.flags.c_contiguous
    assert weights.flags.c_contiguous
    assert vectors.flags.c_contiguous
    assert insert_vals.flags.c_contiguous
    assert corners.flags.c_contiguous
    assert deltas.flags.c_contiguous
    if np.iscomplexobj(densities) or np.iscomplexobj(insert_vals):
        assert densities.dtype == np.complex128
        assert insert_vals.dtype == np.complex128
        density_f.trilinear_insertion_complex(densities.T, weights.T, vectors.T, insert_vals.T, corners.T, deltas.T)
    else:
        assert densities.dtype == np.float64
        assert insert_vals.dtype == np.float64
        density_f.trilinear_insertion_real(densities.T, weights.T, vectors.T, insert_vals.T, corners.T, deltas.T)


def trilinear_insertion(densities, weights, vectors, insert_vals, corners=None, deltas=None, x_min=None, x_max=None):
    r"""
    Perform the "inverse" of a `trilinear interpolation <https://en.wikipedia.org/wiki/Trilinear_interpolation>`__ .
    That is, take an arbitrary set of sample values along with their 3D vector locations and "insert" them into a 3D
    grid of densities.  The values are distributed amongst the nearest 8 grid points so that they sum to the original
    insert value.

    FIXME: Be more clear about the mathematical operation that this function performs...

    Notes:
        * This function behaves as if the density is periodic; points that lie out of bounds will wrap around.  This
          might change in the future, in which case a keyword argument will be added so that you may explicitly decide
          what to do in the case of points that lie outside of the grid.  Note that periodic boundaries avoid the
          need for conditional statements within a for loop, which probably makes the function faster.  For now, if you
          think you have points that lie outside of the grid, consider handling them separately.
        * You may specify the output array, which is useful if you wish to simply add to an existing 3D array that you
          have already allocated.  This can make your code faster and reduce memory.  Beware: the out array is not
          over-written -- the underlying fortran function will *add* to the existing ``densities`` array.
        * Only double precision arrays (both real and complex are allowed).
        * Make sure that all your arrays are c-contiguous.
        * An older version of this code allowed the arguments ``corners`` and ``deltas``.  They are discouraged because
          we aim to standardize on the ``x_min`` and ``x_max`` arguments documented below.  They may be removed in the
          future.
        * The shape of the 3D array is inferred from the ``densities`` argument.

    Arguments:
        densities (numpy array): A 3D array containing the densities, into which values are inserted.  Note that an
                                 "insertion" means that the ``insert_vals`` below are multiplied by ``weights`` below.
        weights (numpy array): A 3D array containing weights.  These are needed in order to perform a weighted average.
                               After calling this function one or more times, the average densities are calculated by
                               dividing ``densities`` by ``weights``.  Be mindful of divide-by-zero errors.
        vectors (numpy array): The 3D vector positions corresponding to the values to be inserted.
        insert_vals (numpy array): The values to be inserted into the 3D map.  They are multiplied by weights before
                                   being inserted into the densities map.
        x_min (float or numpy array): A 3-element vector specifying the *center* of the corner voxel of the 3D array.
                                      If a float is passed instead, it will be replicated to make a 3D array.
        x_max (float or numpy array): Same as x_min, but specifies the opposite corner, with larger values than x_min.

    Returns:
        None.  This function modifies the densities and weights arrays; it returns nothing.
    """
    if (corners is not None) and (deltas is not None):
        corners = np.array(corners).copy()
        deltas = np.array(deltas).copy()
    else:
        if (x_min is None) or (x_max is None):
            raise ValueError('trilinear_insertion requires the x_min and x_max arguments')
        shape = np.array(densities.shape)
        if len(shape) != 3:
            raise ValueError('trilinear_insertion requires a 3D densities argument')
        x_min = np.atleast_1d(np.array(x_min))
        x_max = np.atleast_1d(np.array(x_max))
        if len(x_min) == 1:
            x_min = np.squeeze(np.array([x_min, x_min, x_min]))
        if len(x_max) == 1:
            x_max = np.squeeze(np.array([x_max, x_max, x_max]))
        deltas = (x_max - x_min)/(shape - 1)
        corners = x_min
    corners = corners.astype(np.float64)
    deltas = deltas.astype(np.float64)
    print('>'*80, densities.dtype)
    print('\n','<'*80, densities.dtype != np.float64)
    if densities.dtype != np.float64:
        if densities.dtype != np.complex128:
            raise ValueError('trilinear_interpolation requires densities of numpy.float64 or numpy.complex128 type')
    if insert_vals.dtype != np.float64:
        if insert_vals.dtype != np.complex128:
            raise ValueError('trilinear_interpolation requires insert_vals of numpy.float64 or numpy.complex128 type')
    _trilinear_insertion_fortran(densities, weights, vectors, insert_vals, corners, deltas)
    return None
