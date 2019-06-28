r"""
Some utility functions that might be useful throughout bornagain.  Don't put highly specialized functions here.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from functools import wraps
import sys
import numpy as np
from numpy import sin, cos
import bornagain as ba
try:
    from numba import jit
except ImportError:
    jit = None
try:
    from bornagain import fortran
except ImportError:
    fortran = None


def vec_norm(vec):
    r"""
    Compute normal vectors, which have lengths of one.

    Arguments:
        vec: Input vector of shape (N, 3)

    Returns:
        (numpy array) New unit vectors, shape (N, 3)
    """

    vecnorm = np.sqrt(np.sum(vec**2, axis=(vec.ndim-1)))
    return (vec.T / vecnorm).T


def vec_mag(vec):
    r"""
    Compute the scalar magnitude of an array of vectors.

    Arguments:
        vec (numpy array): Input array of vectors, shape (N, 3)

    Returns: scalar vector magnitudes
    """

    return np.sqrt(np.sum(vec * vec, axis=(vec.ndim-1)))


def depreciate(message):
    r"""
    Utility for sending warnings when some class, method, function, etc. is depreciated.  By default, a message of the
    form "WARNING: DEPRECIATION: blah blah blah" will be printed with sys.stdout.write().  You get to choose the
    "blah blah blah" part of the message, which is the input to this function.

    The output can be silenced with the function bornagain.set_global('warn_depreciated', False), or you can force
    an error to occur if you do bornagain.set_global('force_depreciated', True).

    Arguments:
        message: whatever you want to have printed to the screen

    Returns: None
    """

    if ba.get_global('force_depreciated') is True:
        error('DEPRECIATION: ' + message)
    elif ba.get_global('warn_depreciated') is True:
        warn('DEPRECIATION: ' + message)


def warn(message):
    r"""
    Standard way of sending a warning message.  As of now this simply results in a function call

    sys.stdout.write("WARNING: %s\n" % message)

    The purpose of this function is that folks can search for "WARNING:" to find all warning messages, e.g. with grep.

    Arguments:
        message: the message you want to have printed.

    Returns: None
    """

    sys.stdout.write("WARNING: %s\n" % message)


def error(message):
    r"""
    Standard way of sending an error message.  As of now this simply results in a function call

    sys.stdout.write("ERROR: %s\n" % message)

    Arguments:
        message: the message you want to have printed.

    Returns: None
    """

    sys.stderr.write("ERROR: %s\n" % message)


def random_rotation(deflection=1.0, randnums=None):
    r"""
    Creates a random rotation matrix.

    TODO: documentation

    Arguments:
        deflection (float): the magnitude of the rotation. For 0, no rotation; for 1, competely random
                            rotation. Small deflection => small perturbation.
        randnums (numpy array): 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.

    Returns:
        (numpy array) Rotation matrix
    """

    # from
    # http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    vec = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    rot = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    mat = (np.outer(vec, vec) - np.eye(3)).dot(rot)
    return mat.reshape(3, 3)


def rotation_about_axis(theta, vec):
    r"""
    This needs to be tested.  It was taken from
    https://stackoverflow.com/questions/17763655/rotation-of-a-point-in-3d-about-an-arbitrary-axis-using-python

    Arguments:
        theta (float): Rotation angle
        vec (numpy array): 3D vector specifying rotation axis

    Returns (numpy array): The shape (3, 3) rotation matrix
    """

    vec = vec_norm(np.array(vec)).reshape(3)
    ct = cos(theta)
    st = sin(theta)
    rot = np.array([[ct + vec[0] ** 2 * (1 - ct),
                     vec[0] * vec[1] * (1 - ct) - vec[2] * st,
                     vec[0] * vec[2] * (1 - ct) + vec[1] * st],
                    [vec[0] * vec[1] * (1 - ct) + vec[2] * st,
                     ct + vec[1] ** 2 * (1 - ct),
                     vec[1] * vec[2] * (1 - ct) - vec[0] * st],
                    [vec[0] * vec[2] * (1 - ct) - vec[1] * st,
                     vec[1] * vec[2] * (1 - ct) + vec[0] * st,
                     ct + vec[2] ** 2 * (1 - ct)]])
    return rot.reshape(3, 3)


def random_unit_vector():
    r"""
    Generate a totally random unit vector.

    Returns: numpy array length 3

    """
    return vec_norm(np.random.normal(size=3))


def random_beam_vector(div_fwhm):
    r"""
    A random vector for emulating beam divergence.
    Generates a random normal vector that is nominally along the [0,0,1] direction
    but with a random rotation along the [1,0,0] axis with given FWHM (Gaussian
    distributed and centered about zero) followed by a random rotation about the
    [0,0,1] axis with uniform distribution in the interval [0,2*pi).

    Arguments:
        div_fwhm (float):  FWHM of divergence angle.  Assuming Gaussian, where sigma = FWHM / 2.3548

    Returns:
        (numpy array) of length 3
    """

    # Don't do anything if no divergence
    bvec = np.array([0, 0, 1.0])
    if div_fwhm == 0:
        return bvec

    # First rotate around the x axis with Gaussian prob. dist.
    sig = div_fwhm / 2.354820045
    theta = np.random.normal(0, sig, [1])[0]
    rtheta = rotation_about_axis(theta, [1.0, 0, 0])
    bvec = np.dot(rtheta, bvec)

    # Next rotate around z axis with uniform dist [0,2*pi)
    phi = np.random.random(1)[0] * 2 * np.pi
    rphi = rotation_about_axis(phi, [0, 0, 1.0])
    bvec = np.dot(rphi, bvec)
    bvec /= np.sqrt(np.sum(bvec**2))

    return bvec


# def random_mosaic_rotation(mosaicity_fwhm):
#     r"""
#     Attempt to generate a random orientation for a crystal mosaic domain.  This is a hack.
#     We take the matrix product of three rotations, each of the same FWHM, about the three
#     orthogonal axis.  The order of this product is a random permutation.
#
#     :param mosaicity_fwhm:
#     :return:
#     """
#
#     if mosaicity_fwhm == 0:
#         return np.eye(3)
#
#     rs = list()
#     rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [1.0, 0, 0]))
#     rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [0, 1.0, 0]))
#     rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [0, 0, 1.0]))
#     rind = np.random.permutation([0, 1, 2])
#     return rs[rind[0]].dot(rs[rind[1]].dot(rs[rind[2]]))


def triangle_solid_angle(r1, r2, r3):
    r"""
    Compute solid angle of a triangle whose vertices are r1,r2,r3, using the method of
    Van Oosterom, A. & Strackee, J. Biomed. Eng., IEEE Transactions on BME-30, 125-126 (1983).
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


def triangle_solid_angle2(r1, r2, r3):
    r"""
    Compute solid angle of a triangle whose vertices are r1,r2,r3, using the method of
    Van Oosterom, A. & Strackee, J. Biomed. Eng., IEEE Transactions on BME-30, 125-126 (1983).
    """

    numer = np.abs(np.sum(r1*np.cross(r2, r3), axis=-1))

    r1_n = np.linalg.norm(r1, axis=-1)
    r2_n = np.linalg.norm(r2, axis=-1)
    r3_n = np.linalg.norm(r3, axis=-1)
    denom = r1_n * r2_n * r2_n
    print(r1.shape)
    print(r3_n.shape)
    print(np.sum(r1*r2, axis=-1).shape)
    denom += np.sum(r1*r2, axis=-1) * r3_n
    denom += np.sum(r2*r3, axis=-1) * r1_n
    denom += np.sum(r3*r1, axis=-1) * r2_n
    s_ang = np.arctan2(numer, denom) * 2

    return s_ang


def memoize(function):
    r"""
    This is a function decorator for caching results from a function, to avoid
    excessive computation or reading from disk.  Search the web for more
    details of how this works.
    """

    memo = {}

    @wraps(function)
    def wrapper(*args):

        if args in memo:
            return memo[args]

        rv = function(*args)
        memo[args] = rv
        return rv

    return wrapper


if jit is not None:
    @jit(nopython=True)
    def max_pair_distance(vecs):
        d_max = 0
        for i in range(vecs.shape[0]):
            for j in range(vecs.shape[0]):
                d = np.sum((vecs[i, :] - vecs[j, :])**2)
                if d > d_max:
                    d_max = d
        return np.sqrt(d_max)
else:
    def max_pair_distance(vecs):
        d_max = 0
        for i in range(vecs.shape[0]):
            for j in range(vecs.shape[0]):
                d = np.sum((vecs[i, :] - vecs[j, :])**2)
                if d > d_max:
                    d_max = d
        return np.sqrt(d_max)

def trilinear_insert(data_coord, data_val, x_min, x_max, N_bin, mask):
    r""""
    Trilinear insertion on a regular grid with arbitrary sample points.
    The boundary is defined as [x_min-0.5, x_max+0.5).

    Note 1: All input arrays should be C contiguous.
    Note 2: This code will break if you put a 1 in any of the N_bin entries.

    Arguments:
        data_coord: An Nx3 array of 3-vectors containing coordinates of the data points that you wish to insert into the regular grid.
        data_val  : An array with the N values containing the values of the data points.
        x_min     : An array with the three values corresponding to the smallest data grid center points.
        x_max     : An array with the three values corresponding to the largest data grid center points.
        N_bin     : An array with the three values corresponding to the number of bins in each direction.
        mask      : An array with the N values specifying which data points to ignore. Zero means ignore.
    
    Returns:
        A 3D numpy array with trilinearly inserted values.
    """

    #------------------------------------------
    # Checks
    if fortran is None:
        raise ImportError('You need to compile fortran code to use utils.trilinear_interpolation()')

    if len(data_coord) != len(data_val):
        raise ValueError('The data coordinates and data values must be of the same length.')

    if len(data_coord) != len(mask):
        raise ValueError('The data coordinates and data mask must be of the same length.')

    if len(x_min) != 3:
        raise ValueError('x_min needs to be an array that contains three elements.')

    if len(x_max) != 3:
        raise ValueError('x_max needs to be an array that contains three elements.')

    if len(N_bin) != 3:
        raise ValueError('N_bin needs to be an array that contains three elements.')

    # Check if the non-1D arrays are c_contiguous
    assert data_coord.flags.c_contiguous

    # Convert to appropriate types
    data_coord = data_coord.astype(np.double)
    data_val = data_val.astype(np.double)
    x_min = x_min.astype(np.double)
    x_max = x_max.astype(np.double)
    N_bin = N_bin.astype(np.int)
    #------------------------------------------

    # Bin width
    Delta_x = (x_max - x_min) / (N_bin - 1)

    # Bin volume
    bin_volume = Delta_x[0] * Delta_x[1] * Delta_x[2]
    one_over_bin_volume = 1 / bin_volume

    # To safeguard against round-off errors
    epsilon = 1e-9

    # Constants (these are 3 element arrays)
    c1 = 0.0 - x_min / Delta_x
    c2 = x_max + 0.5 - epsilon
    c3 = x_min - 0.5 + epsilon

    # Initialise memory for Fortran
    dataout = np.zeros(N_bin+2, dtype=np.double, order='C')
    weightout = np.zeros(N_bin+2, dtype=np.double, order='C')
    dataout = np.asfortranarray(dataout)
    weightout = np.asfortranarray(weightout)

    # Mask out data_coord and data_val
    data_coord = data_coord[mask != 0, :]
    data_val = data_val[mask != 0]

    # Number of data points
    N_data = len(data_val)
    
    fortran.interpolations_f.trilinear_insert(data_coord, data_val, x_min, N_data, \
                                              Delta_x, one_over_bin_volume, c1, c2, c3,  \
                                              dataout, weightout)

    # Keep only the inner array - get rid of the boundary padding.
    dataout = dataout[1:N_bin[0]+1, 1:N_bin[1]+1, 1:N_bin[2]+1]
    weightout = weightout[1:N_bin[0]+1, 1:N_bin[1]+1, 1:N_bin[2]+1]

    # Calculate the mean value inserted into the array by dividing dataout by weightout.
    # For locations where weightout is zero, dataout should also be zero (because no values were inserted),
    # deal with this case by setting weightout to 1.
    assert np.sum(dataout[weightout == 0]) == 0

    weightout[weightout == 0] = 1

    dataout /= weightout

    return dataout


def passthrough_decorator(*args1, **kwargs1):
    r"""
    A function decorator that does nothing.  It is useful for dealing with the absence of the numba package; then
    we can define a jit decorator that does nothing.
    """
    def real_decorator(function):
        def wrapper():
            function(*args, **kwargs)
        return wrapper
    return real_decorator