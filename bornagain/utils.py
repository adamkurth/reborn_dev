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

    Compute the normal vector, which has a length of one.

    Args:
        vec: input vector, usually of shape (3) of (N, 3)

    Returns: new vector of length 1.

    """

    vecnorm = np.sqrt(np.sum(vec**2, axis=(vec.ndim-1)))
    return (vec.T / vecnorm).T


def vec_mag(vec):
    r"""

    Compute the scalar magnitude sqrt(sum(x^2)) of an array of vectors, usually shape (N, 3)

    Args:
        vec: input vector or array of vectors

    Returns: scalar vector magnitudes

    """

    return np.sqrt(np.sum(vec * vec, axis=(vec.ndim-1)))


# def rotate(rot, vec):
#     r"""
#
#     This defines a consistent way to rotate vectors.  It is a wrapper that does a simple operation:
#
#     .. code-block:: python
#
#         return np.dot(rot, vec.T).T
#
#     Note the bornagain package generally assumes that a set of N vectors of dimension D will be stored as a numpy array
#     of shape of N x D.
#
#     Args:
#         rot (numpy array): The rotation matrix.
#         vec (numpy array): The vector(s).  For N vectors of dimension D, this should be a NxD array.
#
#     Returns: numpy array of same shape as input vec
#
#     """
#
#     return np.dot(vec, rot.T.copy()) #np.dot(rot, vec.T).T


def depreciate(message):
    r"""

    Utility for sending warnings when some class, method, function, etc. is depreciated.  By default, a message of the
    form "WARNING: blah blah blah" will be printed with sys.stdout.write().  You get to choose the "blah blah blah" part
    of the message, which is the input to this function.

    The output can be silenced with the function bornagain.set_global('warn_depreciated', False), or you can force
    an error to occur if you do bornagain.set_global('force_depreciated', True).

    Args:
        message: whatever you want to have printed to the screen

    Returns: nothing

    """

    if ba.get_global('force_depreciated') is True:
        error(message)
    elif ba.get_global('warn_depreciated') is True:
        warn(message)


def warn(message):
    r"""

    Standard way of sending a warning message.  As of now this simply results in a function call

    sys.stdout.write("WARNING: %s\n" % message)

    The purpose of this function is that folks can search for "WARNING:" to find all warning messages, e.g. with grep.


    Args:
        message: the message you want to have printed.

    Returns: nothing

    """

    sys.stdout.write("WARNING: %s\n" % message)


def error(message):
    r"""

    Standard way of sending an error message.  As of now this simply results in a function call

    sys.stdout.write("ERROR: %s\n" % message)


    Args:
        message: the message you want to have printed.

    Returns: nothing
    """

    sys.stderr.write("ERROR: %s\n" % message)


# def axisAndAngleToMatrix(axis, angle):
#     """Generate the rotation matrix from the axis-angle notation.
#
#     Conversion equations
#     ====================
#
#     From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix),
#     the conversion is given by::
#
#         c = cos(angle); s = sin(angle); C = 1-c
#         xs = x*s;   ys = y*s;   zs = z*s
#         xC = x*C;   yC = y*C;   zC = z*C
#         xyC = x*yC; yzC = y*zC; zxC = z*xC
#         [ x*xC+c   xyC-zs   zxC+ys ]
#         [ xyC+zs   y*yC+c   yzC-xs ]
#         [ zxC-ys   yzC+xs   z*zC+c ]
#
#
#     @param matrix:  The 3x3 rotation matrix to update.
#     @type matrix:   3x3 numpy array
#     @param axis:    The 3D rotation axis.
#     @type axis:     numpy array, len 3
#     @param angle:   The rotation angle.
#     @type angle:    float
#     """
#
#     # Trig factors.
#     ca = np.cos(angle)
#     sa = np.sin(angle)
#     C = 1 - ca
#
#     # Depack the axis.
#     a = axis.ravel()
#     x = a[0]
#     y = a[1]
#     z = a[2]
# #     x, y, z = axis
#
#     # Multiplications (to remove duplicate calculations).
#     xs = x * sa
#     ys = y * sa
#     zs = z * sa
#     xC = x * C
#     yC = y * C
#     zC = z * C
#     xyC = x * yC
#     yzC = y * zC
#     zxC = z * xC
#
#     matrix = np.zeros([3, 3])
#
#     # Update the rotation matrix.
#     matrix[0, 0] = x * xC + ca
#     matrix[0, 1] = xyC - zs
#     matrix[0, 2] = zxC + ys
#     matrix[1, 0] = xyC + zs
#     matrix[1, 1] = y * yC + ca
#     matrix[1, 2] = yzC - xs
#     matrix[2, 0] = zxC - ys
#     matrix[2, 1] = yzC + xs
#     matrix[2, 2] = z * zC + ca
#
#     return matrix
#
#
# class ScalarMonitor(object):
#
#     """ Class for monitoring a scalar for which we expect many observations.
#         Array will grow as needed, and basic calculations can be done."""
#
#     def __init__(self, size=1000):
#
#         self.idx = 0         # Current index of observation
#         self.size = size     # Size of array
#         self.data = np.zeros([size])  # Data array
#         self.maxSize = 10e6  # Don't grow array larger than this
#
#     def append(self, value):
#
#         if (self.idx + 1) > self.size:
#             if (self.size * 2) > self.maxSize:
#                 print("Cannot grow array larger than %d" % self.size * 2)
#                 return None
#             self.data = np.concatenate([self.data, np.zeros([self.size])])
#             self.size = self.data.shape[0]
#         self.data[self.idx] = value
#         self.idx += 1
#
#     def getData(self):
#
#         return self.data[0:self.idx]
#
#     def getMean(self):
#
#         return np.mean(self.getData())
#
#     def getStd(self):
#
#         return np.std(self.getData())


def random_rotation(deflection=1.0, randnums=None):
    r"""
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
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


def rotation_about_axis(theta, u):
    r"""
    This needs to be tested.  It was taken from
    https://stackoverflow.com/questions/17763655/rotation-of-a-point-in-3d-about-an-arbitrary-axis-using-python
    """

    u = vec_norm(np.array(u)).reshape(3)
    ct = cos(theta)
    st = sin(theta)
    rot = np.array([[ct + u[0]**2 * (1 - ct),
                      u[0] * u[1] * (1 - ct) - u[2] * st,
                      u[0] * u[2] * (1 - ct) + u[1] * st],
                     [u[0] * u[1] * (1 - ct) + u[2] * st,
                      ct + u[1]**2 * (1 - ct),
                      u[1] * u[2] * (1 - ct) - u[0] * st],
                     [u[0] * u[2] * (1 - ct) - u[1] * st,
                      u[1] * u[2] * (1 - ct) + u[0] * st,
                      ct + u[2]**2 * (1 - ct)]])
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

    :param div_fwhm:
    :return:
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


def random_mosaic_rotation(mosaicity_fwhm):
    r"""
    Attempt to generate a random orientation for a crystal mosaic domain.  This is a hack.
    We take the matrix product of three rotations, each of the same FWHM, about the three
    orthogonal axis.  The order of this product is a random permutation.

    :param mosaicity_fwhm:
    :return:
    """

    if mosaicity_fwhm == 0:
        return np.eye(3)

    rs = list()
    rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [1.0, 0, 0]))
    rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [0, 1.0, 0]))
    rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [0, 0, 1.0]))
    rind = np.random.permutation([0, 1, 2])
    return rs[rind[0]].dot(rs[rind[1]].dot(rs[rind[2]]))


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

def trilinear_interpolation(data, min_corners, max_corners, samples, mask=None):
    r""""
    Trilinear interpolation on a regular grid with arbitrary sample points.

    Note: all input arrays should be C contiguous.

    Arguments:
        data : 3D Numpy array that you wish to sample
        min_corners : An array with the three minimum values corresponding to the data grid center points
        max_corners : An array with the three maximum values corresponding to the data grid center points
        samples : An Nx3 array of 3-vectors containing coordinates of points that you wish to sample
        mask : Numpy array specifying which voxels in data to ignore.  Zero means ignore.

    Returns:
        Numpy arrays with interpolated values
    """

    if fortran is None:
        raise ImportError('You need to compile fortran code to use utils.trilinear_interpolation()')

    min_corners = np.array(min_corners)
    max_corners = np.array(max_corners)
    shape = np.array(data.shape)

    # TODO: fix this
    min_corners = -max_corners
    deltas = (max_corners - min_corners)/(shape-1)
    ####################################

    data = data.astype(np.double)
    shape = shape.astype(np.int)

    assert data.flags.c_contiguous
    assert samples.flags.c_contiguous
    if mask is not None:
        assert mask.flags.c_contiguous

    dataout = np.empty((samples.shape[0]), dtype=np.double)
    fortran.interpolations_f.trilinear_interpolation(data, samples, max_corners, deltas, dataout)

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