"""
Some utility functions that might be useful throughout bornagain.  Don't put highly specialized functions here.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from functools import wraps
import sys
import numpy as np
from numpy import sin, cos
import bornagain as ba

# not sure where to seed
np.random.seed()

def vec_check2(vec, *args, **kwargs):

    r"""
    Same as vec_check(vec, dimension=2).  See :func:`vec_check <bornagain.utils.vec_check>`
    """

    return vec_check(vec, dimension=2, *args, **kwargs)

def vec_check3(vec, *args, **kwargs):

    r"""
    Same as vec_check(vec, dimension=2)
    """

    return vec_check(vec, dimension=3, *args, **kwargs)

def vec_check(vec, hardcheck=False, dimension=3):

    r"""
    Check that a vector meets our assumptions, and correct it if it doesn't.

    Our assumptions are that:
        1) The array shape is Nxd, where d is the dimension of the vector, and
           N is the number of vectors.  This is important when applying rotation
           operations to many vectors.
        2) The array is c-contiguous.  This is important for passing arrays into
           external c functions or to opencl kernels.

    The above also helps when we want to ensure that dot products and broadcasting
    will work as expected.

    Note that we have chosen to keep vector components close in memory.  I.e.,
    the x, y, and z components of a given vector are contiguous.  This will ensure
    that rotations and broadcasting are as fast as can be.

    Input:

    vec: The object that we are trying to make conform to our assumption
    about vectors.

    hardcheck: If True, then this function will raise a ValueError if the check
    fails.  If False, then this function attempts to fix the problem
    with the input.

    dimension: The expected dimension of the vectors

    ==== Output:
    vec: The original input if it satisfies our conditions.  Otherwise
    return a modified numpy ndarray with the correct shape.
    """

    if hardcheck:  # Raise an error if the input isn't perfect
        if not isinstance(vec, np.ndarray):
            raise ValueError('Vectors must be Nx3 numpy ndarrays')
        if len(vec.shape) != 2:
            raise ValueError('Vectors must be Nx3 numpy ndarrays')
        if vec.shape[1] != 3:
            raise ValueError('Vectors must be Nx3 numpy ndarrays')
        return vec

    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    if len(vec.shape) == 1:
        vec = vec[np.newaxis]
    if len(vec.shape) != 2:
        raise ValueError('Vectors must be Nx%d numpy ndarrays' % (dimension))
    if vec.shape[1] != dimension:
        if vec.shape[0] != dimension:
            raise ValueError('Vectors must be Nx3 numpy ndarrays')
        else:
            vec = vec.T
    if ~vec.flags['C_CONTIGUOUS']:
        vec = vec.copy()

    return vec


def vec_norm(vec):

    r"""

    Compute the normal vector, which has a length of one.

    Args:
        vec: input vector, usually of shape (3) of (N, 3)

    Returns: new vector of length 1.

    """

    vec = vec_check(vec)
    if vec.ndim != 2:
        raise ValueError("V must have one or two dimensions.")
    vecnorm = np.sqrt(np.sum(vec * vec, axis=1))
    return (vec.T / vecnorm).T


def vec_mag(vec):

    r"""

    Compute the scalar magnitude sqrt(sum(x^2)) of an array of vectors, usually shape (N, 3)

    Args:
        vec: input vector or array of vectors

    Returns: scalar vector magnitudes

    """

    vec = vec_check(vec)
    if vec.ndim != 2:
        raise ValueError("V must have one or two dimensions.")
    return np.sqrt(np.sum(vec * vec, axis=1))


def rotate(rot, vec):

    r"""

    This defines a consistent way to rotate vectors.  It is a wrapper that does a simple operation:

    .. code-block:: python

        return np.matmul(rot, vec.T).T

    Note the bornagain package generally assumes that a set of N vectors of dimension D will be stored as a numpy array
    of shape of N x D.

    Args:
        rot (numpy array): The rotation matrix.
        vec (numpy array): The vector(s).  For N vectors of dimension D, this should be a NxD array.

    Returns: numpy array of same shape as input vec

    """

    # return np.matmul(rot, vec)
    return np.matmul(rot, vec.T).T


def depreciate(message):

    r"""

    Utility for sending warnings when some class, method, function, etc. is depreciated.  By default, a message of the
    form "WARNING: blah blah blah" will be printed with sys.stdout.write().  You get to choose the "blah blah blah" part
    of the message, which is the input to this function.

    The output can be silenced with the function bornagain.set_global('warn_depreciated', False), or you can force
    an error to occur if you do bornagain.set_global('force_depreciated', True).

    TODO: we need to formally raise a depreciated exception

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

    TODO: need to raise an exception intead of simply printing something...


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
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def rotation_about_axis(theta, u):

    """
    This needs to be tested.  It was taken from
    https://stackoverflow.com/questions/17763655/rotation-of-a-point-in-3d-about-an-arbitrary-axis-using-python
    """

    return np.array([[cos(theta) + u[0]**2 * (1-cos(theta)),
                      u[0] * u[1] * (1-cos(theta)) - u[2] * sin(theta),
                      u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta)],
                     [u[0] * u[1] * (1-cos(theta)) + u[2] * sin(theta),
                      cos(theta) + u[1]**2 * (1-cos(theta)),
                      u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta)],
                     [u[0] * u[2] * (1-cos(theta)) - u[1] * sin(theta),
                      u[1] * u[2] * (1-cos(theta)) + u[0] * sin(theta),
                      cos(theta) + u[2]**2 * (1-cos(theta))]])


def random_beam_vector(div_fwhm):

    """
    A random vector for emulating beam divergence.
    Generates a random normal vector that is nominally along the [0,0,1] direction
    but with a random rotation along the [1,0,0] axis with given FWHM (Gaussian
    distributed and centered about zero) followed by a random rotation about the
    [0,0,1] axis with uniform distribution in the interval [0,2*pi).

    :param div_fwhm:
    :return:
    """

    # Don't do anything if no divergence
    B = np.array([0, 0, 1.0])
    if div_fwhm == 0:
        return B

    # First rotate around the x axis with Gaussian prob. dist.
    sig = div_fwhm/2.354820045
    theta = np.random.normal(0, sig, [1])[0]
    Rtheta = rotation_about_axis(theta, [1.0, 0, 0])
    B = np.dot(Rtheta, B)

    # Next rotate around z axis with uniform dist [0,2*pi)
    phi = np.random.random(1)[0]*2*np.pi
    Rphi = rotation_about_axis(phi, [0, 0, 1.0])
    B = np.dot(Rphi, B)
    B /= np.sqrt(np.sum(B**2))

    return B

def random_mosaic_rotation(mosaicity_fwhm):

    """
    Attempt to generate a random orientation for a crystal mosaic domain.  This is a hack.
    We take the matrix product of three rotations, each of the same FWHM, about the three
    orthogonal axis.  The order of this product is a random permutation.

    :param mosaicity_fwhm:
    :return:
    """

    if mosaicity_fwhm == 0:
        return np.eye(3)

    Rs = []
    Rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [1.0, 0, 0]))
    Rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [0, 1.0, 0]))
    Rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [0, 0, 1.0]))
    rind = np.random.permutation([0, 1, 2])
    return Rs[rind[0]].dot(Rs[rind[1]].dot(Rs[rind[2]]))


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
