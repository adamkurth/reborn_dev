"""
Some useful utility functions for bornagain.  Utilities pertaining to unit
conversions will be removed.  There are some old functions here that will
likely also be removed.
"""

import sys
import numpy as np

from bornagain.simulate import refdata


def amplitudes_with_cmans(q, r, Z):
    """
    compute scattering amplitudes 
    
    q: 2D np.array of q vectors 
    r: 2D np.array of atom coors
    Z: 1D np.array of atomic numbers corresponding to r
    """

    cman = refdata.get_cromermann_parameters(Z)
    form_facts = refdata.get_cmann_form_factors(cman, q)
    ff_mat = np.array([form_facts[z] for z in Z]).T.astype(np.float32)
    amps = (np.dot(q, r.T)).astype(np.float32)
    amps = np.exp(1j * amps) .astype(np.complex64)
    amps = np.sum(amps * ff_mat, 1).astype(np.complex64)
    return amps


def amplitudes(q, r):
    """
    compute scattering amplitudes without form factors

    q: 2D np.array of q vectors 
    r: 2D np.array of atom coors
    """
    amps = np.dot(q, r.T)
    amps = np.exp(1j * amps)
    amps = np.sum(amps, 1)
    return amps


def sphericalize(lattice):
    """attempt to sphericalize a 2D lattice point array"""
    center = lattice.mean(0)
    rads = np.sqrt(np.sum((lattice - center)**2, 1))
    max_rad = min(lattice.max(0)) / 2.
    return lattice[rads < max_rad]


def vec_check(vec, hardcheck=False, dimension=3):
    """
Check that a vector meets our assumption of an Nx3 numpy array.  This is
helpful, for example, when we want to ensure that dot products and broadcasting
will work as expected. We could of course add an argument for vectors of
dimension other than 3, but for now 3-vectors are all that we work with.

Input:

vec: The object that we are trying to make conform to our assumption
about vectors.

hardcheck: If True, then this function will raise a ValueError if the check
fails.  If False, then this function attempts to fix the problem
with the input.

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
        raise ValueError('Vectors must be Nx3 numpy ndarrays')
    if vec.shape[1] != 3:
        if vec.shape[0] != 3:
            raise ValueError('Vectors must be Nx3 numpy ndarrays')
        else:
            vec = vec.T
    if ~vec.flags['C_CONTIGUOUS']:
        vec = vec.copy()

    return vec


def vec_norm(V):

    v = V
    if v.ndim != 2:
        raise ValueError("V must have one or two dimensions.")
    n = np.sqrt(np.sum(V * V, axis=1))
    return (V.T / n).T


def vec_mag(V):

    if V.ndim != 2:
        raise ValueError("V must have one or two dimensions.")
    return np.sqrt(np.sum(V * V, axis=1))


def warn(message):
    """ Simple warning message """

    sys.stdout.write("WARNING: %s\n" % message)


def error(message):
    """ Simple error message (to be replaced later...) """

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


def random_rotation_matrix(deflection=1.0, randnums=None):
    """
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
