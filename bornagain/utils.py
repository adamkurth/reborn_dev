"""
Some useful utility functions for bornagain.  Utilities pertaining to unit
conversions will be removed.  There are some old functions here that will
likely also be removed.
"""

import sys
import numpy as np

c = 299792458  # Speed of light
h = 6.62606957e-34  # Planck constant
hc = h * c  # Planck's constant * light speed
re = 2.818e-15  # classical electron radius
re2 = re ** 2
joulesPerEv = 1.60217657e-19


def vecCheck(vec, hardcheck=False):
    """
Check that a vector meets our assumption of an Nx3 numpy array.  This is
helpful, for example, when we want to ensure that dot products and broadcasting
will work as expected. We could of course add an argument for vectors of
dimension other than 3, but for now 3-vectors are all that we work with.

Input:

vec:        The object that we are trying to make conform to our assumption
              about vectors.

hardcheck:  If True, then this function will raise a ValueError if the check
              fails.  If False, then this function attempts to fix the problem
              with the input.

==== Output:
vec:        The original input if it satisfies our conditions.  Otherwise
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


def eV2Joules(eV):
    """ Convert electron volts into Joules. """

    return eV * 1.60217657e-19


def photonEnergy2Wavelength(energy):
    """ Convert photon energy to photon wavelength. SI units, as always. """

    return hc / energy


def photonWavelength2Energy(wavelength):
    """ Convert photon wavelength to energy. SI units, as always. """

    return hc / wavelength


def vecNorm(V):

    v = V
    if v.ndim != 2:
        raise ValueError("V must have one or two dimensions.")
    n = np.sqrt(np.sum(V * V, axis=1))
    return (V.T / n).T


def vecMag(V):

    if V.ndim != 2:
        raise ValueError("V must have one or two dimensions.")
    return np.sqrt(np.sum(V * V, axis=1))


def warn(message):
    """ Simple warning message """

    sys.stdout.write("WARNING: %s\n" % message)


def error(message):
    """ Simple error message (to be replaced later...) """

    sys.stderr.write("ERROR: %s\n" % message)


def rot1(angle):
    """ Rotate about axis 1. """

    return np.array([[0, np.cos(angle), -np.sin(angle)],
                     [1, 0, 0],
                     [0, np.sin(angle), np.cos(angle)]])


def rot2(angle):
    """ Rotate about axis 2. """

    return np.array([[np.cos(angle), 0, -np.sin(angle)],
                     [0, 1, 0],
                     [np.sin(angle), 0, np.cos(angle)]])


def rot3(angle):
    """ Rotate about axis 2. """

    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [0, 0, 1],
                     [np.sin(angle), np.cos(angle), 0]])


def axisAndAngleToMatrix(axis, angle):
    """Generate the rotation matrix from the axis-angle notation.

    Conversion equations
    ====================

    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix),
    the conversion is given by::

        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]


    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """

    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca

    # Depack the axis.
    a = axis.ravel()
    x = a[0]
    y = a[1]
    z = a[2]
#     x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    matrix = np.zeros([3, 3])

    # Update the rotation matrix.
    matrix[0, 0] = x * xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y * yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z * zC + ca

    return matrix


class ScalarMonitor(object):

    """ Class for monitoring a scalar for which we expect many observations.
        Array will grow as needed, and basic calculations can be done."""

    def __init__(self, size=1000):

        self.idx = 0         # Current index of observation
        self.size = size     # Size of array
        self.data = np.zeros([size])  # Data array
        self.maxSize = 10e6  # Don't grow array larger than this

    def append(self, value):

        if (self.idx + 1) > self.size:
            if (self.size * 2) > self.maxSize:
                print("Cannot grow array larger than %d" % self.size * 2)
                return None
            self.data = np.concatenate([self.data, np.zeros([self.size])])
            self.size = self.data.shape[0]
        self.data[self.idx] = value
        self.idx += 1

    def getData(self):

        return self.data[0:self.idx]

    def getMean(self):

        return np.mean(self.getData())

    def getStd(self):

        return np.std(self.getData())
