"""

Some useful utility functions for pydiffract

"""

import sys
import numpy as np
from Scientific.Geometry.Quaternion import Quaternion
from Scientific.Geometry import Tensor
from Scientific.Geometry.Transformation import Rotation

c = 299792458  # Speed of light
h = 6.62606957e-34  # Planck constant
hc = h * c  # Planck's constant * light speed
re = 2.818e-15  # classical electron radius
re2 = re ** 2
joulesPerEv = 1.60217657e-19

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

    return np.array([[0, np.cos(angle), -np.sin(angle)], [1, 0, 0], [0, np.sin(angle), np.cos(angle)]])


def rot2(angle):

    """ Rotate about axis 2. """

    return np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])


def rot3(angle):

    """ Rotate about axis 2. """

    return np.array([[np.cos(angle), -np.sin(angle), 0], [0, 0, 1], [np.sin(angle), np.cos(angle), 0]])


def kabschRotation(Vi1, Vi2):

    """ Find the best rotation to bring two vector lists into coincidence."""

    assert Vi1.shape[0] == Vi2.shape[0]
    assert Vi1.shape[0] > 0

    V1 = Vi1 - np.mean(Vi1, axis=0)
    V2 = Vi2 - np.mean(Vi2, axis=0)

    V, S, Wt = np.linalg.svd(np.dot(np.transpose(V2), V1))

    d = np.round(np.linalg.det(Wt.dot(V.T)))

    print(d)

    if d < 1:
        V[:, -1] *= -1

    U = (V.dot(Wt)).T

    return U

# def kabschRotation(Ai, Bi):
#
#     A = np.matrix(Ai)
#     B = np.matrix(Bi)
#
#     assert len(A) == len(B)
#
#     N = A.shape[0];  # total points
#
#     centroid_A = np.mean(A, axis=0)
#     centroid_B = np.mean(B, axis=0)
#
#     # centre the points
#     AA = A - np.tile(centroid_A, (N, 1))
#     BB = B - np.tile(centroid_B, (N, 1))
#
#     # dot is matrix multiplication for array
#     H = np.transpose(AA) * BB
#
#     U, S, Vt = np.linalg.svd(H)
#
#     R = Vt.T * U.T
#
#     # special reflection case
#     if np.linalg.det(R) < 0:
#         print "Reflection detected"
#         Vt[2, :] *= -1
#         R = Vt.T * U.T
#
#     t = -R * centroid_A.T + centroid_B.T
#
#     return np.transpose(R)


def randomRotationMatrix():

    """ Create a random rotation matrix."""

    q = Quaternion(np.random.rand(4))
    q = q.normalized()
    R = q.asRotation()
    return R.tensor.array


def axisAndAngle(R):

    """ Rotation angle and axis from rotation matrix."""

    sR = R.copy()
    T = Tensor(sR)
    sR = Rotation(T)
    V, phi = sR.axisAndAngle()
    VV = np.array([[V.x(), V.y(), V.z()]])
    return VV, phi

def axisAndAngleToMatrix(axis, angle):

    """Generate the rotation matrix from the axis-angle notation.

    Conversion equations
    ====================

    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::

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
