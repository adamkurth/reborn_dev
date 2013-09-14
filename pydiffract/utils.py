"""

Some useful utility functions for pydiffract

"""

import sys
import numpy as np
from Scientific.Geometry.Quaternion import Quaternion
from Scientific.Geometry import Tensor
from Scientific.Geometry.Transformation import Rotation

def vecNorm(V):

    n = np.sqrt(np.sum(V * V, axis=-1))
    return V / np.tile(n, (1, 3)).reshape(3, len(n)).T

def warn(message):

    """ Simple warning message """

    sys.stdout.write("WARNING: %s\n" % message)


def error(message):

    """ Simple error message (to be replaced later...) """

    sys.stderr.write("ERROR: %s\n" % message)


def kabschRotation(Vi1, Vi2):

    V1 = Vi1.copy()
    V2 = Vi2.copy()

    assert V1.shape[0] == V2.shape[0]
    L = V1.shape[0]
    assert L > 0

    COM1 = np.sum(V1, axis=0) / float(L)
    COM2 = np.sum(V2, axis=0) / float(L)
    V1 -= COM1
    V2 -= COM2

    V, S, Wt = np.linalg.svd(np.dot(np.transpose(V2), V1))

    reflect = float(str(float(np.linalg.det(V) * np.linalg.det(Wt))))

    if reflect == -1.0:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = np.dot(V, Wt)

    return U


def randomRotationMatrix():

    q = Quaternion(np.random.rand(4))
    q = q.normalized()
    R = q.asRotation()
    return R.tensor.array


def axisAndAngle(R):

    sR = R.copy()
    T = Tensor(sR)
    sR = Rotation(T)
    V, phi = sR.axisAndAngle()
    VV = np.array([[V.x(), V.y(), V.z()]])
    return VV, phi


