"""

Some useful utility functions for pydiffract

"""

import sys
import numpy as np

def vecNorm(V):

    n = np.sqrt(np.sum(V * V, axis=-1))
    return V / np.tile(n, (1, 3)).reshape(3, len(n)).T

def warn(message):

    """ Simple warning message """

    sys.stdout.write("WARNING: %s\n" % message)


def error(message):

    """ Simple error message (to be replaced later...) """

    sys.stderr.write("ERROR: %s\n" % message)
