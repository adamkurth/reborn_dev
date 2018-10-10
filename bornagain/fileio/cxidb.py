import h5py
import numpy as np


class SimpleReader(object):

    """
    This file reader makes all kinds of assumptions about what is in the cxidb file.

    Use at your own risk...
    """

    def __init__(self, filename=filename):

        self.h5file = h5py.File(filename)


