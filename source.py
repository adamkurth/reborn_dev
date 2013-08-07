'''
Created on Jul 27, 2013

@author: kirian
'''

import numpy as np
import utils


class beam(object):

    def __init__(self):

        self.wavelength = 0
        self.B = np.array([0, 0, 1])

    def copy(self):

        b = beam()
        b.wavelength = self.wavelength
        b.B = self.B.copy()
        return b

    def __str__(self):

        s = ""
        s += "wavelength : %g\n" % self.wavelength
        s += "B : [%g, %g, %g]\n" % (self.B[0], self.B[1], self.B[2])
        return s

    def check(self):

        if self.wavelength <= 0:
            utils.warn("Bad wavelength: %f" % self.wavelength)
            return False
        return True
