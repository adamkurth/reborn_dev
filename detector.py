
import numpy as np
from utils import warn
import source

"""
Classes for analyzing diffraction data contained in pixel array detectors (PAD)
"""


class panel(object):

    """
    Individual detector panel, assumed to be a flat 2D array of square pixels.
    """

    def __init__(self, name=""):

        """
        There are no default initialization parameters.  Zeros generally mean
        "uninitialized"
        """

        self.name = name
        self.pixSize = 0
        self.F = np.zeros(3)
        self.S = np.zeros(3)
        self.T = np.zeros(3)
        self.aduPerEv = 0
        self.dataPlan = None
        self.beam = source.beam()
        self.I = np.zeros([0, 0])
        self.V = np.zeros([0, 0, 3])
        self.K = np.zeros([0, 0, 3])

        self.panelList = None

    def __str__(self):

        s = ""
        s += " name = \"%s\"\n" % self.name
        s += " pixSize = %g\n" % self.pixSize
        s += " F = [%g, %g, %g]\n" % (self.F[0], self.F[1], self.F[2])
        s += " S = [%g, %g, %g]\n" % (self.S[0], self.S[1], self.S[2])
        s += " nF = %d\n" % self.nF
        s += " nS = %d\n" % self.nS
        s += " T = [%g, %g, %g]\n" % (self.T[0], self.T[1], self.T[2])
        s += " aduPerEv = %g" % self.aduPerEv
        return s

    @property
    def nF(self):
        if self.I is not None:
            return self.I.shape[1]
        return 0

    @property
    def nS(self):
        if self.I is not None:
            return self.I.shape[0]
        return 0

    @property
    def nPix(self):
        if self.I is not None:
            return self.I.size
        return 0

    def check(self):

        if self.pixSize <= 0:
            warn("Bad pixel size in panel %s" % self.name)
            return False
        return True

    def computeRealSpaceGeometry(self):

        if self.nF == 0 or self.nS == 0:
            return False

        i = np.arange(self.nF)
        j = np.arange(self.nS)
        [i, j] = np.meshgrid(i, j)
        i.flatten()
        j.flatten()
        F = np.outer(i, self.F)
        S = np.outer(j, self.S)
        self.V = np.tile(self.T, [self.nPix, 1]) + F + S

        return True


class panelList(list):

    def __init__(self):

        """ Just make an empty panel array """

        self.beam = None

    def __str__(self):

        s = ""
        for p in self:
            s += "\n\n" + p.__str__()
        return(s)

    def __getitem__(self, key):

        if isinstance(key, str):
            key = self.getPanelIndexByName(key)
            if key is None:
                raise IndexError("There is no panel named %s" % key)
                return None
        return super(panelList, self).__getitem__(key)

    def __setitem__(self, key, value):

        if not isinstance(value, panel):
            raise TypeError("You may only append panels to a panelList object")
        if value.name == "":
            value.name = "%d" % key
        super(panelList, self).__setitem__(key, value)

    def append(self, p=None, name=""):

        if p is None:
            p = panel()
        if not isinstance(p, panel):
            raise TypeError("You may only append panels to a panelList object")
        p.panelList = self
        if name != "":
            p.name = name
        else:
            p.name = "%d" % len(self)
        super(panelList, self).append(p)

    def getPanelIndexByName(self, name):

        """ Find the integer index of a panel by it's unique name """
        i = 0
        for p in self:
            if p.name == name:
                return i
            i += 1
        return None

    def computeRealSpaceGeometry(self):

        out = []
        for p in self:
            out.append(p.computeRealSpaceGeometry())

        return out


#     def check(self):
#
#         checks = [c for c in ]

    def read(self, fileName):

        self[0].dataPlan.read(self, fileName)
