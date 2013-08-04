
import numpy as np
from utils import warn

""" Classes for analyzing diffraction data contained in pixel array detectors (PAD) """

class panel(object):

    """ Individual detector panel, assumed to be a flat 2D array of square pixels. """

    def __init__(self,name=""):
        
        """ There are no default initialization parameters. """
        
        self.name = name
        self.pixSize = 0
        self.F = np.zeros(3)
        self.S = np.zeros(3)
        self.T = np.zeros(3)
        self.aduPerEv = 0
        self.dataPlan = None
        self.beam = None
        self.I = None
        self.V = None
        self.K = None
        
        self.panelArray = None
        
    def __str__(self):
        
        s = ""
        s += " name = \"%s\"\n" % self.name
        s += " pixSize = %g\n" % self.pixSize
        s += " F = [%g, %g, %g]\n" % (self.F[0],self.F[1],self.F[2])
        s += " S = [%g, %g, %g]\n" % (self.S[0],self.S[1],self.S[2])
        s += " nF = %d\n" % self.nF
        s += " nS = %d\n" % self.nS
        s += " T = [%g, %g, %g]\n" % (self.T[0],self.T[1],self.T[2])
        s += " aduPerEv = %g" % self.aduPerEv
        return s

    @property
    def nF(self):
        if self.I != None:
            return self.I.shape[1]
        return 0
    
    @property
    def nS(self):
        if self.I != None:
            return self.I.shape[0]
        return 0

    def check(self):

        if self.pixSize <= 0:
            warn("Bad pixel size in panel %s" % self.name)
            return False
        return True
        

    def computeRealSpaceGeometry(self):
        
        if self.nF == 0 or self.nS == 0:
            return False
        X = np.arange(0,self.nF)
        Y = np.arange(0,self.nS)
        [X,Y] = np.meshgrid(X,Y)
        return [X,Y]


class panelArray(list):

    def __init__(self):

        """ Just make an empty panel array """

        self.beam = None
    
    def __str__(self):
        
        s = ""
        for p in self: s += "\n\n" + p.__str__()
        return(s)

    def __getitem__(self,key):
        
        if isinstance(key,str):
            key = self.getPanelIndexByName(key)
            if key == None:
                raise IndexError("There is no panel named %s" % key)
                return None
        return super(panelArray,self).__getitem__(key)

    def __setitem__(self,key,value):
        
        if not isinstance(value,panel):
            raise TypeError("You may only append panel type to a panelArray object")
        if value.name == "":
            value.name = "%d" % key
        super(panelArray,self).__setitem__(key,value)

    def append(self,p=None,name=""):
        
        if p == None:
            p = panel()
        if not isinstance(p,panel):
            raise TypeError("You may only append panel type to a panelArray object")
        p.panelArray = self
        if name != "":
            p.name = name
        else:
            p.name = "%d" % len(self)
        super(panelArray,self).append(p)

    def getPanelIndexByName(self,name):
        
        """ Find the integer index of a panel by it's unique name """
        i = 0
        for p in self:
            if p.name == name:
                return i
            i += 1
        return None

    def read(self,fileName):
        
        self[0].dataPlan.read(self,fileName)
