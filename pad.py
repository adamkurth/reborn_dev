
import numpy as np
from utils import warn
import data
import source

""" Classes for analyzing diffraction data contained in pixel array detectors (PAD) """

class panel(object):

    def __init__(self):
        
        self.name = ""
        self.pixSize = 0
        self.F = np.zeros(3)
        self.S = np.zeros(3)
        self.nF = 0
        self.nS = 0
        self.T = np.zeros(3)
        self.aduPerEv = 0
        self.dataPlan = data.genericDataPlan()
        self.source = source.source()
        self.I = None
        self.V = None
        
        self.panelArray = None
        
    def __str__(self):
        
        s = ""
        s += " name = %s\n" % self.name
        s += " pixSize = %g\n" % self.pixSize
        s += " F = [%g, %g, %g]\n" % (self.F[0],self.F[1],self.F[2])
        s += " S = [%g, %g, %g]\n" % (self.S[0],self.S[1],self.S[2])
        s += " nF = %d\n" % self.nF
        s += " nS = %d\n" % self.nS
        s += " T = [%g, %g, %g]\n" % (self.T[0],self.T[1],self.T[2])
        s += " B = [%g, %g, %g]\n" % (self.B[0],self.B[1],self.B[2])
        s += " aduPerEv = %g" % self.aduPerEv
        return s

    def check(self):

        if self.pixSize <= 0:
            warn("Bad pixel size in panel %s" % self.name)
            return False
        return True
        

    def computeGeometry(self):
        
        
        pass


class panelArray(object):

    def __init__(self):

        """ Just make an empty panel array """

        self.panels = []
        self.source = source.source()

    def __len__(self):
        
        """ Use the built-in python len() function to get the number of panels """
        
        return len(self.panels)
    
    def __str__(self):
        
        """ Print useful information on request """
        
        s = ""
        for p in range(len(self)):
            s += "\n\n"
            s += self.panels[p].__str__()
        return(s)

    def print_(self):
        for p in range(len(self)):
            print("")
            self.panels[p].print_()

    def addPanel(self,p):
        if p == None:
            p = panel()
        p.panelArray = self
        self.panels.append(p)

    def findPanelIndexByName(self,name):
        
        """ Find the integer index of a panel by it's unique name """
        
        for i in range(len(self)):
            if self.panels[i].name == name:
                return i
        return None

    def read(self,fileName):
        
        self.panels[0].dataPlan.read(self,fileName)




















