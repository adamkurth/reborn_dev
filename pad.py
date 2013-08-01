
import numpy as np
from utils import warn, norm2
import data
import source

""" Classes for analyzing diffraction data contained in pixel array detectors (PAD) """

class panel(object):

    def __init__(self):
        
        self.name = ""
        self.pixSize = 0
        self.F = np.zeros(3)
        self.S = np.zeros(3)
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

        self._lastFileOpened = None
        self._lastFileHandle = None

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

    def loadPypadTxt(self,fileName):

        fh = open(fileName,"r")
        pn = -1        

        for line in fh:
            line = line.split()
            if len(line) == 10:
                try:
                    int(line[0])
                except:
                    continue
            
                pn += 1
                self.panels.append(panel())    

                B = np.array([0, 0, 1])
                F = np.array([ float(line[7]), float(line[8]), float(line[9]) ])            
                S = np.array([ float(line[4]), float(line[5]), float(line[6]) ])
                T = np.array([ float(line[1]), float(line[2]), float(line[3]) ])    

                self.panels[pn].B = B
                self.panels[pn].F = F
                self.panels[pn].S = S
                self.panels[pn].T = T

                self.panels[pn].pixSize = norm2(F)
                self.panels[pn].T *= 1e-3
                
        fh.close()


    def loadCrystfelGeom(self, fileName):

        # All panel-specific keys that are recognized 
        allKeys = set(["fs","ss","corner_x","corner_y","min_fs","max_fs","min_ss","max_ss","clen","coffset","res","adu_per_eV"])

        fh = open(fileName,"r")
        
        # Global settings affecting all panels
        globalCOffset = None
        globalCLenField = None
        globalCLen = None
        globalAduPerEv = None

        for line in fh:
            
            line = line.split("=")
            
            if len(line) != 2:
                continue

            value = line[1].strip()
            key = line[0].strip()

            # Check for global keys first
            if key == "coffset":
                globalCOffset = float(value)
            if key == "clen":
                try:
                    globalCLen = float(value)
                except:
                    globalCLenField = value
            if key == "adu_per_eV":
                globalAduPerEv = float(value)

            # If not a global key, check for panel-specific keys
            key = key.split("/")
            if len(key) != 2:
                continue

            name = key[0].strip()
            key = key[1].strip()

            if not key in allKeys:
                continue

            # Get index of this panel
            p = self.findPanelIndexByName(name)
            
            # If it is a new panel:
            if p == None:
                self.addPanel(None)
                p = len(self) - 1
                self.panels[p].name = name
                self.panels[p].dataPlan = data.h5v1Plan()

            # Parse the simple keys
            if key == "corner_x":
                self.panels[p].T[0] = float(value)
            if key == "corner_y":
                self.panels[p].T[1] = float(value)
            if key == "min_fs":
                self.panels[p].dataPlan.fRange[0] = int(value)
            if key == "max_fs":
                self.panels[p].dataPlan.fRange[1] = int(value)
            if key == "min_ss":
                self.panels[p].dataPlan.sRange[0] = int(value)
            if key == "max_ss":
                self.panels[p].dataPlan.sRange[1] = int(value)
            if key == "coffset":            
                self.panels[p].T[2] = float(value)
            if key == "res":
                self.panels[p].pixSize = 1/float(value)

            # Parse the more complicated keys
            if key == "fs":
                value = value.split("y")[0].split("x")
                self.panels[p].F[0] = float(value[0].replace(" ",""))
                self.panels[p].F[1] = float(value[1].replace(" ",""))
            if key == "ss":
                value = value.split("y")[0].split("x")
                self.panels[p].S[0] = float(value[0].replace(" ",""))
                self.panels[p].S[1] = float(value[1].replace(" ",""))

        fh.close()

        
        # Set the global configurations
        self.source = source.source()

        for p in range(len(self)):
        
            # Link array beam to panel beam
            self.panels[p].source = self.source
    
            # These are defaults
            self.panels[p].dataPlan.dataField = "/data/rawdata0"
            self.panels[p].dataPlan.wavelengthField = "/LCLS/photon_wavelength_A"
            self.panels[p].B = np.array([0,0,1])

            # Unit conversions
            self.panels[p].T = self.panels[p].T*self.panels[p].pixSize

            # Check for extra global configurations
            if globalAduPerEv != None:
                self.panels[p].aduPerEv = globalAduPerEv
            if globalCLen != None:
                self.panels[p].T[2] += globalCLen
            if globalCLenField != None:
                self.panels[p].dataPlan.detOffsetField = globalCLenField
            if globalCOffset != None:
                self.panels[p].T[2] += globalCOffset



















