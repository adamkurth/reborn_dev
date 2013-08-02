'''
Created on Aug 1, 2013

@author: kirian
'''

import pad
import source
import data
import utils
import numpy as np

def crystfelToPanelArray(fileName):

    # All panel-specific keys that are recognized 
    allKeys = set(["fs", "ss", "corner_x", "corner_y", "min_fs", "max_fs", "min_ss", "max_ss", "clen", "coffset", "res", "adu_per_eV"])

    fh = open(fileName, "r")
        
    # Global settings affecting all panels
    globalCOffset = None
    globalCLenField = None
    globalCLen = None
    globalAduPerEv = None

    pa = pad.panelArray()

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
        p = pa.findPanelIndexByName(name)
        
        # If it is a new panel:
        if p == None:
            pa.addPanel(None)
            p = len(pa) - 1
            pa.panels[p].name = name
            pa.panels[p].dataPlan = data.h5v1Plan()
            pa.panels[p].dataPlan.panel = pa.panels[p]

        # Parse the simple keys
        if key == "corner_x":
            pa.panels[p].T[0] = float(value)
        if key == "corner_y":
            pa.panels[p].T[1] = float(value)
        if key == "min_fs":
            pa.panels[p].dataPlan.fRange[0] = int(value)
        if key == "max_fs":
            pa.panels[p].dataPlan.fRange[1] = int(value)
        if key == "min_ss":
            pa.panels[p].dataPlan.sRange[0] = int(value)
        if key == "max_ss":
            pa.panels[p].dataPlan.sRange[1] = int(value)
        if key == "coffset":            
            pa.panels[p].T[2] = float(value)
        if key == "res":
            pa.panels[p].pixSize = 1 / float(value)

        # Parse the more complicated keys
        if key == "fs":
            value = value.split("y")[0].split("x")
            pa.panels[p].F[0] = float(value[0].replace(" ", ""))
            pa.panels[p].F[1] = float(value[1].replace(" ", ""))
        if key == "ss":
            value = value.split("y")[0].split("x")
            pa.panels[p].S[0] = float(value[0].replace(" ", ""))
            pa.panels[p].S[1] = float(value[1].replace(" ", ""))

    fh.close()

    
    # Set the global configurations
    pa.source = source.source()

    for p in range(len(pa)):
    
        # Link array beam to panel beam
        pa.panels[p].source = pa.source

        # These are defaults
        pa.panels[p].dataPlan.dataField = "/data/rawdata0"
        pa.panels[p].dataPlan.wavelengthField = "/LCLS/photon_wavelength_A"
        pa.panels[p].B = np.array([0, 0, 1])

        # Unit conversions
        pa.panels[p].T = pa.panels[p].T * pa.panels[p].pixSize

        # Check for extra global configurations
        if globalAduPerEv != None:
            pa.panels[p].aduPerEv = globalAduPerEv
        if globalCLen != None:
            pa.panels[p].T[2] += globalCLen
        if globalCLenField != None:
            pa.panels[p].dataPlan.detOffsetField = globalCLenField
        if globalCOffset != None:
            pa.panels[p].T[2] += globalCOffset
        
        fmin = pa.panels[p].dataPlan.fRange[0]
        fmax = pa.panels[p].dataPlan.fRange[1]
        smin = pa.panels[p].dataPlan.sRange[0]
        smax = pa.panels[p].dataPlan.sRange[1]
        pa.panels[p].nF = fmax - fmin + 1
        pa.panels[p].nS = smax - smin + 1
            
    return pa


def pypadTxtToPanelArray(self,fileName):

    pa = pad.panelArray()

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
            pa.addPanel(None)    

            B = np.array([0, 0, 1])
            F = np.array([ float(line[7]), float(line[8]), float(line[9]) ])            
            S = np.array([ float(line[4]), float(line[5]), float(line[6]) ])
            T = np.array([ float(line[1]), float(line[2]), float(line[3]) ])    

            pa.panels[pn].B = B
            pa.panels[pn].F = F
            pa.panels[pn].S = S
            pa.panels[pn].T = T

            pa.panels[pn].pixSize = utils.norm2(F)
            pa.panels[pn].T *= 1e-3
            
    fh.close()
    
    return pa

