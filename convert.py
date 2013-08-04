'''
Created on Aug 1, 2013

@author: kirian
'''

import detector
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

    pa = detector.panelArray()
    pa.beam = source.beam()

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
        i = pa.getPanelIndexByName(name)
        
        # If it is a new panel:
        if i == None:
            pa.append()
            p = pa[len(pa) - 1]
            p.name = name
            p.dataPlan = data.h5v1Plan()
            p.dataPlan.panel = p
        else:
            p = pa[i]

        # Parse the simple keys
        if key == "corner_x":
            p.T[0] = float(value)
        if key == "corner_y":
            p.T[1] = float(value)
        if key == "min_fs":
            p.dataPlan.fRange[0] = int(value)
        if key == "max_fs":
            p.dataPlan.fRange[1] = int(value)
        if key == "min_ss":
            p.dataPlan.sRange[0] = int(value)
        if key == "max_ss":
            p.dataPlan.sRange[1] = int(value)
        if key == "coffset":            
            p.T[2] = float(value)
        if key == "res":
            p.pixSize = 1 / float(value)

        # Parse the more complicated keys
        if key == "fs":
            value = value.split("y")[0].split("x")
            p.F[0] = float(value[0].replace(" ", ""))
            p.F[1] = float(value[1].replace(" ", ""))
        if key == "ss":
            value = value.split("y")[0].split("x")
            p.S[0] = float(value[0].replace(" ", ""))
            p.S[1] = float(value[1].replace(" ", ""))

    fh.close()


    for p in pa:
    
        # Link array beam to panel beam
        p.beam = pa.beam

        # These are defaults
        p.dataPlan.dataField = "/data/rawdata0"
        p.dataPlan.wavelengthField = "/LCLS/photon_wavelength_A"
        p.B = np.array([0, 0, 1])

        # Unit conversions
        p.T = p.T * p.pixSize

        # Check for extra global configurations
        if globalAduPerEv != None:
            p.aduPerEv = globalAduPerEv
        if globalCLen != None:
            p.T[2] += globalCLen
        if globalCLenField != None:
            p.dataPlan.detOffsetField = globalCLenField
        if globalCOffset != None:
            p.T[2] += globalCOffset
            
    return pa


def pypadTxtToPanelArray(self,fileName):

    pa = detector.panelArray()

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

