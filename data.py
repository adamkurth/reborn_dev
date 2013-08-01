'''
Created on Jul 27, 2013

@author: kirian
'''

import h5py
import numpy as np


class genericDataPlan(object):
    
    def __init__(self):
        
        self.type = "genericv1"
        self.origin = 0
        self.stride = 0
        self.nf = 0
        self.ns = 0
        
    def __str__(self):
        
        s = ""
        s += " type : %s\n" % self.type
        s += " origin : %d\n" % self.origin
        s += " stride : %d\n" % self.stride
        s += " nf : %d\n" % self.nf
        s += " ns : %d\n" % self.ns
        return s

    def check(self):
        
        if self.nf == 0:
            return False
        if self.ns == 0:
            return False
        return True


class h5v1Plan(object):

    def __init__(self):

        self.type = "h5v1"
        self.fRange = [0,0]
        self.sRange = [0,0]
        self.dataField = ""
        self.wavelengthField = ""
        self.detOffsetField = ""
        self.read = h5v1Read
    
    def __str__(self):

        s = ""
        s += " type : %s\n" % self.type
        s += " fRange : [%d, %d]\n" % (self.fRange[0],self.fRange[1])
        s += " sRange : [%d, %d]\n" % (self.sRange[0],self.sRange[1])
        s += " dataField : %s\n" % self.dataField
        s += " wavelengthField : %s\n" % self.wavelengthField
        s += " detOffsetField : %s" % self.detOffsetField
        return s

    def check(self):
        
        if self.fRange[1] == 0:
            return False
        if self.sRange[1] == 0:
            return False
        if self.dataField == "":
            return False
        return True


def h5v1Read(filePath,panelArray):

    f = h5py.File(filePath,"r")
    for p in range(len(panelArray)):
        # Load wavelength from hdf5 file
        if panelArray.panels[p].dataPlan.wavelengthField != "":
            panelArray.panels[p].source.wavelength = f[panelArray.panels[p].dataPlan.wavelengthField].value[0]*1e-10
        # Load camera length
        if panelArray.panels[p].dataPlan.detOffsetField != "":
            panelArray.panels[p].T[2] += f[panelArray.panels[p].dataPlan.detOffsetField].value[0]*1e-3
        if panelArray.panels[p].dataPlan.dataField != "":
            dset = f[panelArray.panels[p].dataPlan.dataField]
            fmin = panelArray.panels[p].dataPlan.fRange[0]
            fmax = panelArray.panels[p].dataPlan.fRange[1]+1
            smin = panelArray.panels[p].dataPlan.sRange[0]
            smax = panelArray.panels[p].dataPlan.sRange[1]+1
            panelArray.panels[p].I = np.array(dset[smin:smax,fmin:fmax],dtype=np.double)

    f.close()
