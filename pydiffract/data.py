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
        s += "type : %s\n" % self.type
        s += "origin : %d\n" % self.origin
        s += "stride : %d\n" % self.stride
        s += "nf : %d\n" % self.nf
        s += "ns : %d\n" % self.ns
        return s

    def check(self):

        if self.nf == 0:
            return False
        if self.ns == 0:
            return False
        return True


class h5v1Plan(object):

    def __init__(self):

        self.fRange = [0, 0]
        self.sRange = [0, 0]
        self.dataField = ""
        self.wavelengthField = ""
        self.detOffsetField = ""

    def __str__(self):

        s = ""
        s += "fRange : [%d, %d]\n" % (self.fRange[0], self.fRange[1])
        s += "sRange : [%d, %d]\n" % (self.sRange[0], self.sRange[1])
        s += "dataField : %s\n" % self.dataField
        s += "wavelengthField : %s\n" % self.wavelengthField
        s += "detOffsetField : %s" % self.detOffsetField
        return s

    def check(self):

        if self.fRange[1] == 0:
            return False
        if self.sRange[1] == 0:
            return False
        if self.dataField == "":
            return False
        return True


class h5v1Reader(object):

    def __init__(self):

        self.plan = None

    def setPlan(self, plan):

        self.plan = plan

    def getShot(self, panelList, filePath):

        if self.plan is None:
            raise ValueError("You don't have a data reading plan!")

        if len(panelList) != len(self.plan):
            raise ValueError("Length of panel list does not match data reading plan!")

        f = h5py.File(filePath, "r")

        for i in range(len(panelList)):
            p = panelList[i]
            h = self.plan[i]
            # Load wavelength from hdf5 file
            if h.wavelengthField != "":
                p.beam.wavelength = f[h.wavelengthField].value[0] * 1e-10
            # Load camera length
            if h.detOffsetField != "":
                p.T[2] += f[h.detOffsetField].value[0] * 1e-3
            if h.dataField != "":
                dset = f[h.dataField]
                fmin = h.fRange[0]
                fmax = h.fRange[1] + 1
                smin = h.sRange[0]
                smax = h.sRange[1] + 1
                p.data = np.array(dset[smin:smax, fmin:fmax], dtype=p.dtype)

        f.close()
