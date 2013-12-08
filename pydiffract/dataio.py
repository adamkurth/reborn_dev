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
        self.dataField = None
        self.wavelengthField = None
        self.detOffsetField = None

    def __str__(self):

        s = ""
        s += "fRange : [%d, %d]\n" % (self.fRange[0], self.fRange[1])
        s += "sRange : [%d, %d]\n" % (self.sRange[0], self.sRange[1])
        if self.dataField is not None:
            s += "dataField : %s\n" % self.dataField
        if self.wavelengthField is not None:
            s += "wavelengthField : %s\n" % self.wavelengthField
        if self.detOffsetField is not None:
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

    """ Read an hdf5 file with "cheetah" format """

    def __init__(self):

        self.plan = None

    def setPlan(self, plan):

        self.plan = plan

    def getShot(self, panelList, filePath):

        """ Populate a panel list with image data. """

        if self.plan is None:
            raise ValueError("You don't have a data reading plan!")

        if len(panelList) != len(self.plan):
            raise ValueError("Length of panel list does not match data reading plan!")

        f = h5py.File(filePath, "r")

        for i in range(len(panelList)):
            p = panelList[i]
            h = self.plan[i]
            # Load wavelength from hdf5 file
            if h.wavelengthField is not None:
                p.beam.wavelength = f[h.wavelengthField].value[0] * 1e-10
            # Load camera length
            if h.detOffsetField is not None:
                p.T[2] += f[h.detOffsetField].value[0] * 1e-3
            if h.dataField is not None:
                dset = f[h.dataField]
                fmin = h.fRange[0]
                fmax = h.fRange[1] + 1
                smin = h.sRange[0]
                smax = h.sRange[1] + 1
                p.data = np.array(dset[smin:smax, fmin:fmax], dtype=p.dtype)

        f.close()


class diproiPlan(object):

    def __init__(self):

        self.dataField = None
        self.wavelengthField = None

    def __str__(self):

        s = ""
        s += "fRange : [%d, %d]\n" % (self.fRange[0], self.fRange[1])
        s += "sRange : [%d, %d]\n" % (self.sRange[0], self.sRange[1])
        if self.dataField is not None:
            s += "dataField : %s\n" % self.dataField
        if self.wavelengthField is not None:
            s += "wavelengthField : %s\n" % self.wavelengthField
        if self.detOffsetField is not None:
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


class diproiReader(object):

    def __init__(self):

        self.plan = None

    def setPlan(self, plan):

        if not isinstance(plan, list):
            self.plan = [plan]
        else:
            self.plan = plan

    def getFrame(self, panelList, filePath):

        if self.plan is None:
            raise ValueError("No data reading plan specified.")

        if len(panelList) != len(self.plan):
            raise ValueError("Length of panel list does not match data reading plan.")


        f = h5py.File(filePath, "r")

        for i in range(len(panelList)):
            p = panelList[i]
            h = self.plan[i]
            # Load wavelength from hdf5 file
#             if h.wavelengthField is not None:
#                 p.beam.wavelength = f[h.wavelengthField].value[0] * 1e-10
            if h.dataField is not None:
                p.data = np.array(f[h.dataField], dtype=p.dtype)



        f.close()


class frameGetter(object):

    def __init__(self):

        self.fileList = None
        self.reader = None
        self.frameNumber = -1

    def getFrame(self, pl, num=None):

        if num is None:
            self.frameNumber += 1
            num = self.frameNumber

        if self.fileList is None:
            raise ValueError('No file list specified')

        if num > len(self.fileList) - 1:
            return False

        filePath = self.fileList[num]
        self.reader.getFrame(pl, filePath)

        return num

    def nextFrame(self, pl):

        self.frameNumber += 1
        if self.frameNumber > len(self.fileList) - 1:
            self.frameNumber = 0
        return self.getFrame(pl, self.frameNumber)

    def previousFrame(self, pl):

        self.frameNumber -= 1
        if self.frameNumber < 0:
            self.frameNumber = len(self.fileList) - 1
        return self.getFrame(pl, self.frameNumber)




















