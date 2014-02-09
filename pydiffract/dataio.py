'''
Created on Jul 27, 2013

@author: kirian
'''

import h5py
import numpy as np
from pydiffract import detector

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

        self.getFrame(panelList, filePath)


    def getFrame(self, panelList, filePath):

        """ Populate a panel list with image data. """

        if self.plan is None:
            raise ValueError("You don't have a data reading plan!")

        if len(panelList) != len(self.plan):
            raise ValueError("Length of panel list does not match data reading plan!")

        f = h5py.File(filePath, "r")

        # each panel could come from a different data set within the hdf5
        # we don't want to load the data set each time, since that's slow...
        # so the code gets ugly here...
        prevDataField = self.plan[0].dataField
        dset = f[prevDataField]
        dat = np.array(dset, dtype=panelList[0].dtype)

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
                thisDataField = h.dataField
                if thisDataField != prevDataField:
                    dset = f[thisDataField]
                    dat = np.array(dset, dtype=p.dtype)
                prevDataField = thisDataField
                fmin = h.fRange[0]
                fmax = h.fRange[1] + 1
                smin = h.sRange[0]
                smax = h.sRange[1] + 1
                p.data = dat[smin:smax, fmin:fmax]


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


class saclaReader(object):

    def __init__(self, filePath=None):

        self.filePath = None  # List of file paths
        self.nFrames = None  # Total number of frames
        self.runKey = None  # ID of run
        self.fileID = None  # hdf5 file ID
        self.detectorKeys = None  # Detector panel keys
        self.shotKeys = None  # One key for each shot (!)

        if filePath is not None:

            self.setupFile(filePath)

    def setupFile(self, filePath):

        """ Do this when a new file is introduced. """

        self.filePath = filePath
        self.fileID = h5py.File(self.filePath, 'r')
        self.runKey = self.fileID.keys()[1]
        # FIXME: search for actual detector keys (no assumptions)
        self.detectorKeys = ['detector_2d_%d' % n for n in np.arange(1, 8 + 1)]
        self.shotKeys = self.fileID[self.runKey][self.detectorKeys[0]].keys()[1:]

    def getFrame(self, panelList, frameNumber):

        # FIXME: fill in this function
        pass

    def setupGeometry(self, pa):

        n = 0

        for detectorKey in self.detectorKeys:

            pixSize = np.double(self.fileID[self.runKey][detectorKey]['detector_info']['pixel_size_in_micro_meter'])[0] * 1e-6
            T = np.double(self.fileID[self.runKey][detectorKey]['detector_info']['detector_coordinate_in_micro_meter']) * 1e-6
            T[1] *= -1
            rot = -np.double(self.fileID[self.runKey][detectorKey]['detector_info']['detector_rotation_angle_in_degree']) * np.pi / 180.0
            data = np.double(self.fileID[self.runKey][detectorKey][self.shotKeys[0]]['detector_data'])
            if pa.nPanels == 0:
                p = detector.panel()
            elif pa.nPanels == len(self.detectorKeys):
                p = pa[n]
            else:
                raise ValueError("Panel list length doesn't match the hdf5 file.")
            p.T = T
            R = np.array([[np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0, 0, 1.0]])
            p.S = R.dot(np.array([0, 1, 0])) * pixSize
            p.F = R.dot(np.array([1, 0, 0])) * pixSize
            p.nF = data.shape[1]
            p.nS = data.shape[0]
            if pa.nPanels == 0:
                pa.append(p)
            n += 1


class frameGetter(object):

    """ Methods for getting data from a list of files. """

    def __init__(self, reader=None, fileList=None):

        self._fileList = None
        self._reader = None
        self._frameNumber = -1
        self._nFrames = None

        if reader is not None:
            self.reader = reader

        if fileList is not None:
            self.fileList = fileList

    @property
    def reader(self):

        """ The reader class to be used when loading frame data. """

        return self._reader

    @reader.setter
    def reader(self, value):

        self._reader = value

    @property
    def fileList(self):

        """ The file list from which to grab frames. (List object containing strings). """

        return self._fileList

    @fileList.setter
    def fileList(self, value):

        self._fileList = value
        self._nFrames = len(value)

    @property
    def frameNumber(self):

        """ The current frame number. """

        return self._frameNumber

    @frameNumber.setter
    def frameNumber(self, value):

        self._frameNumber = value

    @property
    def nFrames(self):

        """ Number of frames available. """

        if self.fileList is None:

            return 0

        return len(self.fileList)

    def loadFileList(self, fileList):

        """ Load a file list.  Should be a text file, one full file path per line. """

        self.fileList = [i.strip() for i in open(fileList).readlines()]

    def getFrame(self, panelList, frameNumber=None):

        """ Get the frame data, given index number. """

        pl = panelList
        num = frameNumber

        if num is None:
            self.frameNumber = 0
            num = self.frameNumber

        if self.fileList is None:
            raise ValueError('No file list specified')

        if num > self.nFrames - 1:
            return False

        filePath = self.fileList[num]
        self.reader.getFrame(pl, filePath)

        return num

    def nextFrame(self, panelList):

        """ Get the data from the next frame. """

        pl = panelList

        self.frameNumber += 1
        if self.frameNumber > len(self.fileList) - 1:
            self.frameNumber = 0
        return self.getFrame(pl, self.frameNumber)

    def previousFrame(self, panelList):

        """ Get the data from the previous frame. """

        pl = panelList

        self.frameNumber -= 1
        if self.frameNumber < 0:
            self.frameNumber = len(self.fileList) - 1
        return self.getFrame(pl, self.frameNumber)




















