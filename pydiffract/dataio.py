import h5py
import numpy as np
import copy
from pydiffract import detector


class h5v1Plan(object):

    """ A container for the various information needed to pull 
    out the data corresponding to a detector panel."""

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
        self.nFrames = 1

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
        dat = np.array(dset)

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
                    dat = np.array(dset)
                prevDataField = thisDataField
                fmin = h.fRange[0]
                fmax = h.fRange[1] + 1
                smin = h.sRange[0]
                smax = h.sRange[1] + 1
                p.data = dat[smin:smax, fmin:fmax]


        f.close()


class diproiReader(detector.panelList):

    def __init__(self):

        super(diproiReader, self).__init__()

        self.filePath = None
        self.fileList = None
        self.dataField = None
        self.append()

    def deepcopy(self):

        pa = copy.deepcopy(self)

        return pa

    @property
    def nFrames(self):

        if self.fileList is None:
            return 0

        return len(self.fileList)

    def loadFileList(self, fileList):

        self.fileList = loadFileList(fileList)

    def getFrame(self, frameNumber=0):

        self.filePath = self.fileList[frameNumber]
        f = h5py.File(self.filePath, "r")
        self[0].data = np.array(f[self.dataField])
        f.close()


class saclaReader(object):

    def __init__(self, filePath=None):

        self.filePath = None  # List of file paths
        self.nFrames = None  # Total number of frames
        self.runKey = None  # ID of run
        self.fileID = None  # hdf5 file ID
        self.detectorKeys = None  # Detector panel keys
        self.shotKeys = None  # One key for each shot (!)
        self.dummyPanelList = detector.panelList()  # Dummy panel list for geometry values

        if filePath is not None:

            self.setupFile(filePath)

    def setupFile(self, filePath):

        """ Do this when a new file is introduced. Check contents of hdf5 file, 
        get geometry information. """

        self.filePath = filePath
        self.fileID = h5py.File(self.filePath, 'r')
        self.runKey = self.fileID.keys()[1]
        # FIXME: search for actual detector keys (no assumptions)
        self.detectorKeys = ['detector_2d_%d' % n for n in np.arange(1, 8 + 1)]
        self.shotKeys = self.fileID[self.runKey][self.detectorKeys[0]].keys()[1:]
        self.nFrames = len(self.shotKeys)
        self.setupGeometry(self.dummyPanelList)

    def getFrame(self, panelList, frameNumber):

        """ Populate the panelList data with intensities from given frame number. """

        pa = panelList
        if pa.geometryHash != self.dummyPanelList.geometryHash:
            self.setupGeometry(pa)

        n = 0
        for detectorKey in self.detectorKeys:
            pa[n].data = np.double(self.fileID[self.runKey][detectorKey][self.shotKeys[frameNumber]]['detector_data'])
            n += 1

    def setupGeometry(self, panelList):

        """ Geometry configuration for a particular hdf5 file. """

        n = 0
        pa = panelList
        if pa is None:
            pa = detector.panelList()

        for detectorKey in self.detectorKeys:

            pixSize = np.double(self.fileID[self.runKey][detectorKey]['detector_info']['pixel_size_in_micro_meter'])[0] * 1e-6
            T = np.double(self.fileID[self.runKey][detectorKey]['detector_info']['detector_coordinate_in_micro_meter']) * 1e-6
            T[1] *= -1
            rot = -np.double(self.fileID[self.runKey][detectorKey]['detector_info']['detector_rotation_angle_in_degree']) * np.pi / 180.0
            data = np.double(self.fileID[self.runKey][detectorKey][self.shotKeys[0]]['detector_data'])
            if pa.nPanels == n:
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
            if pa.nPanels == n:
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


def loadFileList(fileList):

    """ Create a list of file paths from a file with one path per line. """

    return [i.strip() for i in open(fileList).readlines()]


def crystfelToPanelList(geomFilePath=None, beamFilePath=None):

    """ Convert a crystfel "geom" file into a panel list """

    # All panel-specific keys that are recognized
    all_keys = set(["fs", "ss", "corner_x", "corner_y",
                    "min_fs", "max_fs", "min_ss", "max_ss",
                    "clen", "coffset", "res", "adu_per_eV"])

    if geomFilePath is None:
        raise ValueError("No geometry file specified")

    h = open(geomFilePath, "r")

    # Global settings affecting all panels
    global_coffset = None
    global_clen_field = None
    global_clen = None
    global_adu_per_ev = None

    pa = detector.panelList()
    ra = []
    pixSize = np.ones(10000)

    for line in h:

        line = line.split("=")

        if len(line) != 2:
            continue

        value = line[1].strip()
        key = line[0].strip()

        # Check for global keys first
        if key == "coffset":
            global_coffset = float(value)
        if key == "clen":
            try:
                global_clen = float(value)
            except ValueError:
                global_clen_field = value
        if key == "adu_per_eV":
            global_adu_per_ev = float(value)

        # If not a global key, check for panel-specific keys
        key = key.split("/")
        if len(key) != 2:
            continue

        name = key[0].strip()
        key = key[1].strip()

        if not key in all_keys:
            continue

        # Get index of this panel
        i = pa.getPanelIndexByName(name)

        # If it is a new panel:
        if i is None:
            pa.append()
            p = pa[len(pa) - 1]
            ra.append(h5v1Plan())
            r = ra[len(pa) - 1]
            p.name = name
            p.T = np.zeros(3)
            p.F = np.zeros(3)
            p.S = np.zeros(3)
        else:
            p = pa[i]
            r = ra[i]

        # Parse the simple keys
        if key == "corner_x":
            p.T[0] = float(value)
        if key == "corner_y":
            p.T[1] = float(value)
        if key == "min_fs":
            r.fRange[0] = int(value)
        if key == "max_fs":
            r.fRange[1] = int(value)
        if key == "min_ss":
            r.sRange[0] = int(value)
        if key == "max_ss":
            r.sRange[1] = int(value)
        if key == "coffset":
            p.T[2] = float(value)
        if key == "res":
            pixSize[i] = 1.0 / float(value)

        # Parse the more complicated keys
        if key == "fs":
            value = value.split("y")[0].split("x")
            p.F[0] = float(value[0].replace(" ", ""))
            p.F[1] = float(value[1].replace(" ", ""))
        if key == "ss":
            value = value.split("y")[0].split("x")
            p.S[0] = float(value[0].replace(" ", ""))
            p.S[1] = float(value[1].replace(" ", ""))

    h.close()



    for i in range(len(pa)):

        p = pa[i]
        r = ra[i]

        # Link array beam to panel beam
        p.beam = pa.beam

        # These are defaults
        r.dataField = "/data/rawdata0"
        r.wavelengthField = "/LCLS/photon_wavelength_A"
        p.beam.B = np.array([0, 0, 1])

        # Unit conversions
        p.T = p.T * pixSize[i]
        p.pixSize = pixSize[i]  # (this modifies F and S vectors)
#         p.F = p.F * pixSize[i]
#         p.S = p.S * pixSize[i]

        # Data array size
        p.nF = r.fRange[1] - r.fRange[0] + 1
        p.nS = r.sRange[1] - r.sRange[0] + 1

        # Check for extra global configurations
        if global_adu_per_ev is not None:
            p.aduPerEv = global_adu_per_ev
        if global_clen is not None:
            p.T[2] += global_clen
        if global_clen_field is not None:
            r.detOffsetField = global_clen_field
        if global_coffset is not None:
            p.T[2] += global_coffset

    reader = h5v1Reader()
    reader.setPlan(ra)

    return [pa, reader]











