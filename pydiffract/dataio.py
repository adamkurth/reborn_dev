import h5py
import numpy as np
import copy
from pydiffract import detector



class h5Reader(detector.panelList):

    """ Read an hdf5 file with "cheetah" format """

    def __init__(self):

        super(h5Reader, self).__init__()

        self.filePath = None
        self.fileList = None

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

    def loadCrystfel(self, geomFile=None, beamFile=None):

        crystfelToPanelList(geomFile=geomFile, beamFile=beamFile, panelList=self)

    def getFrame(self, frameNumber=0):

        """ Populate a panel list with image data. """

        filePath = self.fileList[frameNumber]

        f = h5py.File(filePath, "r")

        # each panel could come from a different data set within the hdf5
        # we don't want to load the data set each time, since that's slow...
        # so the code gets ugly here...
        prevDataField = self[0].dataField
        dset = f[prevDataField]
        dat = np.array(dset)

        for p in self:

            # Load wavelength from hdf5 file
            if p.wavelengthField is not None:
                p.beam.wavelength = f[p.wavelengthField].value[0] * 1e-10
            # Load camera length
            if p.detOffsetField is not None:
                p.T[2] += f[p.detOffsetField].value[0] * 1e-3
            if p.dataField is not None:
                thisDataField = p.dataField
                if thisDataField != prevDataField:
                    dset = f[thisDataField]
                    dat = np.array(dset)
                prevDataField = thisDataField
                fmin = p.fRange[0]
                fmax = p.fRange[1] + 1
                smin = p.sRange[0]
                smax = p.sRange[1] + 1
                p.data = dat[smin:smax, fmin:fmax]

        f.close()


def loadCheetahH5(pl, filePath):

    f = h5py.File(filePath, "r")

    # each panel could come from a different data set within the hdf5
    # we don't want to load the data set each time, since that's slow...
    # so the code gets ugly here...
    prevDataField = pl[0].dataField
    dset = f[prevDataField]
    dat = np.array(dset)

    for p in pl:

        # Load wavelength from hdf5 file
        if p.wavelengthField is not None:
            p.beam.wavelength = f[p.wavelengthField].value[0] * 1e-10

        # Load camera length
        if p.detOffsetField is not None:
            p.T[2] += f[p.detOffsetField].value[0] * 1e-3

        if p.dataField is not None:
            thisDataField = p.dataField
            if thisDataField != prevDataField:
                dset = f[thisDataField]
                dat = np.array(dset)
            prevDataField = thisDataField
            fmin = p.fRange[0]
            fmax = p.fRange[1] + 1
            smin = p.sRange[0]
            smax = p.sRange[1] + 1
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







def crystfelToPanelList(geomFile=None, beamFile=None, panelList=None):

    """ Convert a crystfel "geom" file into a panel list """



    # For parsing the very loose fast/slow scan vector specification
    def splitxysum(s):

        s = s.strip()
        coords = list("".join(i for i in s if i in "xyz"))
        vals = {}
        for coord in coords:
            s = s.split(coord)
            vals[coord] = float(s[0].replace(" ", ""))
            s = s[1]

        vec = [vals['x'], vals['y'], 0]
        if len(vals) > 2:
            vec[2] = vals['z']

        return vec

    # All panel-specific keys that are recognized
    all_keys = set(["fs", "ss", "corner_x", "corner_y",
                    "min_fs", "max_fs", "min_ss", "max_ss",
                    "clen", "coffset", "res", "adu_per_eV"])

    # Geometry file is required
    if geomFile is None:
        raise ValueError("No geometry file specified")

    h = open(geomFile, "r")

    # Global settings affecting all panels
    global_coffset = None
    global_clen_field = None
    global_clen = None
    global_adu_per_ev = None
    global_res = None

    if panelList is None:
        pa = h5Reader()
    else:
        pa = panelList

    # Place holder for pixel sizes
    pixSize = np.zeros(10000)

    for line in h:

        # Search for appropriate lines
        line = line.split("=")
        if len(line) != 2:
            continue

        # Split key/values
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
        if key == 'res':
            global_res = float(value)

        # If not a global key, check for panel-specific keys
        key = key.split("/")
        if len(key) != 2:
            continue

        # Split name from key/value pairs
        name = key[0].strip()
        key = key[1].strip()

        # Skip unknown panel-specific key
        if not key in all_keys:
            continue

        # Get index of this panel
        i = pa.getPanelIndexByName(name)
        # If it is a new panel:
        if i is None:
            pa.append()
            p = pa[len(pa) - 1]
            p.name = name
            p.F = np.zeros(3)
            p.S = np.zeros(3)
            p.T = np.zeros(3)
            # add some extra attributes to the panel object
            p.fRange = [0, 0]
            p.sRange = [0, 0]
            p.dataField = None
            p.wavelengthField = None
            p.detOffsetField = None
        else:
            p = pa[i]

        # Parse the simple keys
        if key == "corner_x":
            p.T[0] = float(value)
        if key == "corner_y":
            p.T[1] = float(value)
        if key == "min_fs":
            p.fRange[0] = int(value)
        if key == "max_fs":
            p.fRange[1] = int(value)
        if key == "min_ss":
            p.sRange[0] = int(value)
        if key == "max_ss":
            p.sRange[1] = int(value)
        if key == "coffset":
            p.T[2] = float(value)
        if key == "res":
            pixSize[i] = 1.0 / float(value)

        # Parse the more complicated keys
        if key == "fs":
            vec = splitxysum(value)
            p.F = np.array(vec)
        if key == "ss":
            vec = splitxysum(value)
            p.S = np.array(vec)

    h.close()

    pa.beam.B = np.array([0, 0, 1])

    i = 0
    for p in pa:

        # These are defaults
        p.dataField = "/data/rawdata0"
        p.wavelengthField = "/LCLS/photon_wavelength_A"

        # Unit conversions
        if pixSize[i] == 0:
            pixSize[i] = 1 / global_res
        p.pixSize = pixSize[i]
        p.T *= pixSize[i]

        # Data array size
        p.nF = p.fRange[1] - p.fRange[0] + 1
        p.nS = p.sRange[1] - p.sRange[0] + 1

        # Check for extra global configurations
        if global_adu_per_ev is not None:
            p.aduPerEv = global_adu_per_ev
        if global_clen is not None:
            p.T[2] += global_clen
        if global_clen_field is not None:
            p.detOffsetField = global_clen_field
        if global_coffset is not None:
            p.T[2] += global_coffset

        i += 1
        print(p.T)

    return pa
