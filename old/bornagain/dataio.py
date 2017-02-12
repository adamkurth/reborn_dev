import h5py
import numpy as np
import copy
from bornagain import detector
import re


# A "reader" should be initialized with file paths, and instructuions on how to access a
# sequence of frames.  Readers are not feature rich -- just the bare minimum methods to laod
# raw data for the sequence of frames.
# The "frameGetter" class makes access to a sequence of files opaque; shouldn't matter what type of files.

# Minimally, a reader should have the method getFrame(frameNumber=0), which should return a panelList object


class cheetahH5Reader(object):

    """ Read an hdf5 file with "cheetah" format """

    def __init__(self, panelList=None):

        if panelList is None:
            self.panelList = detector.panelList()
        else:
            self.panelList = panelList
        self._fileList = []

    @property
    def fileList(self):
        return self._fileList

    @fileList.setter
    def fileList(self, value):
        self._fileList = value

    def getFrame(self, frameNumber=0):

        """ Get a frame. """

        filePath = self.fileList[frameNumber]
        loadCheetahH5V1(self.panelList, filePath)

        return self.panelList


def loadCheetahH5V1(panelList, filePath):

    f = h5py.File(filePath, "r")

    # each panel could come from a different data set within the hdf5
    # we don't want to load the data set each time, since that's slow...
    # so the code gets ugly here...
    prevDataField = panelList[0].dataField
    dset = f[prevDataField]
    dat = np.array(dset)

    for p in panelList:

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

    return panelList

class cxidbReader(object):

    """ Read an hdf5 file with "CXIDB" format """

    def __init__(self, panelList=None, fileList=[]):

        self.panelList = panelList
        self.fileList = []
        self.hdf5Files = []

        self.dataPaths = None
        self.panelNames = None
        self.nFs = None
        self.nSs = None

        if self.fileList is not None:
            self.loadHdf5Files()

        if self.panelList is None:
            self.panelList = detector.panelList()

    def loadHdf5Files(self):

        for thisFile in self.fileList:
            self.hdf5Files.append(h5py.File(thisFile, "r"))

    def initializePanelList(self):

        pl = self.panelList
        for dp in range(len(self.dataPaths)):
            for pi in range(len(self.panelNames)):
                p = detector.panel()
                p.name = self.panelNames[dp,pi]
                p.nF = self.nFs[dp,pi]
                p.nS = self.nSs[dp,pi]
                pl.append(p)




    def getFrame(self, frameNumber=0):









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

    def __init__(self):

        self._fileList = []
        self._reader = None
        self._frameNumber = 0
        self._loop = False

    @property
    def reader(self):

        """ The reader class to be used when loading frame data. """

        return self._reader

    @reader.setter
    def reader(self, value):

        # TODO: check type
        self._reader = value

    @property
    def fileList(self):

        """ The file list from which to grab frames. (List object containing strings). """

        return self._fileList

    @fileList.setter
    def fileList(self, value):

        self._fileList = value
        self.reader.fileList = value

    @property
    def frameNumber(self):

        """ The current frame number. """

        return self._frameNumber

    @frameNumber.setter
    def frameNumber(self, value):

        if value >= self.nFrames:
            raise ValueError('Frame number out of bounds (too large).')

        if value < 0:
            raise ValueError('Frame number out of bounds (negative).')

        self._frameNumber = value

    @property
    def nFrames(self):

        """ Number of frames available. """

        return len(self.fileList)

    def loadFileList(self, fileList):

        """ Load a file list.  Should be a text file, one full file path per line. """

        if self.nFrames > 0:
            self.fileList += [i.strip() for i in open(fileList).readlines()]
            return

        self.fileList = [i.strip() for i in open(fileList).readlines()]

    def loadCrystfelGeometry(self, geomFile=None, beamFile=None):

        crystfelToPanelList(geomFile=geomFile, beamFile=beamFile, panelList=self.reader.panelList)

    def getFrame(self, frameNumber=0):

        """ Get the frame data, given index number. """

        if self.nFrames == 0:
            raise ValueError('No file list specified.')

        self.reader.getFrame(frameNumber)
        # print('getting frame %d' % (frameNumber))
        return self.reader.panelList

    def nextFrame(self):

        """ Get the data from the next frame. """

        self.frameNumber += 1

        if self.frameNumber >= self.nFrames:
            if self._loop == True:
                self.frameNumber = 0
            else:
                return None

        return self.getFrame(self.frameNumber)

    def previousFrame(self):

        """ Get the data from the previous frame. """

        fn = self._frameNumber

        if fn <= 0:
            if self._loop == True:
                self.frameNumber = self.nFrames - 1
            else:
                self.frameNumber -= 1
        else:
            self.frameNumber -= 1

        return self.getFrame(fn)


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
    global_photon_energy = None
    global_photon_energy_field = None
    global_data = None

    if panelList is None:
        pa = detector.panelList()
    else:
        pa = panelList

    # Place holder for pixel sizes
    pixSize = np.zeros(10000)

    rigidGroupNames = []
    rigidGroups = []
    rigidGroupCollectionNames = []
    rigidGroupCollections = []

    for line in h:

        line = line.strip()

        # Skip empty lines
        if len(line) == 0:
            continue

        # Skip commented lines
        if line[0] == ';':
            continue

        # Remove comments
        line = line.split(';')
        line = line[0]
        line = line.strip()

        # Search for appropriate lines, which must have an "=" character
        line = line.split("=")
        if len(line) != 2:
            continue

        # Split key/values
        value = line[1].strip()
        key = line[0].strip()

        # Check for global keys first
        if key == "coffset":
            # This will be summed with clen
            global_coffset = float(value)
        if key == "clen":
            try:
                # If a value is given
                global_clen = float(value)
            except ValueError:
                # If a path to value in hdf5 file is given
                global_clen_field = value
                global_clen = None
        if key == "adu_per_eV":
            global_adu_per_ev = float(value)
        if key == 'res':
            global_res = float(value)
        if key == "data":
            # This is the hdf5 path to the data array
            global_data = value
        if key == "photon_energy":
            try:
                # If a value is given
                global_photon_energy = float(value)
            except ValueError:
                # If a path to value in hdf5 file is given
                global_photon_energy_field = value
                global_photon_energy = None

        # For dealing with rigid groups
        if re.search("^rigid_group_collection", key):
            keymod = key.split("_")
            rigidGroupCollectionNames.append(keymod[-1])
            rigidGroupCollections.append([k.strip() for k in value.split(',')])
            continue
        if re.search("^rigid_group", key):
            keymod = key.split("_")
            rigidGroupNames.append(keymod[-1])
            rigidGroups.append([k.strip() for k in value.split(',')])
            continue

        # If not a global key, check for panel-specific keys, which always have a "/" character
        key = key.split("/")
        if len(key) != 2:
            continue

        # Split name from key/value pairs
        name = key[0].strip()
        key = key[1].strip()

        # Get index of this panel
        i = pa.getPanelIndexByName(name)
        # If it is a new panel:
        if i is None:
            # Initialize panel
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
            p.photonEnergyField = None
            p.detOffsetField = None
        else:
            p = pa[i]

        # Parse panel-specific keys
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
        if key == "fs":
            vec = splitxysum(value)
            p.F = np.array(vec)
        if key == "ss":
            vec = splitxysum(value)
            p.S = np.array(vec)

    # We are now done reading the geometry file
    h.close()

    # Initialize beam information for this panel array
    pa.beam.B = np.array([0, 0, 1])

    # Now adjust panel list according to global parameters, convert units, etc.
    i = 0
    for p in pa:

        # Unit conversions
        if pixSize[i] == 0:
            pixSize[i] = 1 / global_res
        p.pixSize = pixSize[i]
        p.T *= pixSize[i]

        # Data array size
        p.nF = p.fRange[1] - p.fRange[0] + 1
        p.nS = p.sRange[1] - p.sRange[0] + 1

        # Check for extra global configurations
        p.aduPerEv = global_adu_per_ev
        p.dataField = global_data
        p.detOffsetField = global_clen_field
        if global_clen is not None:
            p.T[2] += global_clen
            p.detOffsetField = None  # Cannot have clen value *and* path
        if global_coffset is not None:
            p.T[2] += global_coffset
            p.detOffsetField = None  # Cannot have offset value *and* field
        if global_photon_energy is not None:
            p.beam.wavelength = 1.2398e-6 / global_photon_energy  # CrystFEL uses eV units
            p.photonEnergyField = None  # Cannot have both energy value *and* field

        i += 1

    for i in range(len(rigidGroups)):
        pa.addRigidGroup(rigidGroupNames[i], rigidGroups[i])

    for i in range(len(rigidGroupCollections)):
        cn = rigidGroupCollectionNames[i]
        rgc = rigidGroupCollections[i]
        rgn = []
        for j in rgc:
            rg = pa.rigidGroup(j)
            for k in rg:
                rgn.append(k.name)
        pa.addRigidGroup(cn, rgn)

    return pa
