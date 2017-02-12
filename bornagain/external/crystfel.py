import    numpy         as np
from      bornagain import detector
import                     re


def geomToDict(geomFile=None,rawStrings=False):

	""" Convert a crystfel "geom" file into a sensible Python dictionary.  For details
		see:
		
		http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html.
		
		By default, this will convert values to floats, integers, or bools, as
		appropriate. Use the option "rawStrings = True" to load only raw strings."""

	def interpretVectorString(s):

		""" Parse fast/slow scan vectors """

		s = s.strip()
		coords = list("".join(i for i in s if i in "xyz"))
		vals = {}
		for coord in coords:
			s = s.split(coord)
			for i in range(len(s)):
				s[i] = s[i].strip()
			if s[0] == '':   # CrystFEL allows simply "x" in place of "1x"
				s[0] = 1
			vals[coord] = float(s[0])
			s = s[1]

		vec = [0,0,0]
		if 'x' in vals.keys():
			vec[0] = vals['x']
		if 'y' in vals.keys():
			vec[1] = vals['y']
		if 'z' in vals.keys():
			vec[2] = vals['z']

		return vec

	def seekDictionaryByName(dictionaryList,dictionaryName):

		""" We will deal with lists of dictionaries.  Here is how we seek a specific entry in the list by name """

		theDictionary = None
		for item in dictionaryList:
			if item['name'] == dictionaryName:
				theDictionary = item
		return theDictionary

	def convertTypes(mydict):

		""" Convert dictionary entries into numeric types, as appropriate """

		for key in mydict.keys():
			# Convert floating point values
			if key in set(['adu_per_eV','res','clen','coffset','corner_x','corner_y','max_adu','photon_energy','photon_energy_scale']):
				try:
					mydict[key] = float(mydict[key])
					continue
				except:
					continue
			# Integers
			if key in set(['min_fs','min_ss','max_fs','max_ss']):
				try:
					mydict[key] = int(mydict[key])
					continue
				except:
					continue
			# Vectors
			if key in set(['fs','ss']):
				try:
					mydict[key] = interpretVectorString(mydict[key])
					continue
				except:
					continue
			# Boolean
			if key in set(['no_index']):
				if mydict[key] == '1':
					mydict[key] = True
				else:
					mydict[key] = False
			continue

	# These are the panel keys allowed by crystfel
	panelKeys = ['name','data','dim0','dim1','dim2','min_fs','min_ss','max_fs','max_ss','adu_per_eV','badrow_direction','res','clen','coffset','fs','ss','corner_x','corner_y','max_adu','no_index','mask','mask_file','saturation_map','saturation_map_file','mask_good','mask_bad','photon_energy','photon_energy_scale']

	# Here is our template panel dictionary.  Also for the global panel dictionary,
	# which fills in the undefined properties of panels.
	panelDictTemplate = {}
	# Initialized with None for all keys
	for key in panelKeys:
		panelDictTemplate[key] = None
	
	# These are the bad region keys allowed by crystfel
	badRegionKeys = ['name','min_x','max_x','min_y','max_y','panel']

	# Initialize a template dictionary for bad regions.
	badRegionDictTemplate = {}
	for key in badRegionKeys:
		badRegionDictTemplate[key] = None

	# Initialize a template dictionary for rigid groups
	rigidGroupDictTemplate = {'name' : None, 'panels' : []}
	
	# Initialize a template dictionary for rigid group collections
	rigidGroupCollectionDictTemplate = {'name' : None, 'groups' : []}

	# Initialize the lists we will populate as we scan the file
	globals = panelDictTemplate.copy()
	panels = []
	rigidGroups = []
	rigidGroupCollections = []
	badRegions = []

	# Now begin scanning the file

	if geomFile is None:
		raise ValueError("No geometry file specified")

	h = open(geomFile, "r")

	for originalline in h:

		# Avoid any issues with whitespace
		line = originalline.strip()

		# Skip empty lines
		if len(line) == 0:
			continue

		# Skip comment lines
		if line[0] == ';':
			continue

		# Remove trailing comments
		line = line.split(';')
		line = line[0]
		line = line.strip()

		# Search for appropriate lines.  Firstly, we must have an "=" character
		line = line.split("=")
		if len(line) != 2:
			continue

		# Split key/values
		value = line[1].strip()
		key = line[0].strip()

		# Check for global key
		if key in set(panelDictTemplate.keys()):
			globals[key] = value
			continue

		# Check for rigid group collection
		if re.search("^rigid_group_collection", key):
			name = key.split("_")[-1]
			thisDict = seekDictionaryByName(rigidGroupCollections,name)
			if thisDict is None:
				thisDict = rigidGroupCollectionDictTemplate.copy()
				thisDict['name'] = name
				rigidGroupCollections.append(thisDict)
			thisDict['groups'] = ([k.strip() for k in value.split(',')])
			continue

		# Check for rigid group
		if re.search("^rigid_group", key):
			name = key.split("_")[-1]
			thisDict = seekDictionaryByName(rigidGroups,name)
			if thisDict is None:
				thisDict = rigidGroupDictTemplate.copy()
				thisDict['name'] = name
				rigidGroups.append(thisDict)
			thisDict['panels'] = ([k.strip() for k in value.split(',')])
			continue

		# Check for bad region
		if re.search("^bad", key):
			key = key.split("/")
			name = key[0].strip()
			key = key[1].strip()
			thisDict = seekDictionaryByName(badRegions,name)
			if thisDict is None:
				thisDict = badRegionDictTemplate.copy()
				thisDict['name'] = name
				badRegions.append(thisDict)
			thisDict[key] = value
			continue

		# If not any of the above types, check if it is a panel-specific specification
		key = key.split("/")
		if len(key) != 2:
			print('Cannot interpret string %s' % originalline)
			continue

		# Split name from key/value pairs
		name = key[0].strip()
		key = key[1].strip()
		
		if key in set(panelKeys):
			thisDict = seekDictionaryByName(panels,name)
			if thisDict is None:
				thisDict = panelDictTemplate.copy()
				thisDict['name'] = name
				panels.append(thisDict)
			thisDict[key] = value
			continue
		
		print("Cannot interpret: %s" % originalline)

	# Convert strings to numeric types as appropriate
	if rawStrings is not True:
		for panel in panels:
			convertTypes(panel)
		convertTypes(globals)
		for badRegion in badRegions:
			convertTypes(badRegion)

	# Populate the missing values in each panel with global values
	for panel in panels:
		for key in panelKeys:
			if panel[key] is None:
				panel[key] = globals[key]

	# Now package up the results and return
	geomDict = {'globals' : globals, 'panels' : panels, 'rigidGroups' : rigidGroups, 'rigidGroupCollections' : rigidGroupCollections, 'badRegions' : badRegions}
	return geomDict
	

def geomDictToPanelList(geomDict):

	""" Convert a CrystFEL geometry dictionary to a PanelList object. """

	PanelList = detector.PanelList()
	PanelList.beam.B = np.array([0, 0, 1])
	
	for p in geomDict['panels']:

		panel = detector.panel()

		panel.name = p['name']
		panel.F = np.array(p['fs']) # Unit vectors
		panel.S = np.array(p['ss'])
		panel.nF = p['max_fs'] - p['min_fs'] + 1
		panel.nS = p['max_ss'] - p['min_ss'] + 1
		z = 0.0
		if type(p['coffset']) is not str:
			z += p['coffset']
		panel.T = np.array([p['corner_x']/p['res'],p['corner_y']/p['res'],z])
		panel.pixelSize = 1.0 / p['res']
		panel.aduPerEv = p['adu_per_eV']
		PanelList.append(panel)

	return PanelList

#        if key == "coffset":
#            p.T[2] += float(value)
#        if key == "clen":
#            p.T[2] += float(value)
#		
#    # Now adjust panel list according to global parameters, convert units, etc.
#    i = 0
#    for p in pa:
#
#        # Unit conversions
#        if pixelSize[i] == 0:
#            if global_res is not None:
#                pixelSize[i] = 1 / global_res
#        p.pixelSize = pixelSize[i]
#        p.T[0:2] *= pixelSize[i]
#
#        # Data array size
#        p.nF = p.fRange[1] - p.fRange[0] + 1
#        p.nS = p.sRange[1] - p.sRange[0] + 1
#
#        # Check for extra global configurations
#        p.aduPerEv = global_adu_per_ev
#        p.dataField = global_data
#        p.detOffsetField = global_clen_field
#        if global_clen is not None:
#            p.T[2] += global_clen
#            p.detOffsetField = None  # Cannot have clen value *and* path
#        if global_coffset is not None:
#            p.T[2] += global_coffset
#            p.detOffsetField = None  # Cannot have offset value *and* field
#        if global_photon_energy is not None:
#            p.beam.wavelength = 1.2398e-6 / global_photon_energy  # CrystFEL uses eV units
#            p.photonEnergyField = None  # Cannot have both energy value *and* field
#
#        i += 1
#
#    for i in range(len(rigidGroups)):
#        pa.addRigidGroup(rigidGroupNames[i], rigidGroups[i])
#
#    for i in range(len(rigidGroupCollections)):
#        cn = rigidGroupCollectionNames[i]
#        rgc = rigidGroupCollections[i]
#        rgn = []
#        for j in rgc:
#            rg = pa.rigidGroup(j)
#            for k in rg:
#                rgn.append(k.name)
#        pa.addRigidGroup(cn, rgn)
#
#    return pa


def loadCheetahMask(maskArray,PanelList,geomDict):

	""" Populate panel masks with cheetah mask array. """

	fail = False

	for p in geomDict['panels']:
		PanelList[p['name']].mask = maskArray[p['min_ss']:(p['max_ss']+1),p['min_fs']:(p['max_fs']+1)]

	return fail


def geomToPanelList(geomFile=None, beamFile=None, PanelList=None):

    """ Convert a crystfel "geom" file into a panel list """

    # For parsing the very loose fast/slow scan vector specification
    def splitxysum(s):

        s = s.strip()
        coords = list("".join(i for i in s if i in "xyz"))
        vals = {}
        for coord in coords:
            s = s.split(coord)
            for i in range(len(s)):
                s[i] = s[i].strip()
            if s[0] == '':   # CrystFEL allows simply "x" in place of "1x"
                s[0] = 1
            vals[coord] = float(s[0])
            s = s[1]

        vec = [0,0,0]
        if 'x' in vals.keys():
            vec[0] = vals['x']
        if 'y' in vals.keys():
            vec[1] = vals['y']
        if 'z' in vals.keys():
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

    if PanelList is None:
        pa = detector.PanelList()
    else:
        pa = PanelList

    # Place holder for pixel sizes
    pixelSize = np.zeros(10000)

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

        # Specification of "bad" regions... we don't handle these for now
        if re.search("^bad_", key):
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
            p.T[2] += float(value)
        if key == "clen":
            p.T[2] += float(value)
        if key == "res":
            pixelSize[i] = 1.0 / float(value)
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
        if pixelSize[i] == 0:
            if global_res is not None:
                pixelSize[i] = 1 / global_res
        p.pixelSize = pixelSize[i]
        p.T[0:2] *= pixelSize[i]

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

