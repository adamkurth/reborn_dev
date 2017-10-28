import re

import numpy as np

from bornagain import detector


def geom_to_dict(geom_file, raw_strings=False):
    r"""
    Convert a crystfel "geom" file into a sensible Python dictionary.  For details see:
    http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html.

    Arguments:
        geom_file (str) :
            Path to the geometry file.
        raw_strings (bool) :
            By default, this function will convert values to floats, integers, or bools, as appropriate.
            Set this option to "True" if you want to store the raw strings in the output dictionary.

    Returns:
        geom_dict (dict) :
            Dictionary container filled with geom file information.
    """

    def interpret_vector_string(s):
        r""" Parse fast/slow scan vectors """

        s = s.strip()
        coords = list("".join(i for i in s if i in "xyz"))
        vals = {}
        for coord in coords:
            s = s.split(coord)
            for i in range(len(s)):
                s[i] = s[i].strip()
            if s[0] == '':  # CrystFEL allows simply "x" in place of "1x"
                s[0] = 1
            vals[coord] = float(s[0])
            s = s[1]

        vec = [0, 0, 0]
        if 'x' in vals.keys():
            vec[0] = vals['x']
        if 'y' in vals.keys():
            vec[1] = vals['y']
        if 'z' in vals.keys():
            vec[2] = vals['z']

        return vec

    def seek_dictionary_by_name(dictionaryList, dictionaryName):
        r""" We will deal with lists of dictionaries.  Here is how we seek a
        specific entry in the list by name """

        theDictionary = None
        for item in dictionaryList:
            if item['name'] == dictionaryName:
                theDictionary = item
        return theDictionary

    def convert_types(mydict):
        """ Convert dictionary entries into numeric types, as appropriate """

        for key in mydict.keys():
            # Convert floating point values
            if key in {'adu_per_eV', 'res', 'clen', 'coffset', 'corner_x',
                       'corner_y', 'max_adu', 'photon_energy',
                       'photon_energy_scale'}:
                try:
                    mydict[key] = float(mydict[key])
                    continue
                except:
                    continue
            # Integers
            if key in {'min_fs', 'min_ss', 'max_fs', 'max_ss'}:
                try:
                    mydict[key] = int(mydict[key])
                    continue
                except:
                    continue
            # Vectors
            if key in {'fs', 'ss'}:
                try:
                    mydict[key] = interpret_vector_string(mydict[key])
                    continue
                except:
                    continue
            # Boolean
            if key in {'no_index'}:
                if mydict[key] == '1':
                    mydict[key] = True
                else:
                    mydict[key] = False
            continue

    # These are the Panel keys allowed by crystfel
    panelKeys = ['name', 'data', 'dim0', 'dim1', 'dim2', 'min_fs', 'min_ss',
                 'max_fs', 'max_ss', 'adu_per_eV', 'badrow_direction', 'res',
                 'clen', 'coffset', 'fs', 'ss',
                 'corner_x', 'corner_y', 'max_adu', 'no_index', 'mask',
                 'mask_file', 'saturation_map', 'saturation_map_file',
                 'mask_good', 'mask_bad', 'photon_energy',
                 'photon_energy_scale']

    # Here is our template Panel dictionary.  Also for the global Panel
    # dictionary, which fills in the undefined properties of panels.
    panelDictTemplate = {}
    # Initialized with None for all keys
    for key in panelKeys:
        panelDictTemplate[key] = None

    # These are the bad region keys allowed by crystfel
    badRegionKeys = ['name', 'min_x', 'max_x', 'min_y', 'max_y', 'Panel']

    # Initialize a template dictionary for bad regions.
    badRegionDictTemplate = {}
    for key in badRegionKeys:
        badRegionDictTemplate[key] = None

    # Initialize a template dictionary for rigid groups
    rigidGroupDictTemplate = {'name': None, 'panels': []}

    # Initialize a template dictionary for rigid group collections
    rigidGroupCollectionDictTemplate = {'name': None, 'groups': []}

    # Initialize the lists we will populate as we scan the file
    globals_ = panelDictTemplate.copy()
    panels = []
    rigidGroups = []
    rigidGroupCollections = []
    badRegions = []

    # Now begin scanning the file

    if geom_file is None:
        raise ValueError("No geometry file specified")

    h = open(geom_file, "r")

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
            globals_[key] = value
            continue

        # Check for rigid group collection
        if re.search("^rigid_group_collection", key):
            name = key.split("_")[-1]
            thisDict = seek_dictionary_by_name(rigidGroupCollections, name)
            if thisDict is None:
                thisDict = rigidGroupCollectionDictTemplate.copy()
                thisDict['name'] = name
                rigidGroupCollections.append(thisDict)
            thisDict['groups'] = ([k.strip() for k in value.split(',')])
            continue

        # Check for rigid group
        if re.search("^rigid_group", key):
            name = key.split("_")[-1]
            thisDict = seek_dictionary_by_name(rigidGroups, name)
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
            thisDict = seek_dictionary_by_name(badRegions, name)
            if thisDict is None:
                thisDict = badRegionDictTemplate.copy()
                thisDict['name'] = name
                badRegions.append(thisDict)
            thisDict[key] = value
            continue

        # If not any of the above types, check if it is a Panel-specific
        # specification
        key = key.split("/")
        if len(key) != 2:
            print('Cannot interpret string %s' % originalline)
            continue

        # Split name from key/value pairs
        name = key[0].strip()
        key = key[1].strip()

        if key in set(panelKeys):
            thisDict = seek_dictionary_by_name(panels, name)
            if thisDict is None:
                thisDict = panelDictTemplate.copy()
                thisDict['name'] = name
                panels.append(thisDict)
            thisDict[key] = value
            continue

        print("Cannot interpret: %s" % originalline)

    # Convert strings to numeric types as appropriate
    if raw_strings is not True:
        for p in panels:
            convert_types(p)
        convert_types(globals_)
        for badRegion in badRegions:
            convert_types(badRegion)

    # Populate the missing values in each Panel with global values
    for p in panels:
        for key in panelKeys:
            if p[key] is None:
                p[key] = globals_[key]

    # Now package up the results and return
    geomDict = {'globals_': globals_,
                'panels': panels,
                'rigidGroups': rigidGroups,
                'rigidGroupCollections': rigidGroupCollections,
                'badRegions': badRegions}

    return geomDict


def geom_dict_to_padgeometry(geomDict):
    r""" Convert a CrystFEL geometry dictionary to a list of PADGeometry() instances. """

    # TODO: Test and document this function

    pads = []

    for p in geomDict['panels']:

        pad = detector.PADGeometry()

        pad.name = p['name']
        pad.fs_vec = np.array(p['fs']) * 1.0 / p['res']
        pad.ss_vec = np.array(p['ss']) * 1.0 / p['res']
        pad.n_fs = p['max_fs'] - p['min_fs'] + 1
        pad.n_ss = p['max_ss'] - p['min_ss'] + 1
        z = 0.0
        if not isinstance(p['coffset'], str):
            if p['coffset'] is not None:
                z += p['coffset']
        pad.t_vec = np.array([p['corner_x'] / p['res'], p['corner_y'] / p['res'], z])
        pad.adu_per_ev = p['adu_per_eV']

        pads.append(pad)

    return pads
