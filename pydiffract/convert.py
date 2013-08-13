'''
Created on Aug 1, 2013

@author: kirian
'''

from pydiffract import detector, source, data
import numpy as np


def crystfel_to_panel_list(filename):

    """ Convert a crystfel "geom" file into a panel list """

    # All panel-specific keys that are recognized
    all_keys = set(["fs", "ss", "corner_x", "corner_y",
                    "min_fs", "max_fs", "min_ss", "max_ss",
                    "clen", "coffset", "res", "adu_per_eV"])

    h = open(filename, "r")

    # Global settings affecting all panels
    global_coffset = None
    global_clen_field = None
    global_clen = None
    global_adu_per_ev = None

    pa = detector.panelList()
    pa.beam = source.beam()

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
            except:
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
            p.name = name
            p.T = np.zeros(3)
            p.F = np.zeros(3)
            p.S = np.zeros(3)
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

    h.close()

    for p in pa:

        # Link array beam to panel beam
        p.beam = pa.beam

        # These are defaults
        p.dataPlan.dataField = "/data/rawdata0"
        p.dataPlan.wavelengthField = "/LCLS/photon_wavelength_A"
        p.B = np.array([0, 0, 1])

        # Unit conversions
        p.T = p.T * p.pixSize

        # Data array size
        p.nF = p.dataPlan.fRange[1] - p.dataPlan.fRange[0] + 1
        p.nS = p.dataPlan.sRange[1] - p.dataPlan.sRange[0] + 1

        # Check for extra global configurations
        if global_adu_per_ev is not None:
            p.aduPerEv = global_adu_per_ev
        if global_clen is not None:
            p.T[2] += global_clen
        if global_clen_field is not None:
            p.dataPlan.detOffsetField = global_clen_field
        if global_coffset is not None:
            p.T[2] += global_coffset

    return pa


def pypad_txt_to_panel_list(filename):

    """ Convert a pypad txt file to a panel list """

    pa = detector.panelList()

    fh = open(filename, "r")
    pn = -1

    for line in fh:
        line = line.split()
        if len(line) == 10:
            try:
                int(line[0])
            except:
                continue

            pn += 1
            pa.append()
            p = pa[pn]

            p.B = np.array([0, 0, 1])
            p.F = np.array([float(line[7]), float(line[8]), float(line[9])])
            p.S = np.array([float(line[4]), float(line[5]), float(line[6])])
            p.T = np.array([float(line[1]), float(line[2]), float(line[3])])

            p.pixSize = np.linalg.norm(p.F)
            p.T *= 1e-3

    fh.close()

    return pa
