r"""
Utilities for working with CrystFEL files.  Note that you can convert a CrystFEL geometry file to a Python
dictionary object with the cfelpyutils.crystfel_utils.load_crystfel_geometry() function.  Most of the functions below
wrap around the cfelpyutils package.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import numpy as np

from .. import detector
from .. import units
from cfelpyutils import crystfel_utils

#------------------------------------------------
# Stream file delimiters
sta_geom    = "----- Begin geometry file -----"
sta_chunk   = "----- Begin chunk -----"
sta_crystal = "--- Begin crystal"

end_geom    = "----- End geometry file -----"
end_chunk   = "----- End chunk -----"
end_crystal = "--- End crystal"
#------------------------------------------------

def load_crystfel_geometry(geometry_file):

    r"""
    Given a CrystFEL geometry file, create a python dictionary object.  This uses the cfelpyutils module - blame
    Valerio Mariani if it's broken :)

    Arguments:
        geometry_file (str): Path to geometry file

    Returns:
        Dict
    """

    return crystfel_utils.load_crystfel_geometry(geometry_file)


def geometry_dict_to_pad_geometry_list(geometry_dict):

    r"""
    Given a CrystFEL geometry dictionary, create a list of `:class:<bornagain.geometry.PADGeometry` objects.
    This will also append the name of the panel to the PADGeometry instance.

    Arguments:
        geometry_dict (str): Path to geometry file

    Returns: List of PADGeometry instances

    """

    geom = geometry_dict

    pads = []
    for panel_name in geometry_dict['panels'].keys():
        pad = detector.PADGeometry()
        pad.name = panel_name
        p = geom['panels'][panel_name]
        pix = 1.0 / p['res']
        pad.fs_vec = np.array([p['fsx'], p['fsy'], p['fsz']]) * pix
        pad.n_fs = p['max_fs'] - p['min_fs'] + 1
        pad.ss_vec = np.array([p['ssx'], p['ssy'], p['ssz']]) * pix
        pad.n_ss = p['max_ss'] - p['min_ss'] + 1
        pad.t_vec = np.array([p['cnx'] * pix, p['cny'] * pix, p['clen']])
        pads.append(pad)

    return pads


def geometry_file_to_pad_geometry_list(geometry_file):

    r"""
    Given a CrystFEL geometry file, create a list of `:class:<bornagain.geometry.PADGeometry` objects.  This will also
    append extra crystfel-specific items like fsx, max_fs, etc.

    Arguments:
        geometry_file (str): Path to geometry file

    Returns:
        List of PADGeometry instances
    """

    geometry_dict = load_crystfel_geometry(geometry_file)
    pad_list = geometry_dict_to_pad_geometry_list(geometry_dict)

    return pad_list


def split_image(data, geom_dict):

    r"""
    Split a 2D image into individual panels (useful for working with Cheetah output).

    Arguments:
        data (numpy array) : Contiguous block of image data
        geom_dict (dict) : Geometry dictionary

    Returns:
        split_data (list) :
            List of individual PAD panel data
    """

    split_data = []
    for panel_name in geom_dict['panels']:
        p = geom_dict['panels'][panel_name]
        split_data.append(data[p['min_ss']:(p['max_ss'] + 1), p['min_fs']:(p['max_fs'] + 1)])

    return split_data


def write_geom_file_single_pad(file_path=None, beam=None, pad_geometry=None):

    pad = pad_geometry
    geom_file = os.path.join(file_path)
    fid = open(geom_file, 'w')
    fid.write("photon_energy = %g\n" % (beam.photon_energy * units.eV))
    fid.write("clen = %g\n" % pad.t_vec.flat[2])
    fid.write("res = %g\n" % (1 / pad.pixel_size()))
    fid.write("adu_per_eV = %g\n" % (1.0 / (beam.photon_energy * units.eV)))
    fid.write("0/min_ss = 0\n")
    fid.write("0/max_ss = %d\n" % (pad.n_ss - 1))
    fid.write("0/min_fs = 0\n")
    fid.write("0/max_fs = %d\n" % (pad.n_fs - 1))
    fid.write("0/corner_x = %g\n" % ((pad.t_vec.flat[0] - pad.pixel_size())/2.,))
    fid.write("0/corner_y = %g\n" % ((pad.t_vec.flat[1] - pad.pixel_size())/2,))
    fid.write("0/fs = x\n")
    fid.write("0/ss = y\n")
    fid.close()



def readStreamfile_get_total_number_of_frames(streamfile_name):

    r"""
    Get the total number of frames from a stream file.
    """

    # Load the streamfile.
    f = open(streamfile_name, 'r') 

    count = 0
    for line in f:
        if sta_chunk in line:
            count += 1

    # close the file
    f.close()

    return count


def readStreamfile_get_nth_frame(streamfile_name, n):

    r"""
    Get the A matrix, CXI file path and CXI frame number from the nth frame in a stream file.
    """

    # Load the streamfile
    f = open(streamfile_name, 'r')

    count = 0
    for line in f:
        if sta_chunk in line:
            count += 1

        if count == n:
            break


    # Initialising 
    A = np.zeros((3,3))
    cxiFilepath = 0
    cxiFileFrameNumber = 0

    astar_exist = False

    for line in f:
        if "Image filename:" in line:
            cxiFilepath = line[16:-1]

        if "Event:" in line:
            cxiFileFrameNumber = int(line[9:])

        if "astar = " in line:
            A[0,0] = float(line[8:19])
            A[0,1] = float(line[19:30])
            A[0,2] = float(line[29:40])
            astar_exist = True

        if "bstar = " in line:
            A[1,0] = float(line[8:19])
            A[1,1] = float(line[19:30])
            A[1,2] = float(line[29:40])

        if "cstar = " in line:
            A[2,0] = float(line[8:19])
            A[2,1] = float(line[19:30])
            A[2,2] = float(line[29:40])

        if end_chunk in line:
            break

    # If frame does not contain a star, etc. set the A star matrix to None
    if astar_exist == False:
        A = None

    # close the file
    f.close()

    return A, cxiFilepath, cxiFileFrameNumber
