import re

import numpy as np

from .. import detector
from .cfelpyutils.crystfel_utils import load_crystfel_geometry



def geometry_file_to_dict(geometry_file):

    """
    Given a CrystFEL geometry file, create a python dictionary object.  This uses the cfelpyutils module - blame
    Valerio if it's broken :)

    Args:
        geometry_file (str): Path to geometry file

    Returns: Dict

    """

    return load_crystfel_geometry(geometry_file)


def geometry_dict_to_pad_geometry_list(geometry_dict):

    """
    Given a CrystFEL geometry dictionary, create a list of PADGeometry objects.  This will also append extra
    crystfel-specific items like fsx, max_fs, etc.

    Args:
        geometry_file (str): Path to geometry file

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

    """
    Given a CrystFEL geometry file, create a list of PADGeometry objects.  This will also append extra
    crystfel-specific items like fsx, max_fs, etc.

    Args:
        geometry_file (str): Path to geometry file

    Returns: List of PADGeometry instances

    """

    geometry_dict = geometry_file_to_dict(geometry_file)
    pad_list = geometry_dict_to_pad_geometry_list(geometry_dict)

    return pad_list


def split_image(data, geom_dict):
    r"""
    Split a 2D image into individual panels (useful for working with Cheetah output).

    Arguments:
        data (numpy array) :
            Image data
        geom_dict (dict) :
            Geometry dictionary

    Returns:
        split_data (list) :
            List of individual PAD panel data
    """
    split_data = []
    for p in geom_dict['panels']:
        split_data.append(data[p['min_ss']:(p['max_ss'] + 1), p['min_fs']:(p['max_fs'] + 1)])

    return split_data
