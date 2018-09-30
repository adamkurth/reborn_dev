r"""
Utilities for working with LCLS data.  Most of what you need is already in the psana package.  I don't know where the
official psana documentation is but if you work with LCLS data you should at least skim through all of the material in
the `LCLS Data Analysis Confluence pages <https://confluence.slac.stanford.edu/display/PSDM/LCLS+Data+Analysis>`_.
Note that there is documentation on `LCLS PAD geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import numpy as np
import bornagain
from bornagain import utils

def reshape_psana_cspad_array_to_cheetah_array(psana_array):
    r"""
    Transform  a native psana cspad numpy array of shape (32,185,388) into a "Cheetah array" of shape (1480, 1552).
    Conversion to Cheetah format requires a re-write of the data in memory, and each detector panel is no longer stored
    contiguously in memory.

    Arguments:
        psana_array (numpy array) :
            A numpy array of shape (32,185,388) produced by the psana module

    Returns:
        cheetah_array (numpy array) :
            A numpy array of shape (1480, 1552); same data as the psana array but mangled as done within Cheetah
    """

    utils.depreciate("Function reshape_psana_cspad_array_to_cheetah_array() is now "
                     "in external.cheetah.  Don't import it from external.lcls")

    imlist = []
    for i in range(0, 4):
        slist = []
        for j in range(0, 8):
            slist.append(psana_array[j + i * 8, :, :])
        imlist.append(np.concatenate(slist))
    cheetah_array = np.concatenate(imlist, axis=1)

    return cheetah_array


def cheetah_cspad_array_to_pad_list(psana_array, geom_dict):
    r"""
    This function is helpful if you have a CrystFEL geom file that refers to Cheetah output, but you wish to work with
    data in the native psana format.  First you should create a crystfel geometry dictionary using the function
    :func:`geometry_file_to_pad_geometry_list() <bornagain.external.crystfel.geometry_file_to_pad_geometry_list>`.

    Arguments:
        psana_array (numpy array) :
            A numpy array of shape (32,185,388) produced by the psana module.
        geom_dict (dict) :
            A CrystFEL geometry dictionary produced by external.crystfel.geom_to_dict() .

    Returns:
        pad_list (list) :
            A list containing data from each pixel array
    """
    
    utils.depreciate("Function cheetah_cspad_array_to_pad_list() is now in external.cheetah."
                     "  Don't import it from external.lcls")

    cheetah_array = reshape_psana_cspad_array_to_cheetah_array(psana_array)
    
    return cheetah_remapped_cspad_array_to_pad_list(cheetah_array, geom_dict)


def cheetah_remapped_cspad_array_to_pad_list(cheetah_array, geom_dict):


    utils.depreciate("Function cheetah_remapped_cspad_array_to_pad_list() is now in external.cheetah."
                     "  Don't import it from external.lcls")

    pad_list = []
    for pn in geom_dict['panels']:
        p = geom_dict['panels'][pn]
        pad_list.append(cheetah_array[p['min_ss']:(p['max_ss'] + 1), p['min_fs']:(p['max_fs'] + 1)])

    return pad_list
