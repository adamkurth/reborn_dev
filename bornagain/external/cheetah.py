r"""
Utilities for working with data created by Cheetah.  I make no promises that any of this will be helpful for you. 
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from bornagain.external import crystfel
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

    cheetah_array = reshape_psana_cspad_array_to_cheetah_array(psana_array)

    return cheetah_remapped_cspad_array_to_pad_list(cheetah_array, geom_dict)


def cheetah_remapped_cspad_array_to_pad_list(cheetah_array, geom_dict):

    utils.depreciate('Dont use cheetah_remapped_cspad_array_to_pad_list() function.  Instead, use'
                     'crystfel.split_image()')

    return crystfel.split_image(cheetah_array, geom_dict)


def reshape_psana_pnccd_array_to_cheetah_array(psana_array):
    r"""
    Transform  a native psana cspad numpy array of shape (32,185,388) into a "Cheetah array" of shape (1480, 1552).
    Conversion to Cheetah format requires a re-write of the data in memory, and each detector panel is no longer stored
    contiguously in memory.

    Arguments:
        psana_array (numpy array) :
            A numpy array of shape (32,185,388) produced by the psana module

    Returns:
        cheetah_array (numpy array) :
            A numpy array of shape (1024, 1024); same data as the psana array but mangled as done within Cheetah
    """

    slab = np.zeros((1024, 1024), dtype=psana_array.dtype)
    slab[0:512, 0:512] = psana_array[0]
    slab[512:1024, 0:512] = psana_array[1][::-1, ::-1]
    slab[512:1024, 512:1024] = psana_array[2][::-1, ::-1]
    slab[0:512, 512:1024] = psana_array[3]

    return slab


