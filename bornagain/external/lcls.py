r"""
Utilities to convert
"""

import numpy as np


def reshape_psana_cspad_array_to_cheetah_array(psana_array):
    r"""
    Transform psana cspad numpy array of shape (32,185,388) into array to Cheetah array of shape (1480, 1552).
    Unfortunately, the Cheetah format requires a re-write of the data in memory, and one result of this re-write is that
    each detector panel is no longer stored contiguously in memory.  A second result is that newcomers will of course
    be very confused when they first encounter this undocumented aspect of Cheetah.

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
    data in the native psana format.  First you should create a crystfel geometry dictionary.

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
    pad_list = []
    for p in geom_dict['panels']:
        pad_list.append(cheetah_array[p['min_ss']:(p['max_ss'] + 1), p['min_fs']:(p['max_fs'] + 1)])

    return pad_list
