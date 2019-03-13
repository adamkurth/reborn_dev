r"""
Utilities for working with LCLS data.  Most of what you need is already in the psana package.  I don't know where the
official psana documentation is but if you work with LCLS data you should at least skim through all of the material in
the `LCLS Data Analysis Confluence pages <https://confluence.slac.stanford.edu/display/PSDM/LCLS+Data+Analysis>`_.
Note that there is documentation on `LCLS PAD geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import re
import numpy as np
from bornagain import utils
import bornagain
try:
    import psana
except:
    pass

class AreaDetector(object):

    """
    Thin wrapper for psana.Detector class.  Adds methods to generate list of PADGeometry instances and to split the PAD
    data in to a list of 2d arrays.
    """

    _psf = None
    _type = None
    _splitter = None

    def __init__(self, *args, **kwargs):

        self.detector = psana.Detector(*args, **kwargs)
        self.detector_type = self.get_detector_type()

    def get_detector_type(self):
        detector_id = self.detector.source.__str__()
        print(detector_id)
        print(type(detector_id))
        if re.match(r'.*CsPad', detector_id, re.IGNORECASE) is not None:
            detector_type = 'cspad'
        elif re.match(r'pnccd.*', detector_id, re.IGNORECASE) is not None:
            detector_type = 'pnccd'
        else:
            detector_type = 'unknown'
        return detector_type

    @property
    def type(self):
        if _type is None:
            self._type = self.get_detector_type()
        return self._type

    def get_psf(self, run):
        if self._psf is None:
            return self.detector.geometry(run).get_psf()
        return self._psf

    def get_calib_split(self, event):

        calib = self.detector.calib(event)
        if calib is None:
            return None
        return self.split_pad(calib)

    def get_raw_split(self, event):

        raw = self.detector.raw(event)
        if raw is None:
            return None
        return self.split_pad(raw)

    def get_pad_geometry(self, run):

        psf = self.get_psf(run)
        if self.detector.is_cspad():
            shift = 194. * 109.92 + (274.8 - 109.92) * 2.
            for i in range(0, 32, 2):
                a = psf[i]
                f = np.array(a[2])
                t = np.array(a[0]) + shift * f/np.sqrt(np.sum(f**2))
                b = ((t[0],    t[1],    t[2]   ), (a[1][0], a[1][1], a[1][2]), (a[2][0], a[2][1], a[2][2]))
                psf.insert(i+1, b)
        geom = []
        for i in range(len(psf)):
            g = bornagain.detector.PADGeometry()
            g.t_vec = np.array(psf[i][0])*1e6
            g.ss_vec = np.array(psf[i][1])*1e6
            g.fs_vec = np.array(psf[i][2])*1e6
            geom.append(g)
        return geom

    def split_pad(self, data):

        # Optional custom splitter function e.g. for funky crystfel/cheetah conventions that scamble data
        if self._splitter is not None:
            return self._splitter(data)
        if self.detector.is_cspad():
            return self.cspad_data_splitter(data)
        pads = []
        for i in range(data.shape[0]):
            pads.append(data[i, :, :])
        return pads

    @staticmethod
    def cspad_data_splitter(data):
        """Thanks to Derek for this."""
        asics64 = []
        for split_asic in [(asic[:, :194], asic[:, 194:]) for asic in data]:
            for sub_asic in split_asic:  # 185x194 arrays
                asics64.append(sub_asic)
        return asics64


def reshape_psana_cspad_array_to_cheetah_array(psana_array):

    r"""
    Transform a native psana cspad numpy array of shape (32,185,388) into a "Cheetah array" of shape (1480, 1552).
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

    r"""

    This will make a list of PAD data arrays, provided a data array that is in the form that Cheetah works with.  This
    is needed because people mostly make geom files that work with the Cheetah format.  See the function
    :func:`reshape_psana_cspad_array_to_cheetah_array <bornagain.external.cheetah.reshape_psana_cspad_array_to_cheetah_array>` for more details on array shapes.

    Args:
        cheetah_array: The data in Cheetah format
        geom_dict: The dictionary created from a CrystFEL geom file via :func:``

    Returns:

    """

    utils.depreciate("Function cheetah_remapped_cspad_array_to_pad_list() is now in external.cheetah."
                     "  Don't import it from external.lcls")

    pad_list = []
    for pn in geom_dict['panels']:
        p = geom_dict['panels'][pn]
        pad_list.append(cheetah_array[p['min_ss']:(p['max_ss'] + 1), p['min_fs']:(p['max_fs'] + 1)])

    return pad_list
