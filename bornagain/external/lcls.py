r"""
Utilities for working with LCLS data.  Most of what you need is already in the psana package.  I don't know where the
official psana documentation is but if you work with LCLS data you should at least skim through all of the material in
the `LCLS Data Analysis Confluence pages <https://confluence.slac.stanford.edu/display/PSDM/LCLS+Data+Analysis>`_.
Note that there is documentation on `LCLS PAD geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import re
import numpy as np
import bornagain
try:
    import psana
except ImportError:
    psana = None


class AreaDetector(object):
    r"""
    Thin wrapper for psana.Detector class.  Adds methods to generate list of PADGeometry instances and to split the PAD
    data into a list of 2d arrays.
    """

    _psf = None
    _type = None
    _splitter = None

    def __init__(self, *args, **kwargs):
        r"""
        Instantiate with same arguments you would use to instantiate a psana.Detector instance.  Usually this means to
        suppply a psana.DataSource instance.
        """
        self.detector = psana.Detector(*args, **kwargs)
        self.detector_type = self.get_detector_type()

    def get_detector_type(self):
        """ The psana detector fails to provide reliable information on detector type. """
        detector_id = self.detector.source.__str__()
        if re.match(r'.*cspad', detector_id, re.IGNORECASE) is not None:
            detector_type = 'cspad'
        elif re.match(r'.*pnccd.*', detector_id, re.IGNORECASE) is not None:
            detector_type = 'pnccd'
        else:
            detector_type = 'unknown'
        return detector_type

    @property
    def type(self):
        """ See get_detector_type method. """
        if self._type is None:
            self._type = self.get_detector_type()
        return self._type

    def get_calib_split(self, event):
        """ Just like the calib data but split into a list of panels """
        calib = self.detector.calib(event)
        if calib is None:
            return None
        return self.split_pad(calib)

    def get_raw_split(self, event):
        """ Just like the raw data but split into a list of panels """
        raw = self.detector.raw(event)
        if raw is None:
            return None
        return self.split_pad(raw)

    def get_pad_geometry(self, run):
        """ See documentation for the function get_pad_geometry(). """
        return get_pad_geometry(self.detector, run)

    def split_pad(self, data):
        """ Split psana data block into a PAD list """
        if self._splitter is not None:
            # Optional custom splitter function e.g. for funky crystfel/cheetah conventions
            return self._splitter(data)
        if self.detector.is_cspad():
            return self.cspad_data_splitter(data)
        pads = []
        for i in range(data.shape[0]):
            pads.append(data[i, :, :])
        return pads

    def cspad_data_splitter(data):
        return cspad_data_splitter(data)


def cspad_data_splitter(data):
    r"""
    Split the stack of 32 asics into 64 asics.  While it is true that there are 32 physical asics, they have two columns
    of pixels that are

    Arguments:
        data (numpy array): An array of PAD data exactly as it is presented by the psana Detector class

    Returns:
        A list of separated PADs.

    Thanks to Derek Mendez for this.
    """
    asics64 = []
    for split_asic in [(asic[:, :194], asic[:, 194:]) for asic in data]:
        for sub_asic in split_asic:  # 185x194 arrays
            asics64.append(sub_asic)
    return asics64


def get_pad_geometry(detector, run):
    r"""
    Create a list of PADGeometry instances from a psana detector and a run instance.  I have no idea of where this data
    actually originates; presumabely someone at LCLS tried to estimate the geometry.  The geometry is exposed by the
    Detector class in the psana package.

    Special considerations are taken for the case of the CSPAD detector, since it has rows of pixels that are elongated
    (did you know that?).  The nominal pixel size is 109.92 x 109.92 microns, but the central two columns (193 and 194)
    have pixels of size 274.80 x 109.92 microns.  This is documented here:
    https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry

    Credit goes to Derek Mendez for this.

    Arguments:
        detector: a psana.Detector object
        run: a psana run object

    Returns:
        A list of bornagain PADGeometry objects
    """

    psf = detector.geometry(run).get_psf()
    geom = []
    n_fs = detector.shape()[2]
    n_ss = detector.shape()[1]
    if detector.is_cspad():
        # We must deal with the fact that the 32 PADs returned by psana are really 64 PADs.
        shift = 194. * 109.92 + (274.8 - 109.92) * 2.
        for i in range(0, 64, 2):
            a = psf[i]
            f = np.array(a[2])
            t = np.array(a[0]) + shift * f / np.sqrt(np.sum(f ** 2))
            b = ((t[0], t[1], t[2]), (a[1][0], a[1][1], a[1][2]), (a[2][0], a[2][1], a[2][2]))
            psf.insert(i + 1, b)
        n_fs = 194
        n_ss = 185
    for i in range(len(psf)):
        g = bornagain.detector.PADGeometry()
        g.t_vec = np.array(psf[i][0]) * 1e6
        g.ss_vec = np.array(psf[i][1]) * 1e6
        g.fs_vec = np.array(psf[i][2]) * 1e6
        g.n_fs = n_fs
        g.n_ss = n_ss
        geom.append(g)
    return geom
