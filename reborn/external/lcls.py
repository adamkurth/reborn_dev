# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

r"""
Utilities for working with LCLS data.  Most of what you need is already in the psana package.  I don't know where the
official psana documentation is but if you work with LCLS data you should at least skim through all of the material in
the `LCLS Data Analysis Confluence pages <https://confluence.slac.stanford.edu/display/PSDM/LCLS+Data+Analysis>`_.
Note that there is documentation on `LCLS PAD geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.
"""

import re
import numpy as np
import reborn
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

    def cspad_data_splitter(self, data):
        return cspad_data_splitter(data)


def cspad_data_splitter(data):
    r"""
    Split the stack of 32 asics into 64 asics.  Unfortunately, this default data layout is awkward.  While it is true
    that there are 32 physical asics, which suggests that the data should form a stack of 32 2D images, these asics have
    two columns of pixels in the middle that are elongated.  This elongation thus requires that the 32 asics cannot
    be treated as regular grids of pixels -- they form irregular grids.  In order to cope with this without re-writing
    nearly all software that deals with pixel-array detectors, we split the stack of 32 asics to form a stack of 64
    asics.

    Arguments:
        data (numpy array): An array of PAD data formatted exactly the same as in the psana Detector class.

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
    Create a list of PADGeometry instances from psana Detector and Run instances.  I have no idea of where this data
    actually originates, but it may be accessed by the psana.Detector class.

    Special considerations are taken for the case of the CSPAD detector, since it has rows of pixels that are elongated
    (did you know that?).  The nominal pixel size is 109.92 x 109.92 microns, but the central two columns (193 and 194)
    have pixels of size 274.80 x 109.92 microns.  This is documented here:
    https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry
    https://confluence.slac.stanford.edu/display/PSDM/CSPAD+Geometry+and+Alignment

    Credit goes to Derek Mendez for this.

    Arguments:
        detector: a psana.Detector object
        run: a psana run object

    Returns:
        A list of reborn PADGeometry objects
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
        g = reborn.detector.PADGeometry()
        g.t_vec = np.array(psf[i][0]) * 1e6
        g.ss_vec = np.array(psf[i][1]) * 1e6
        g.fs_vec = np.array(psf[i][2]) * 1e6
        g.n_fs = n_fs
        g.n_ss = n_ss
        geom.append(g)
    return geom


class LCLSFrameGetterV1(object):

    event = None

    def __init__(self, experiment_id=None, run_number=None, pad_ids=None, indexing=':idx'):

        ds = psana.DataSource('exp=%s:run=%d%s' %(experiment_id, run_number, indexing))
        self.data_source = ds
        self.pad_detectors = [ds.Detector(p) for p in pad_ids]
        self.ebeam_detector = ds.Detector('EBeam')
        self.run = ds.runs().next()
        self.event_ids = self.run.times()
        self.n_events = len(self.event_ids)

        self.set_event(0)

    def set_event(self, event_id=None, event_number=None):
        r"""
        event_id is a time stamp from data_source.run().next().times()
        event_number is an integer, which fetches data_source.run().next().times()[event_number]
        """
        if event_number is not None:
            event_id = self.event_ids[event_number]
        else:
            pass
        self.event_id = event_id
        self.event = self.run.event(event_id)

    def get_event(self, event_id=None, event_number=None):
        r"""
        Return an event.  By default the current event is returned, but a specific event number may be provided.
        Args:
            event_id:
            event_number:

        Returns:
            self.event
        """
        if event_number is not None:
            self.set_event(event_number=event_number)
        if event_id is not None:
            self.set_event(event_id=event_id)
        return self.event

    def set_pad_geometry(self, pad_geometries=None):

        pass











