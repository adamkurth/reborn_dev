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
from scipy import constants


def pad_to_asic_data_split(data, n, m):
    r""" 
    Split an array of shape (P, N, M) into a list of n*m*P arrays of shape (N/n, M/m).

    For epix we split into 2x2

    For cspad we split into 1x2 (?)

    Arguments:
        data (np.ndarray): An array of PAD data.

    Returns:
        pads (list): List of separated PADs.
    """
    pads = []
    for a in data:
        b = np.vsplit(a, n)
        for c in b:
            d = np.hsplit(c, m)
            for e in d:
                pads.append(e)
    return pads


def get_pad_pixel_coordinates(pad_det, run_number, splitter):
    r"""
    This should works for any detector (except Rayonix, not implemented in psana)
    without modification so long as splitter is set up correctly.

    Parameters:
        pad_det (obj): psana detector object
        run_number (int): run number of interest
        splitter (func): function to split the arrays
                         returned by pad_det into
                         individual asics

    Returns:
        x (list)
        y (list)
        z (list)
    """
    n_panels, n_ss, n_fs = pad_det.shape()
    xdc, ydc, zdc = pad_det.coords_xyz(run_number)

    if xdc.size != n_panels * n_ss * n_fs:
        return None
    if ydc.size != n_panels * n_ss * n_fs:
        return None
    if zdc.size != n_panels * n_ss * n_fs:
        return None

    x = splitter(xdc)
    y = splitter(ydc)
    z = splitter(zdc)
    return x, y, z


def get_pad_geometry_from_psana(pad_det, run_number, splitter):
    r"""
    Creates PADGeometryList from psana detector object.

    This should work for any detector without modification
    so long as splitter is set up correctly.

    Parameters:
        pad_det (obj): psana detector object
        run_number (int): run number of interest
        splitter (func): function to split the arrays
                         returned by pad_det into
                         individual asics

    Returns:
        PADGeometryList (list of reborn PADGeometry objects)
    
    Notes:
        CSPAD: Has rows of pixels that are elongated
               The nominal pixel size is 109.92 x 109.92 microns, but the central two columns (193 and 194)
               have pixels of size 274.80 x 109.92 microns. This is documented here:
               https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry
               https://confluence.slac.stanford.edu/display/PSDM/CSPAD+Geometry+and+Alignment
    """
    xx, yy, zz = get_pad_pixel_coordinates(pad_det, run_number, splitter)

    geom = reborn.detector.PADGeometryList()
    for (x, y, z) in zip(xx, yy, zz):
        g = reborn.detector.PADGeometry()
        g.t_vec  = np.array([x[0, 0], y[0, 0], z[0, 0]]) * 1e-6
        g.ss_vec = np.array([x[2, 1] - x[1, 1],
                             y[2, 1] - y[1, 1],
                             z[2, 1] - z[1, 1]]) * 1e-6
        g.fs_vec = np.array([x[1, 2] - x[1, 1],
                             y[1, 2] - y[1, 1],
                             z[1, 2] - z[1, 1]]) * 1e-6
        g.n_ss, g.n_fs = x.shape
        geom.append(g)
    return geom


class AreaDetector(object):
    r"""
    Thin wrapper for psana.Detector class. Adds methods to generate list of PADGeometry instances and to split the PAD
    data into a list of 2d arrays.
    """

    _type = None
    _splitter = None

    def __init__(self, detector_info,
                 run_number=1,
                 **kwargs):
        r"""
        Instantiate with same arguments you would use to instantiate a psana.Detector instance.
        Usually this means to suppply a psana.DataSource instance.
        """
        _detectors = ['cspad', 'pnccd', 'epix10k2m', 'rayonix', 'unknown']
        _det_split = [lambda data: pad_to_asic_data_split(data, 1, 2),
                      None,
                      lambda data: pad_to_asic_data_split(data, 2, 2),
                      None,
                      None]
        _nominal   = [[110e-6, 110e-6],  # [ss, fs]
                      None,
                      [100e-6, 100e-6],
                      [44e-6, 44e-6],  # in unbinned mode (typically 2x2 or 4x4)
                      None]
        _longs     = [[109.92e-6, 274.8e-6],  # [ss, fs]
                      None,
                      [100e-6, 100e-6],
                      None,
                      None]
        _bigs      = [None,  # [ss, fs]
                      None,
                      [100e-6, 100e-6],
                      None,
                      None]
        _dist_pvs  = [None,
                      None,
                      ['MFX:ROB:CONT:POS:X',
                       'MFX:ROB:CONT:POS:Y',
                       'MFX:ROB:CONT:POS:Z'],
                      ['MFX:DET:MMS:01.RBV',  # detector_x
                       'MFX:DET:MMS:02.RBV',  # detector_y1
                       'MFX:DET:MMS:04.RBV'],  # detector_z
                      # 'MFX:DET:MMS:03.RBV']  # detector_y2
                      None]
        _rots_pvs  = [None,
                      None,
                      ['MFX:ROB:CONT:POS:RX',
                       'MFX:ROB:CONT:POS:RY',
                       'MFX:ROB:CONT:POS:RZ'],
                      None,
                      None]
        _splits = {d: s for d, s in zip(_detectors, _det_split)}
        # nominal pixel length along slow-scan and fast-scan
        _pixels = {d: p for d, p in zip(_detectors, _nominal)}
        # long pixel length along slow-scan and fast-scan
        _longpix = {d: p for d, p in zip(_detectors, _longs)}
        # big pixel length along slow-scan and fast-scan (currently only epix10k)
        _big_pix = {d: p for d, p in zip(_detectors, _bigs)}
        _stages = {d: dst for d, dst in zip(_detectors, _dist_pvs)}
        _rotations = {d: r for d, r in zip(_detectors, _rots_pvs)}

        if type(detector_info) == str:
            detector_info = {'pad_id': detector_info}

        self.detector = psana.Detector(detector_info['pad_id'], **kwargs)
        self.detector_type = self.get_detector_type()
        self.splitter = _splits[self.detector_type]
        self.pixel_shape = _pixels[self.detector_type]
        self.long_pixel_shape = _longpix[self.detector_type]
        self.big_pixel_shape = _big_pix[self.detector_type]

        # setup distance detector
        if 'stage' in detector_info:
            self.distance = [psana.Detector(cord, **kwargs) for cord in detector_info['stage']]
        else:
            distance_pv = _stages[self.detector_type]
            if distance_pv is not None:
                self.distance = [psana.Detector(cord) for cord in distance_pv]
            else:
                self.distance = None
        # setup detector rotations detector
        if 'rotations' in detector_info:
            self.rotation = [psana.Detector(rot, **kwargs) for rot in detector_info['rotations']]
        else:
            rotation_pv = _rotations[self.detector_type]
            if rotation_pv is not None:
                self.rotation = [psana.Detector(rot) for rot in rotation_pv]
            else:
                self.rotation = None
        # setup detector geometry
        if 'geometry' in detector_info:
            self.geometry = detector_info['geometry']
        else:
            self.geometry = self.get_pad_geometry(run_number)

        # setup detector geometry mask
        if 'mask' in detector_info:
            self.mask = detector_info['mask']
        else:
            self.mask = None

        if 'data_type' in detector_info:
            self.data_type = detector_info['data_type']
        else:
            self.data_type = 'calib'

    def get_detector_type(self):
        """ The psana detector fails to provide reliable information on detector type. """
        detector_id = self.detector.source.__str__()
        if re.match(r'.*cspad', detector_id, re.IGNORECASE) is not None:
            detector_type = 'cspad'
        elif re.match(r'.*pnccd.*', detector_id, re.IGNORECASE) is not None:
            detector_type = 'pnccd'
        elif re.match(r'.*epix10ka2m.*', detector_id, re.IGNORECASE) is not None:
            detector_type = 'epix10k2m'
        elif re.match(r'.*rayonix.*', detector_id, re.IGNORECASE) is not None:
            detector_type = 'rayonix'
        else:
            detector_type = 'unknown'
        return detector_type

    def get_pad_geometry(self, run, binning=2):
        """ See documentation for the function get_pad_geometry(). """
        if self.detector_type == 'rayonix':
            geom = reborn.detector.rayonix_mx340_xfel_pad_geometry_list(detector_distance=1)
            geometry = geom.binned(binning)
        elif self.detector_type == 'unknown':
            geometry = reborn.detector.PADGeometryList()
        else:
            geometry = get_pad_geometry_from_psana(self.detector, run, self.splitter)
        return geometry

    def split_pad(self, data):
        """ Split psana data block into a PAD list """
        if self.splitter is None:
            return [data]
        else:
            return self.splitter(data)

    def get_data_split(self, event):
        """
        Just like the calib data but split into a list of panels.
        Just like the raw data but split into a list of panels.
        """
        if self.data_type == 'calib':
            data = np.double(self.detector.calib(event))
        elif self.data_type == 'raw':
            data = np.double(self.detector.raw(event))
        else:
            data = None
        if not isinstance(data, np.ndarray):
            data = [g.zeros() for g in self.geometry]
        else:
            data = self.split_pad(data)
        return data

    def get_pixel_coordinates(self, run_number):
        """ Get pixel coordinates from psana. Returns None for rayonix."""
        if self.detector_type == 'rayonix':
            x, y, z = [None, None, None]
        else:
            x, y, z = get_pad_pixel_coordinates(self.detector,
                                                run_number,
                                                self.splitter)
        return x, y, z

    def get_detector_coordinates(self, event):
        return [det(event) * 1e-3 for det in self.distance]

    def get_detector_rotations(self, event):
        return [det(event) for det in self.rotation]

    def update_detector_distance(self, event=None, distance=None):
        if distance is None:
            x, y, z = self.get_detector_coordinates(event)
        else:
            z = distance
        for pad in self.geometry:
            pad.t_vec[2] = z

    def update_detector_rotation(self, event=None, rotations=None):
        if rotations is None:
            rxa, rya, rza = self.get_detector_coordinates(event)
        else:
            rxa, rya, rza = rotations
        rx = np.array([[1, 0, 0],
                      [0, np.cos(rxa), -np.sin(rxa)],
                      [0, np.sin(rxa), np.cos(rxa)]])
        ry = np.array([[np.cos(rya), 0, np.sin(rya)],
                      [0, 1, 0],
                      [-np.sin(rya), 0, np.cos(rya)]])
        rz = np.array([[np.cos(rza), -np.sin(rza), 0],
                      [np.sin(rza), np.cos(rza), 0],
                      [0, 0, 1]])
        for pad in self.geometry:
            pad.t_vec = np.dot(np.dot(np.dot(pad.t_vec, rx.T), ry.T), rz.T)
            pad.ss_vec = np.dot(np.dot(np.dot(pad.ss_vec, rx.T), ry.T), rz.T)
            pad.fs_vec = np.dot(np.dot(np.dot(pad.fs_vec, rx.T), ry.T), rz.T)


class LCLSFrameGetter(reborn.fileio.getters.FrameGetter):

    def __init__(self,
                 experiment_id,
                 run_number,
                 pad_detectors):

        super().__init__()  # initialize the superclass
        self.run_number = run_number

        # setup data source
        self.data_string = f'exp={experiment_id}:run={run_number}:smd'
        ds = psana.DataSource(self.data_string)
        self.data_source = ds

        # LCLS uses 3 numbers to define an event.
        # In LCLS2 this will be one number.
        self.event_timestamp = {'seconds': [],
                                'nanoseconds': [],
                                'fiducials': []}

        # Get the times of events (these could come from a saved "small data" file, for example)
        for nevent, evt in enumerate(self.data_source.events()):
            evtId = evt.get(psana.EventId)
            self.event_timestamp['seconds'].append(evtId.time()[0])
            self.event_timestamp['nanoseconds'].append(evtId.time()[1])
            self.event_timestamp['fiducials'].append(evtId.fiducials())
        self.n_frames = nevent + 1  # why is this + 1?

        # now that we have the times, jump to the events in reverse order
        self.data_source = psana.DataSource(f'exp={experiment_id}:run={run_number}:idx')
        self.run = self.data_source.runs().__next__()
        self.event_ids = self.run.times()
        self.n_events = len(self.event_ids)

        # setup detectors
        self.ebeam_detector = psana.Detector('EBeam')
        self.detectors = [AreaDetector(p, run_number=self.run_number, accept_missing=True) for p in pad_detectors]

        # setup geometries
        self.geometry = reborn.detector.PADGeometryList()
        for det in self.detectors:
            self.geometry.add_group(pads=det.geometry, group_name=det.detector_type)

        # unit conversions
        self.eV_to_J = constants.value('electron volt')

    def get_event(self, frame_number=0):
        r"""
        Return the current event.

        Arguements:
            frame_number:

        Returns:
            event
        """
        # get event
        ts = (self.event_timestamp['seconds'][frame_number],
              self.event_timestamp['nanoseconds'][frame_number],
              self.event_timestamp['fiducials'][frame_number])
        et = psana.EventTime(int((ts[0] << 32) | ts[1]), ts[2])
        return self.run.event(et)

    def get_photon_energy(self, event):
        # get photon energy
        eb = self.ebeam_detector.get(event)
        try:
            photon_energy = eb.ebeamPhotonEnergy() * self.eV_to_J
        except AttributeError:
            print(f'Run {self.run_number} frame {frame_number} causes \
                    ebeamPhotonEnergy failure, skipping this shot.')
            photon_energy = None
        return photon_energy

    def get_data(self, frame_number=0):
        event = self.get_event(frame_number=frame_number)

        photon_energy = self.get_photon_energy(event)
        beam = reborn.source.Beam(photon_energy=photon_energy)

        if frame_number == 0:
            for d in self.detectors:
                d.update_detector_distance()

        # get pad detector data
        pad_data = [data for det in self.detectors for data in det.get_data_split(event)]

        df = reborn.dataframe.DataFrame()
        df.set_beam(beam)
        df.set_pad_geometry(self.geometry)
        df.set_raw_data(pad_data)
        return df
