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
Note that there is documentation on
`LCLS PAD geometry <https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry>`_.
"""

import re
import numpy as np
import reborn
from .. import utils, detector
from . import crystfel, cheetah
try:
    import psana
except ImportError:
    psana = None


debug = False


def debug_message(*args, caller=True, **kwargs):
    r""" Standard debug message, which includes the function called. """
    if debug:
        s = ''
        if caller:
            s = utils.get_caller(1)
        print('DEBUG:lcls.'+s+':', *args, **kwargs)


def pad_to_asic_data_split(data, n, m):
    r""" 
    Split an array of shape (P, N, M) into a list of n*m*P arrays of shape (N/n, M/m).

    For epix we split into 2x2

    For cspad we split into 1x2 (?)

    Arguments:
        data (np.ndarray): PAD data.
        n (int): Number of ASICS to split "vertically" with vsplit
        m (int): Number of ASICS to split "horizontally" with hsplit

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
    This should work for any detector (except Rayonix, not implemented in psana)
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
        debug_message('Wrong size')
        return None
    if ydc.size != n_panels * n_ss * n_fs:
        debug_message('Wrong size')
        return None
    if zdc.size != n_panels * n_ss * n_fs:
        debug_message('Wrong size')
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
    debug_message()
    xx, yy, zz = get_pad_pixel_coordinates(pad_det, run_number, splitter)

    geom = reborn.detector.PADGeometryList()
    for (x, y, z) in zip(xx, yy, zz):
        g = reborn.detector.PADGeometry()
        g.t_vec = np.array([x[0, 0], y[0, 0], z[0, 0]]) * 1e-6
        g.ss_vec = np.array([x[2, 1] - x[1, 1],
                             y[2, 1] - y[1, 1],
                             z[2, 1] - z[1, 1]]) * 1e-6
        g.fs_vec = np.array([x[1, 2] - x[1, 1],
                             y[1, 2] - y[1, 1],
                             z[1, 2] - z[1, 1]]) * 1e-6
        g.n_ss, g.n_fs = x.shape
        geom.append(g)
    return geom


class EpicsTranslationStageMotion:
    r""" A class that updates PADGeometry according to stages with positions specified by EPICS PVs. """
    def __init__(self, epics_pv, vector=np.array([0, 0, 1e-3])):
        r"""
        Arguments:
            epics_pv ('str'): The EPICS PV string.
            vector (|ndarray|): This is the vector indicating the direction and step size.  The stage position will be
                                multiplied by this vector and added to PADGeometry.t_vec
        """
        self.detector = psana.Detector(epics_pv)
        self.vector = vector
    def modify_geometry(self, pad_geometry, event):
        r""" Modify the PADGeometryList.

        Arguments:
            pad_geometry (|PADGeometryList|): PAD geometry.
            event (psana.Event): A psana event from which the stage position derives.
        """
        position = self.detector(event)
        p = pad_geometry.copy()
        p.translate(self.vector * position)
        return p


class TranslationMotion:
    r""" A class that updates PADGeometry with a shift. """
    def __init__(self, vector=np.array([0, 0, 1e-3])):
        r"""
        Arguments:
            vector (|ndarray|): The translation to apply to the PADGeometry
        """
        self.vector = np.array(vector)
    def modify_geometry(self, pad_geometry, event):
        r""" Modify the PADGeometryList.

        Arguments:
            pad_geometry (|PADGeometryList|): PAD geometry.
            event (psana.Event): A psana event from which the stage position derives.
        """
        p = pad_geometry.copy()
        p.translate(self.vector)
        return p


class AreaDetector(object):
    r"""
    Thin wrapper for psana.Detector class. Adds methods to generate list of PADGeometry instances and to split the PAD
    data into a list of 2d arrays.
    """

    splitter = None
    motions = None  # Allow for some translation operations based on epics pvs
    _home_geometry = None  # This is the initial geometry, before translations/rotations
    _funky_cheetah_cspad = False

    def __init__(self, pad_id=None, geometry=None, mask=None, data_type='calib', motions=None, run_number=1, **kwargs):
        r"""
        Instantiate with same arguments you would use to instantiate a psana.Detector instance.
        Usually this means to supply a psana.DataSource instance.

        Arguments:
            pad_id (str): Example: DscCsPad
            geometry (|PADGeometryList|): Geometry, or a path to geometry file.
            mask (|ndarray|): Mask array, or path to mask file
            data_type (str): Default data type ('calib' or 'raw')
            motions (dict): Special dictionaries to describe motorized motions of the detector

        """
        debug_message('AreaDetector')
        self.detector = psana.Detector(pad_id, **kwargs)
        self.detector_type = self.get_detector_type()
        self.data_type = data_type
        self.motions = motions
        if isinstance(motions, str):
            self.motions = [EpicsTranslationStageMotion(epics_pv=motions)]
        if isinstance(motions, list):
            self.motions = [TranslationMotion(vector=motions)]
        if self.detector_type == 'cspad':
            self.splitter = lambda data: pad_to_asic_data_split(data, 1, 2)
        if self.detector_type == 'epix10k2m':
            self.splitter = lambda data: pad_to_asic_data_split(data, 2, 2)

        if isinstance(geometry, str):
            try:  # Check if it is a reborn geometry file
                debug_message('Check for reborn geometry format')
                geometry = detector.load_pad_geometry_list(geometry)
            except:  # Check if it is a crystfel geometry file
                debug_message('Check for CrystFEL geometry format')
                geometry = crystfel.geometry_file_to_pad_geometry_list(geometry)
                if (self.detector_type == 'cspad') and geometry.parent_data_shape[0] == 1480:
                    geometry = crystfel.fix_cspad_cheetah_indexing(geometry)
                    # self._funky_cheetah_cspad = True
                    # self.splitter = None
                    print('Your CrystFEL geometry assumes the psana data has been re-shuffled by Cheetah!!')
        if geometry is None:
            if self.detector_type == 'rayonix':
                geom = reborn.detector.rayonix_mx340_xfel_pad_geometry_list(detector_distance=1)
                geometry = geom.binned(2)
            else:
                geometry = get_pad_geometry_from_psana(self.detector, run_number, self.splitter)
        self._home_geometry = geometry.copy()
        # print(self._home_geometry)
        self.mask = mask
        if isinstance(mask, str):
            self.mask = detector.load_pad_masks(mask)
        if self.mask is None:
            self.mask = [p.ones() for p in self._home_geometry]

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

    def get_pad_geometry(self, event):
        """ See documentation for the function get_pad_geometry(). """
        geometry = self._home_geometry.copy()
        if self.motions:
            for m in self.motions:
                geometry = m.modify_geometry(geometry, event)
        return geometry

    def split_pad(self, data):
        """ Split psana data block into a PAD list """
        if self.splitter is None:
            return self._home_geometry.split_data(data)
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
            data = [g.zeros() for g in self.get_pad_geometry(event)]
        else:
            data = self.split_pad(data)
        return data

    def get_pixel_coordinates(self, run_number):
        """ Get pixel coordinates from psana. Returns None for rayonix."""
        if self.detector_type == 'rayonix':
            x, y, z = [None, None, None]
        else:
            x, y, z = get_pad_pixel_coordinates(self.detector, run_number, self.splitter)
        return x, y, z

    @property
    def n_pixels(self):
        return self._home_geometry.n_pixels


class LCLSFrameGetter(reborn.fileio.getters.FrameGetter):

    mask = None
    event = None
    event_codes = None

    def __init__(self, experiment_id, run_number, pad_detectors, max_events=1e6, psana_dir=None, beam=None, idx=True):
        debug_message('Initializing superclass')
        super().__init__()  # initialize the superclass
        self.init_params = {"experiment_id": experiment_id,
                            "run_number": run_number,
                            "pad_detectors": pad_detectors,
                            "max_events": max_events,
                            "psana_dir": psana_dir,
                            "beam": beam,
                            "idx": idx}
        self.run_number = run_number
        self.data_string = f'exp={experiment_id}:run={run_number}:smd'
        debug_message('datastring', self.data_string)
        if psana_dir is not None:
            self.data_string += f'dir={psana_dir}'
        self.data_source = psana.DataSource(self.data_string)
        self.event_ids = []
        for nevent, evt in enumerate(self.data_source.events()):
            if nevent >= max_events:
                break
            evtId = evt.get(psana.EventId)
            self.event_ids.append((evtId.time()[0], evtId.time()[1], evtId.fiducials()))
        self.n_frames = len(self.event_ids)
        self.has_indexing = idx
        if self.has_indexing:
            self.data_string = self.data_string.replace(':smd', ':idx')
        debug_message('datastring', self.data_string)
        self.data_source = psana.DataSource(self.data_string)
        self.run = self.data_source.runs().__next__()
        self.events = self.run.events()
        self.previous_frame = 0
        self.ebeam_detector = psana.Detector('EBeam')
        self.evr = psana.Detector('evr0')
        self.detectors = [AreaDetector(**p, run_number=self.run_number, accept_missing=True) for p in pad_detectors]
        self.beam = beam
        if beam is None:
            self.beam = reborn.source.Beam()

    def get_data(self, frame_number=0):
        debug_message()
        # This is annoying: ideally we would use indexed data for which we can skip to any frame... but
        # sometimes the index file is missing (e.g. due to DAQ crash or data migration).  So we accommodate
        # the 'smd' mode in addition to the 'idx' mode:
        event = None
        if self.has_indexing:
            ts = self.event_ids[frame_number]
            event = self.run.event(psana.EventTime(int((ts[0] << 32) | ts[1]), ts[2]))
        else:
            if frame_number == self.previous_frame + 1:
                event = self.events.__next__()
            else:
                if not frame_number == 0:
                    debug_message('Skipping frames in the smd mode will be quite slow...')
                ts = self.event_ids[frame_number]
                self.data_source = psana.DataSource(self.data_string)
                self.run = self.data_source.runs().__next__()
                self.events = self.run.events()
                for i in range(frame_number+1):
                    event = self.events.__next__()
        self.previous_frame = frame_number
        self.event = event
        if event is None:
            debug_message('The event is None!')
            return None
        self.event_codes = self.evr.eventCodes(event)
        xray_on = 40 in self.event_codes   # FIXME: This number might differ from one experiment to the next
        laser_on = 41 in self.event_codes  # FIXME: This number might differ from one experiment to the next
        photon_energy = None
        try:
            photon_energy = self.ebeam_detector.get(event).ebeamPhotonEnergy()*reborn.const.eV
        except AttributeError:
            debug_message(f'Run {self.run_number} frame {frame_number} causes ebeamPhotonEnergy failure, skipping this '
                     f'shot.')
        geometry = reborn.detector.PADGeometryList()
        pad_data = []
        pad_mask = []
        for det in self.detectors:
            geom = det.get_pad_geometry(event)
            data = det.get_data_split(event)
            mask = det.mask
            for p, d, m in zip(geom, data, mask):
                geometry.append(p)
                pad_data.append(d)
                if m is not None:
                    pad_mask.append(m)
        df = reborn.dataframe.DataFrame()
        df.set_dataset_id(self.data_string)
        df.set_frame_id(ts)
        df.set_frame_index(frame_number)
        if photon_energy is not None:
            beam = self.beam
            beam.photon_energy = photon_energy
            df.set_beam(beam)
        if geometry:
            df.set_pad_geometry(geometry)
        if pad_data:
            df.set_raw_data(pad_data)
        if pad_mask:
            df.set_mask(pad_mask)
        parameters = {'xray_on': xray_on, 'laser_on': laser_on}
        df.parameters = parameters
        debug_message('returning', df)
        return df
