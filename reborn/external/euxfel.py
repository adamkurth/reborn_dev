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
Utilities for working with EuXFEL data. Uses EuXFEL's extra_data package.
Documentation: https://rtd.xfel.eu/docs/data-analysis-user-documentation/en/latest/index.html
               https://extra-data.readthedocs.io/en/latest/index.html
Source Code: https://github.com/European-XFEL/EXtra-data
"""

import numpy as np
import reborn
from reborn import utils
from reborn.source import Beam
import extra_data

debug = True


def debug_message(*args, caller=True, **kwargs):
    r""" Standard debug message, which includes the function called. """
    if debug:
        s = ''
        if caller:
            s = utils.get_caller(1)
        print(f'DEBUG:euxfel.{s}:', *args, **kwargs)


def inspect_available_data(experiment_id, run_id, source=None):
    r"""
    Prints out available data sources for a specific proposal & run.
    If given a data source, it will also print out the keys for that data source.

    Args:
        experiment_id (int): Experiment proposal number.
        run_id (int): Experiment run number.
        source (str): data source (example='SPB_XTD9_XGM/XGM/DOOCS').
    """
    debug_message('opening run')
    run = extra_data.open_run(proposal=experiment_id, run=run_id, data='raw')
    debug_message('gathering sources')
    print(f'Data Sources:\n\n{run.all_sources}\n\n')
    if source is not None:
        source_data = run[source]
        print(f'Data source keys:\n\n{source_data.keys()}\n\n')


class EuXFELFrameGetter(reborn.fileio.getters.FrameGetter):
    r"""
    EuXFELFrameGetter to retrieve detector data from EuXFEL endstations in the standard way.

    EuXFEL saves a series of exposures each corresponding to an individual x-ray pulse together,
    indexed by the pulse_train. This framegetter handles that for you so you can iterate directly through
    frames as if they were globally indexed as in (LCLS or SACLA). The trains are cached so the
    data is not reloaded if the next frame is within the same train.

    Args:
        experiment_id (int): Experiment proposal number.
        run_id (int): Experiment run number.
        pad_detectors (str): pad detector data path in H5 (example='SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', default='*/DET/*').
        geom (|PADGeometryList|): reborn.detector.PADGeometryList instance with the experiment geometry.
        max_events (int): Maximum number of frames to retrieve.
        beam (|Beam|): reborn.source.Beam instance with the x-ray details for the run.
    """
    current_train_stack = None
    current_train_id = None
    beam = None

    def __init__(self, experiment_id, run_id,
                 geom=None, beam=None, pad_detectors='*/DET/*',
                 pad_detector_motor='SPB_IRU_AGIPD1M/MOTOR/Z_STEPPER',
                 xray_wavelength_detector='SA1_XTD2_XGM/XGM/DOOCS',
                 max_events=None):
        debug_message('Initializing superclass')
        super().__init__()
        self.init_params = {'experiment_id': experiment_id,
                            'run_id': run_id,
                            'pad_detectors': pad_detectors,
                            'geom': geom,
                            'max_events': max_events,
                            'beam': beam}
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.pad_detectors = pad_detectors
        debug_message('setting geometry')
        self.pad_geometry = geom
        # extra data first loads a run
        # in the background this is opening an HDF5 file
        # we are loading the processed data (dark calibrated)
        debug_message('gathering run data')
        run = extra_data.open_run(proposal=self.experiment_id, run=self.run_id, data='proc')
        # here we select the type of data we want (pad detector exposures)
        debug_message('run selection')
        self.selection = run.select(self.pad_detectors, 'image.data', require_all=True)
        debug_message('finding sources')
        # data is saved for each individual panel
        # so there N files for a detector with N pads
        # this finds all the files needed to stitch together a single exposure
        sources = run.all_sources
        # build a mapping from the trains to individual exposures
        debug_message('building detector index')
        detectors = [s for s in sources if '/DET/' in s]
        train_shots = dict()
        for d in detectors:
            t_shots = run[d, 'image.data'].data_counts()
            train_shots.update(t_shots.to_dict())
        self.n_frames = np.sum(list(train_shots.values()))  # count the number of frames
        debug_message('building frame index')
        self.frames = list()
        for k, v in train_shots.items():
            f_num = np.arange(v)
            t_ids = np.ones(v) * k
            self.frames.extend(list(zip(t_ids, f_num)))
        # set maximum number of frames to work with (if specified)
        debug_message('enforcing max_events')
        if max_events is not None:
            self.n_frames = min(max_events, self.n_frames)
        # the photon wavelength is easily accessible by opening the raw data
        debug_message('gather photon energy')
        run_raw = extra_data.open_run(proposal=self.experiment_id, run=self.run_id, data='raw')
        self.photon_data = run_raw[xray_wavelength_detector, 'pulseEnergy.wavelengthUsed.value']
        self.pad_detector_motor_position = run_raw[pad_detector_motor, 'actualPosition.value']

    def update_detector_distance(self, offset=0.125, vector=np.array([0, 0, 1e-3])):
        r"""
        Modify the PADGeometryList.

        Arguments:
            offset (float): Motor position to interaction region offset in meters (typically 120-130mm)
            vector (|ndarray|): This is the vector indicating the direction and step size. The stage
                                offset and vector are added and set to the PADGeometry.t_vec
        """
        pads = self.pad_geometry.copy()
        for p in pads:
            p.t_vec[2] = offset
        pads.translate(vector)
        debug_message(f'Shifted detector to {offset + vector}')
        return pads

    def _get_train_stack(self, train_id):
        train_id, train_data = self.selection.train_from_id(train_id)
        stack = extra_data.stack_detector_data(train_data, 'image.data')
        return np.double(stack)

    def get_data(self, frame_number=0):
        debug_message()
        # load the data from the HDF5 file
        debug_message('loading train')
        train_id, fn = self.frames[frame_number]
        # cache current train stack
        if self.current_train_stack is not None:
            if train_id == self.current_train_id:
                stacked = self.current_train_stack
            else:
                stacked = self._get_train_stack(train_id)
                self.current_train_id = train_id
                self.current_train_stack = stacked
        else:
            stacked = self._get_train_stack(train_id)
            self.current_train_id = train_id
            self.current_train_stack = stacked
        stacked_pulse = stacked[fn]
        debug_message('building dataframe')
        df = reborn.dataframe.DataFrame()
        debug_message('setting calibrated pad detector data')
        df.set_dataset_id(f'run:{self.run_id} (Calibrated Data)')
        df.set_frame_id(f'run:{self.run_id}:{frame_number}')
        df.set_frame_index(frame_number)
        debug_message('getting detector stage position')
        _, detector_position = self.pad_detector_motor_position.train_from_id(train_id)  # result is in mm
        vec = np.array([0, 0, 1e-3 * detector_position])  # convert to m (reborn is in SI)
        pads = self.update_detector_distance(offset=0.125, vector=vec)
        debug_message('setting PADGeometry')
        df.set_pad_geometry(pads)
        df.set_raw_data(stacked_pulse)
        debug_message('retrieving x-ray data')
        _, wavelength = self.photon_data.train_from_id(train_id)  # result is in nm
        debug_message('setting Beam')
        self.beam = Beam(wavelength=wavelength * 1e-9)
        df.set_beam(self.beam)
        debug_message('returning', df)
        return df
