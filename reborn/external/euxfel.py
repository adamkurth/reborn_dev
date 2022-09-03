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

import numpy as np
import reborn
from reborn import utils
from reborn.const import eV
from reborn.source import Beam
import extra_data

debug = True


def debug_message(*args, caller=True, **kwargs):
    r""" Standard debug message, which includes the function called. """
    if debug:
        s = ''
        if caller:
            s = utils.get_caller(1)
        print('DEBUG:euxfel.'+s+':', *args, **kwargs)


class EuXFELFrameGetter(reborn.fileio.getters.FrameGetter):
    # SPB_DET_AGIPD1M-1/DET/*CH0:xtdf
    current_train_stack = None
    current_train_id = None
    def __init__(self, experiment_id, run_id,
                 pad_detectors='*/DET/*', geom=None, max_events=None,
                 beam=None):
        debug_message('Initializing superclass')
        super().__init__()
        self.init_params = {"experiment_id": experiment_id,
                            "run_id": run_id,
                            "pad_detectors": pad_detectors,
                            "geom": geom,
                            "max_events": max_events,
                            "beam": beam}
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.pad_detectors = pad_detectors
        debug_message('gathering run data')
        run = extra_data.open_run(proposal=self.experiment_id, run=self.run_id, data='proc')
        debug_message('run selection')
        self.selection = run.select(self.pad_detectors, 'image.data', require_all=True)
        debug_message('finding sources')
        self.beam = None

        sources = run.all_sources
        debug_message('building detector index')
        detectors = list()
        for s in sources:
            if '/DET/' in s:
                detectors.append(s)
        train_shots = dict()
        for d in detectors:
            t_shots = run[d, 'image.data'].data_counts()
            train_shots.update(t_shots.to_dict())
        self.n_frames = np.sum(list(train_shots.values()))
        debug_message('building frame index')
        self.frames = list()
        for k, v in train_shots.items():
            fnums = np.arange(v)
            vals = np.ones(v) * k
            self.frames.extend(list(zip(vals, fnums)))
        debug_message('enforcing max_events')
        if max_events is not None:
            self.n_frames = min(max_events, self.n_frames)
        debug_message('gather photon energy')
        run_raw = extra_data.open_run(proposal=self.experiment_id, run=self.run_id, data='raw')
        self.photon_energies = run_raw['SA1_XTD2_XGM/XGM/DOOCS', 'pulseEnergy.wavelengthUsed.value']
        debug_message('setting geometry')
        self.geom = geom

    def _get_train_stack(self, train_id):
        train_id, train_data = self.selection.train_from_id(train_id)
        stack = extra_data.stack_detector_data(train_data, 'image.data')
        return np.double(stack)

    def get_data(self, frame_number=0):
        debug_message()
        stacked = None
        debug_message('loading train')
        train_id, fn = self.frames[frame_number]
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
        df.set_dataset_id(f'{self.pad_detectors} run:{self.run_id}')
        df.set_frame_id(frame_number)
        df.set_frame_index(frame_number)
        df.set_pad_geometry(self.geom)
        df.set_raw_data(stacked_pulse)
        pe = self.photon_energies.train_from_id(train_id)
        self.beam = Beam(wavelength=pe[1] * 1e-9)
        df.set_beam(self.beam)
        debug_message('returning', df)
        return df
