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
    import exta_data
except ImportError:
    exta_data = None


debug = True


def debug_message(*args, caller=True, **kwargs):
    r""" Standard debug message, which includes the function called. """
    if debug:
        s = ''
        if caller:
            s = utils.get_caller(1)
        print('DEBUG:lcls.'+s+':', *args, **kwargs)



class EuXFELFrameGetter(reborn.fileio.getters.FrameGetter):
    # SPB_DET_AGIPD1M-1/DET/*CH0:xtdf

    def __init__(self, experiment_id, run_id,
                 pad_detectors='*/DET/*', geom=None, max_events=1e6,
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
        run = exta_data.open_run(proposal=self.experiment_id, run=self.run_id)
        self.selection = run.select(self.pad_detectors, 'image.data', require_all=True)
        self.beam = beam

        sources = run.all_sources
        detectors = list()
        for s in sources:
            if if '/DET/' in s:
                detectors.append(s)
        trains_shot = dict()
        for d in detectors:
            t_shots = run[d, 'image.data'].data_counts()
            train_shots.update(t_shots.to_dict())
        self.n_frames = np.sum(list(train_shots.values()))
        self.frames = dict()
        for i, (k, v) in enumerate(train_shots.items()):
            fnums = np.arange(v)
            vals = np.ones(v) * k
            if i == 1:
                a = 0
            else:
                a = list(self.frames)[-1]
            f = dict(zip(fnum + a, zip(fnums, vals)))
            self.frames.update(f)
        self.photon_energies = run['SA1_XTD2_XGM/XGM/DOOCS', 'pulseEnergy.wavelengthUsed.value']
        self.geom_file = geom

    def get_data(self, frame_number=0):
        debug_message()

        tid, fn = self.frames[frame_number]
        train_id, train_data = self.selection.train_by_id(tid)
        stacked = stack_detector_data(train_data, 'image.data')
        stacked_pulse = stacked[fn][0]
        
        geometry = reborn.detector.PADGeometryList(self.geom_file)
        
        df = reborn.dataframe.DataFrame()
        df.set_dataset_id(f'{self.pad_detectors} run:{self.run_id}')
        df.set_frame_id(frame_number)
        df.set_frame_index(frame_number)
        df.set_pad_geometry(geometry)
        df.set_raw_data(stacked_pulse)
        
        beam = self.beam
        pe = self.photon_energies.train_from_id(tid)
        beam.photon_energy = pe[1] * 1e-9
        df.set_beam(beam)
        debug_message('returning', df)
        return df
