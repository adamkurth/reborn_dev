from __future__ import (absolute_import, division, print_function, unicode_literals)

import h5py
import numpy as np
from bornagain.external.crystfel import load_crystfel_geometry, geometry_file_to_pad_geometry_list
from bornagain.external.cheetah import cheetah_remapped_cspad_array_to_pad_list


class FrameGetter(object):

    r"""

    Experimental - a generic interface for serving up data frames.  This should basically just serve up dictionaries
    that have entries for diffraction data, peak positions, x-ray source, and so on.

    This class is just a template - you're not supposed to use it.  You may want to subclass it so that the assumed
    methods are available, even if they just return None by default.

    Minimally, a "frame getter" should have a simple means to provide infomation on how many frames there are, and
    methods for getting next frame, previous frame, and arbitrary frames.

    Once could imagine making things fast by having parallel threads or processes that are pulling data of disk and
    cleaning it up prior to serving it up to a top-level program.  Or, the getter just serves up raw data.

    We could also have getters that serve up simulated data.

    It is expected that these getters need a lot of customization, since we simply cannot avoid the many ways in which
    data is created and stored...

    """

    def __init__(self):

        n_frames = 1
        current_frame = 1
        geom_dict = None

    def get_frame(self, frame_number=None):

        return None

    def get_next_frame(self, skip=1):

        return None

    def get_previous_frame(self, skip=1):

        return None


class CheetahFrameGetter(FrameGetter):

    r"""

    A frame getter that attempts to read the CXIDB variants that are written by Cheetah.

    """

    def __init__(self, cxi_file_name=None, geom_file_name=None):

        FrameGetter.__init__(self)
        self.geom_dict = load_crystfel_geometry(geom_file_name)
        self.pad_geometry = geometry_file_to_pad_geometry_list(geom_file_name)
        self.n_pads = len(self.pad_geometry)
        print(self.n_pads)
        self.h5file = h5py.File(cxi_file_name, 'r')
        self.h5_data = self.h5file['/entry_1/data_1/data']
        self.n_frames = self.h5_data.shape[0]
        self.current_frame = 0

        self.peaks = None

        # for key in list(self.h5file['entry_1/result_1'].keys()):
        #     print(self.h5file['entry_1/result_1/'+key])

    def get_peaks(self, h5file, frame_number):

        n_peaks = h5file['entry_1/result_1/nPeaks'][frame_number]

        if n_peaks <= 0:
            return None

        fs_pos_raw = h5file['entry_1/result_1/peakXPosRaw'][frame_number, 0:n_peaks]
        ss_pos_raw = h5file['entry_1/result_1/peakYPosRaw'][frame_number, 0:n_peaks]

        if self.peaks is None:
            fs_min = np.zeros((self.n_pads))
            fs_max = fs_min.copy()
            ss_min = fs_min.copy()
            ss_max = fs_min.copy()
            for (i, key) in zip(range(0, self.n_pads), list(self.geom_dict['panels'].keys())):
                pan = self.geom_dict['panels'][key]
                fs_min[i] = pan['min_fs']
                fs_max[i] = pan['max_fs']
                ss_min[i] = pan['min_ss']
                ss_max[i] = pan['max_ss']
            ofset = 0.5  # CrystFEL positions in pixel corner, Cheetah positions in pixel center
            self.fs_min = fs_min - ofset
            self.fs_max = fs_max - ofset
            self.ss_min = ss_min - ofset
            self.ss_max = ss_max - ofset

        fs_min = self.fs_min
        fs_max = self.fs_max
        ss_min = self.ss_min
        ss_max = self.ss_max

        pad_numbers = np.zeros(n_peaks)
        fs_pos = pad_numbers.copy()
        ss_pos = pad_numbers.copy()

        for i in range(0, self.n_pads):
            indices = np.argwhere( (fs_pos_raw > fs_min[i]) * (fs_pos_raw <=  fs_max[i]) * \
                                   (ss_pos_raw > ss_min[i]) * (ss_pos_raw <=  ss_max[i]) )
            if len(indices > 0):
                pad_numbers[indices] = i
                fs_pos[indices] = fs_pos_raw[indices] - fs_min[i]
                ss_pos[indices] = ss_pos_raw[indices] - ss_min[i]

        pad_numbers = pad_numbers.astype(np.int)

        peaks = {'pad_numbers': pad_numbers, 'fs_pos': fs_pos, 'ss_pos': ss_pos, 'n_peaks': n_peaks,
                 'peakXPosRaw': fs_pos_raw, 'peakYPosRaw': ss_pos_raw}

        return peaks

    def get_frame(self, frame_number=0):

        dat = np.array(self.h5_data[frame_number, :, :]).astype(np.double)
        pad_data = cheetah_remapped_cspad_array_to_pad_list(dat, self.geom_dict)

        peaks = self.get_peaks(self.h5file, frame_number)

        dat = {'pad_data': pad_data, 'peaks': peaks}

        return dat

    def get_next_frame(self, skip=1):

        self.current_frame = (self.current_frame + skip) % self.n_frames
        dat = self.get_frame(self.current_frame)

        return dat

    def get_previous_frame(self, skip=1):

        self.current_frame = (self.current_frame - skip) % self.n_frames
        dat = self.get_frame(self.current_frame)

        return dat