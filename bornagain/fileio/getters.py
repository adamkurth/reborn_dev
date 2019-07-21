from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np


class FrameGetter(object):

    r"""

    Experimental - a generic interface for serving up data frames.  This should basically just serve up dictionaries
    that have entries for diffraction data, peak positions, x-ray source, and so on.

    This class is just a template - you're not supposed to use it.  You may want to subclass it so that the assumed
    methods are available, even if they just return None by default.

    Minimally, a "frame getter" should have a simple means to provide infomation on how many frames there are, and
    methods for getting next frame, previous frame, and arbitrary frames.

    Once could imagine making things fast by having parallel threads or processes that are pulling data off disk and
    cleaning it up prior to serving it up to a top-level program.  Or, the getter just serves up raw data.

    We could also have getters that serve up simulated data.

    It is expected that these getters need a lot of customization, since we simply cannot avoid the many ways in which
    data is created and stored...

    """

    def __init__(self):

        self.n_frames = 1
        self.current_frame = 0
        self.geom_dict = None
        self.skip = 1
        self.history_length = 10000
        self.history = np.zeros(self.history_length, dtype=np.int)
        self.history_index = 0

    def get_data(self, frame_number=None):

        pass
        return None

    def get_frame(self, frame_number=None):

        self.log_history()
        return self.get_data(frame_number=frame_number)

    def log_history(self):

        self.history[self.history_index] = self.current_frame
        self.history_index = int(
            (self.history_index + 1) % self.history_length)

    def get_history_previous(self):

        self.history_index = int(
            (self.history_index - 1) % self.history_length)
        self.current_frame = self.history[self.history_index]
        dat = self.get_frame(self.current_frame)

        return dat

    def get_history_next(self):

        self.history_index = int(
            (self.history_index + 1) % self.history_length)
        self.current_frame = self.history[self.history_index]
        dat = self.get_frame(self.current_frame)

        return dat

    def get_next_frame(self, skip=None):

        if skip is None:
            skip = self.skip

        self.log_history()
        self.current_frame = int((self.current_frame + skip) % self.n_frames)
        dat = self.get_frame(self.current_frame)

        return dat

    def get_previous_frame(self, skip=None):

        if skip is None:
            skip = self.skip

        self.log_history()
        self.current_frame = int((self.current_frame - skip) % self.n_frames)
        dat = self.get_frame(self.current_frame)

        return dat

    def get_random_frame(self):

        self.log_history()
        self.current_frame = int(np.ceil(np.random.rand(1)*self.n_frames)) - 1
        dat = self.get_frame(self.current_frame)

        return dat
