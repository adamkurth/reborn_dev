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

import numpy as np


class FrameGetter(object):

    r"""
    FrameGetter is a generic interface for serving up data frames.  It exists so that we can hide the specific details
    of where data come from (be it files on disk, shared memory, simulations, etc.) and build software that can work
    in a way that is agnostic to the source of the data.

    The FrameGetter serves up dictionaries that contain diffraction data, peak positions, x-ray source info, and so on.
    It is assumed that frames are indexed with with integers, starting with zero.  This indexing scheme will eventually
    be generalized in the future (for example, LCLS indexes frames by a tuple of three integers: seconds, nanoseconds,
    and a fiducial).

    This FrameGetter class is only a base class -- you're not supposed to use it directly.  Instead, you should subclass
    it.  Here is a minimal example of how to subclass it:

    .. code-block:: Python

        class MyFrameGetter(FrameGetter):
            def __init__(self, arguments):
                super().__init__()
                # Possibly set the n_frames attribute during initialization
                self.n_frames = something_based_on_arguments
            def get_frame(self, frame_number):
                # Do something to fetch data with specified index
                return {'pad_data':pad_data} or DataFrame instance

    Minimally, your FrameGetter subclass should set the n_frames attribute that specifies how many frames there are, and
    the get_frame method should be defined.  The FrameGetter base class will then implement other conveniences such as
    get_next_frame(), get_previous_frame(), etc.

    The get_frame() method should return a dictionary with standard keys.  The only required key at this time is
    "pad_data" and it should point to a list of 2D numpy arrays.  We have not yet documented how to specify e.g. peak
    lists in a standard way.
    """

    def __init__(self):

        self._n_frames = 1
        self.current_frame = 0
        self.geom_dict = None
        self.skip = 1
        self.history_length = 10000
        self.history = np.zeros(self.history_length, dtype=np.int)
        self.history_index = 0

    @property
    def n_frames(self):
        return self._n_frames

    @n_frames.setter
    def n_frames(self, n_frames):
        self._n_frames = n_frames

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


class ListFrameGetter(FrameGetter):
    r"""
    Very simple FrameGetter subclass that operates on a list or similar type of iterable object.
    """
    def __init__(self, dataframes):
        super().__init__()
        self.n_frames = len(dataframes)
        self.dataframes = dataframes
    def get_frame(self, frame_number):
        return self.dataframes[frame_number]
