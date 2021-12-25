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
from abc import ABC, abstractmethod, abstractproperty
import numpy as np


class FrameGetter(ABC):

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
            def get_data(self, frame_number):
                # Do something to fetch data with specified index
                return {'pad_data':pad_data} or DataFrame instance

    Minimally, your FrameGetter subclass should set the n_frames attribute that specifies how many frames there are, and
    the get_data method should be defined.  The FrameGetter base class will then implement other conveniences such as
    get_next_frame(), get_previous_frame(), etc.

    The get_data() method should return a dictionary with standard keys.  The only required key at this time is
    "pad_data" and it should point to a list of 2D numpy arrays.  We have not yet documented how to specify e.g. peak
    lists in a standard way.
    """

    _n_frames = 1
    current_frame = 0
    geom_dict = None
    history_length = 10000
    history = np.zeros(history_length, dtype=int)
    history_index = 0

    def __init__(self):
        pass

    @property
    def n_frames(self):
        return self._n_frames

    @n_frames.setter
    def n_frames(self, n_frames):
        self._n_frames = n_frames

    @abstractmethod
    def get_data(self, frame_number=0):
        r"""
        This is the only method you should override when making a subclass.
        """
        pass
        return None

    def get_frame(self, frame_number=0, wrap_around=True, log_history=True):
        r""" Do not override this method. """
        if wrap_around:
            frame_number = frame_number % self.n_frames
        if frame_number >= self.n_frames:
            raise StopIteration
        self.current_frame = frame_number
        if log_history:
            self._log_history()
        return self.get_data(frame_number=frame_number)

    def _log_history(self):
        r""" Do not override this method. """
        self.history[self.history_index] = self.current_frame
        self.history_index = int((self.history_index + 1) % self.history_length)

    def get_history_previous(self):
        r""" Do not override this method. """
        self.history_index = int((self.history_index - 1) % self.history_length)
        return self.get_frame(self.history[self.history_index], log_history=False)

    def get_history_next(self):
        r""" Do not override this method. """
        self.history_index = int((self.history_index + 1) % self.history_length)
        return self.get_frame(self.history[self.history_index], log_history=False)

    def get_next_frame(self, skip=1, wrap_around=True):
        r""" Do not override this method. """
        return self.get_frame(self.current_frame+skip, wrap_around=wrap_around)

    def get_previous_frame(self, skip=1, wrap_around=True):
        r""" Do not override this method. """
        return self.get_frame(self.current_frame-skip, wrap_around=wrap_around)

    def get_random_frame(self):
        r""" Do not override this method. """
        return self.get_frame(int(np.floor(np.random.rand(1)*self.n_frames-1e-10)))

    def view(self):
        r""" Create a PADView instance and start it with this FrameGetter instance. """
        from ..viewers.qtviews.padviews import PADView
        pv = PADView(frame_getter=self)
        pv.start()

    def __iter__(self):
        r""" Do not override this method. """
        return _FGIterator(self)


class _FGIterator:
    def __init__(self, fg):
        fg.current_frame = 0
        self.fg = fg
    def __next__(self):
        return self.fg.get_next_frame(wrap_around=False)


class ListFrameGetter(FrameGetter):
    r"""
    Very simple FrameGetter subclass that operates on a list or similar type of iterable object.
    """
    def __init__(self, dataframes):
        super().__init__()
        self.n_frames = len(dataframes)
        self.dataframes = dataframes
    def get_data(self, frame_number=0):
        return self.dataframes[frame_number]
