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
import time, threading
from abc import ABC, abstractmethod, abstractproperty
import numpy as np


class FrameGetter(ABC):

    r"""
    |FrameGetter| is a generic interface for serving up |DataFrame| instances.  It exists so that we have a standard
    interface that hides the details of where data come from (disk, shared memory, on-the-fly simulations, etc.).  With
    a |FrameGetter| we can build software that can work in a way that is agnostic to the data source.  This of course
    comes at a cost -- the reborn |FrameGetter| only deals with basic coherent diffraction data.

    The FrameGetter serves up |DataFrame| instances that contain diffraction raw data, x-ray |Beam| info, |PADGeometry|
    info, etc.  It is assumed that a set of frames can be indexed with integers, starting with zero.

    This FrameGetter class is only an Abstract Base Class (ABC).  You cannot use it directly.  Instead, you must define
    a subclass.  Here is what a very simple subclass should look like:

    .. code-block:: Python

        class MyFrameGetter(FrameGetter):
            def __init__(self, arguments):
                super().__init__()
                # Do whatever is needed to set up the data source.
                # Be sure to set the n_frames attribute:
                self.n_frames = something_based_on_arguments
            def get_data(self, frame_number):
                # Do something to fetch data and create a proper DataFrame instance
                return mydataframe

    Minimally, your |FrameGetter| subclass should set the n_frames attribute that specifies how many frames there
    are, and the get_data method should be defined such that it returns a properly constructed |DataFrame| instance.
    The |FrameGetter| base class will then implement other conveniences such as get_next_frame(), get_previous_frame(),
    etc.  Eventually we hope to implement pre-fetching of dataframes to speed things up (optionally).

    Some advanced notes:

    COPY: It is sometimes useful to copy a |FrameGetter|.  Please understand that this is not always an easy thing to
    implement because a |FrameGetter| might have pointers to file objects that should not be copied, as there may be
    serious issues if multiple threads or processes are using that same pointer.  If you want your subclass to allow
    copies, then you need to store all of the initialization parameters in the init_params dictionary.

    PARALLEL PROCESSING: Parallel processing is not problematic so long as you create a new |FrameGetter| instance
    within each process.  However, if you wish to allow one process to spawn multiple processes that each operates on an
    existing |FrameGetter| instance, then you need to be mindful of the fact that it is rarely possible to pass a
    |FrameGetter| from one process to another without creating a disaster.  To get around this, we currently use
    the following strategy: we pass the init_params dictionary mentioned above (needed for creating a copy) along
    with your |FrameGetter| sub-class type.  We create a dictionary like so:

    .. code-block:: Python

        fgd = {"framegetter": YourSubclass, "kwargs": your_init_params}

    Next, you can create a new instance of the |FrameGetter| like so:

    .. code-block:: Python

        fg = fgd["framegetter"](**fgd["kwargs"])

    """

    _n_frames = 1
    current_frame = 0
    current_dataframe = None
    geom_dict = None
    history_length = 10000
    history = np.zeros(history_length, dtype=int)
    history_index = 0
    init_params = None
    skip_empty_frames = True
    _pandas_dataframe = None
    _getter_thread = None
    _cached_data = None
    _cache_forward = False
    _debug = 0

    def __init__(self):
        pass

    def __copy__(self):
        if self.init_params is not None:
            return type(self)(**self.init_params)
        raise ValueError('Cannot copy because init_params is not defined for this FrameGetter subclass.')

    @property
    def n_frames(self):
        return self._n_frames

    @n_frames.setter
    def n_frames(self, n_frames):
        try:
            n_frames = int(n_frames)
        except OverflowError:  # Could be infinity, which is ok
            pass
        self._n_frames = n_frames

    def debug(self, *args, **kwargs):
        if self._debug:
            print('DEBUG:FrameGetterABC:', *args, **kwargs)

    @abstractmethod
    def get_data(self, frame_number=0):
        r"""
        This is the only method you should override when making a subclass.
        """
        pass
        return None

    def _get_data_cache_forward(self, frame_number=0):
        r"""
        Experimental.  This will return the requested frame from cache if it is available, and launch a thread to
        fetch the next frame.

        Args:
            frame_number:

        Returns:

        """
        # Initialize the data to return:
        dat = None
        # If a thread has been launched already, make sure it is done processing:
        if self._getter_thread is not None:
            self.debug('joining thread')
            self._getter_thread.join()
        # If data has been cached by the thread, check if it is the data we want:
        if self._cached_data is not None:
            if self._cached_data['frame_number'] == frame_number:
                self.debug('found cached data for frame', frame_number)
                dat = self._cached_data['data']
        # If there is no cached data (or the wrong data), then we must fetch it now (and wait for it...):
        if dat is None:
            self.debug('fetching data directly')
            dat = self.get_data(frame_number)
        # Launch a thread to fetch the next frame:
        next_frame = frame_number + 1
        if next_frame < self.n_frames:  # If we expect another frame...
            self.debug('launching thread for frame', next_frame)
            self._getter_thread = threading.Thread(target=self._threaded_get_data, args=(next_frame,))
            self._getter_thread.start()
        else:  # If there are no more frames to fetch...
            self._getter_thread = None
            self._cached_data = None
        return dat

    def _threaded_get_data(self, frame_number=0):
        r""" Gets data, but returns a dictionary that also has frame number."""
        self._cached_data = {'frame_number': frame_number, 'data': self.get_data(frame_number=frame_number)}
        return None

    @property
    def pandas_dataframe(self):
        if self._pandas_dataframe is None:
            import pandas
            self._pandas_dataframe = pandas.DataFrame({'Frame #': np.arange(self.n_frames)})
        return self._pandas_dataframe

    @pandas_dataframe.setter
    def pandas_dataframe(self, df):
        self._pandas_dataframe = df

    def get_frame(self, frame_number=0, wrap_around=True, log_history=True):
        r""" Do not override this method. """
        dat = None
        tic = time.time()
        if wrap_around:
            frame_number = frame_number % self.n_frames
        if frame_number >= self.n_frames:
            raise StopIteration
        if self.current_frame == frame_number and self.current_dataframe is not None:
            dat = self.current_dataframe
        else:
            self.current_frame = frame_number
            if log_history:
                self._log_history()
            if self._cache_forward:
                dat = self._get_data_cache_forward(frame_number=frame_number)
            else:
                dat = self.get_data(frame_number=frame_number)
        self.current_dataframe = dat
        self.debug(time.time() - tic, 'seconds to load frame', frame_number)
        return dat

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
        df = None
        for _ in range(self.n_frames):
            df = self.get_frame(self.current_frame+skip, wrap_around=wrap_around)
            if (df is None) and self.skip_empty_frames:
                continue
            break
        return df

    def get_previous_frame(self, skip=1, wrap_around=True):
        r""" Do not override this method. """
        df = None
        for _ in range(self.n_frames):
            df = self.get_frame(self.current_frame-skip, wrap_around=wrap_around)
            if (df is None) and self.skip_empty_frames:
                continue
            break
        return df

    def get_random_frame(self):
        r""" Do not override this method. """
        df = None
        for _ in range(self.n_frames):
            df = self.get_frame(int(np.floor(np.random.rand(1)*self.n_frames)))
            if (df is None) and self.skip_empty_frames:
                continue
            break
        return df

    def view(self, **kwargs):
        r""" Create a PADView instance and start it with this FrameGetter instance. """
        from ..viewers.qtviews.padviews import PADView
        pv = PADView(frame_getter=self, **kwargs)
        pv.start()

    def get_padview(self, **kwargs):
        r""" Create a PADView instance with this FrameGetter instance. """
        from ..viewers.qtviews.padviews import PADView
        return PADView(frame_getter=self, **kwargs)

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
