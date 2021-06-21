r""" Classes for handling dataframes. """
import numpy as np
from . import detector


class DataFrame:
    r"""
    A dataframe is a new concept in reborn.  It corresponds to a recording event, which is most often an XFEL pulse
    or a synchrotron exposure.  The interface of this class will be changing in the coming weeks.
    At minimum, it should hold the following data:

    - A |Beam| instance.
    - A |PADGeometryList|.
    - The "raw" PAD data arrays.
    - The "processed" PAD data arrays.  (Copy the "raw" data if there are no processing steps.)
    - An event ID.  This may be an integer, or any other data type (such as a tuple in the case of LCLS).
    """

    _frame_index = 0
    _frame_id = 0
    _pad_geometry = None
    _beam = None
    _raw_data = None
    _processed_data = None
    _mask = None
    # Cached arrays
    _q_mags = None
    _q_vecs = None

    def __init__(self, raw_data=None, processed_data=None, mask=None, beam=None, pad_geometry=None, frame_id=0):
        self.set_pad_geometry(pad_geometry)
        self.set_beam(beam)
        self.set_raw_data(raw_data)
        if mask is not None:
            self.set_mask(mask)
        if processed_data is not None:
            self.set_processed_data(processed_data)
        self.set_frame_id(frame_id)

    def validate(self):
        if self._frame_id is None: return False
        if self._beam is None: return False
        if self._pad_geometry is None: return False
        if self._raw_data is None: return False
        if self._beam is False: return False
        if self._beam.validate() is False: return False
        if self._pad_geometry.validate() is False: return False
        if not isinstance(self._raw_data, np.ndarray): return False
        return True

    @property
    def n_pads(self):
        return len(self.get_processed_data_list())

    def clear_cache(self):
        self._q_mags = None
        self._q_vecs = None

    def get_frame_index(self):
        r""" This is an integer index the is unique to this frame.  It is understood to be a context-dependent parameter
        that might, for example, be the index in a list of frames."""
        return self._frame_index

    def set_frame_index(self, index):
        r""" See corresponding get_ method. """
        self._frame_index = int(index)

    def get_frame_id(self):
        r""" Unique identifier for this dataframe.  Most often this is an integer, but in some cases, such as the LCLS,
        it may be something else such as a tuple.  LCLS uses a tuple of integers: seconds, nanoseconds, and fiducial."""
        # FIXME: This should somehow make a copy if need be.  No need to copy integers.
        return self._frame_id

    def set_frame_id(self, frame_id):
        r""" See the corresponding get_ method."""
        self._frame_id = frame_id

    def get_beam(self):
        r""" Get the |Beam| instance, which contains x-ray wavelength, beam direction, etc."""
        return self._beam.copy()

    def set_beam(self, beam):
        r""" See the corresponding get_ method."""
        self.clear_cache()
        beam = beam.copy()
        beam.validate(raise_error=True)
        self._beam = beam

    def get_pad_geometry(self):
        r""" Get the |PADGeometryList| of all PADs in the event.  A future version of this will likely accommodate
        multiple collections of PADs (e.g. when we have a SAXS detector and WAXS detector that are most reasonably
        analyzed separately.)."""
        return self._pad_geometry.copy()

    def set_pad_geometry(self, pads):
        r""" See the corresponding get_ method. """
        self.clear_cache()
        pads = detector.PADGeometryList(pads)
        pads.validate(raise_error=True)
        self._pad_geometry = pads.copy()

    def get_raw_data_list(self):
        r""" Get the raw data as a list of 2D arrays."""
        return self._pad_geometry.split_data(self.get_raw_data_flat())

    def get_raw_data_flat(self):
        r""" Get the raw data as a contiguous 1D array, with all PADs concatenated."""
        return self._raw_data.copy()

    def set_raw_data(self, data):
        r""" Set the raw data.  You may pass a list or a concatentated 1D array."""
        self._raw_data = self._pad_geometry.concat_data(data.copy()).astype(np.double)
        self._raw_data.flags.writeable = False
        self._processed_data = None

    def get_mask_list(self):
        r""" Get the mask as a list of 2D arrays."""
        return self._pad_geometry.split_data(self.get_mask_flat())

    def get_mask_flat(self):
        r""" Get the mask as a contiguous 1D array, with all PADs concatenated."""
        if self._mask is None:
            self.set_mask(np.ones(self.get_raw_data_flat().shape))
        return self._mask.copy()

    def set_mask(self, mask):
        r""" Set the mask.  You may pass a list or a concatentated 1D array."""
        self._mask = self._pad_geometry.concat_data(mask.copy())
        self._mask.flags.writeable = False

    def get_processed_data_list(self):
        r""" See corresponding _raw_ method."""
        if self._processed_data is None:
            self._processed_data = self._raw_data.copy()
        return self._pad_geometry.split_data(self.get_processed_data_flat())

    def get_processed_data_flat(self):
        r""" See corresponding _raw_ method."""
        if self._processed_data is None:
            self._processed_data = self._raw_data.copy()
        return self._processed_data.copy()

    def set_processed_data(self, data):
        r""" See corresponding _raw_ method."""
        self._processed_data = self._pad_geometry.concat_data(data).astype(np.double)

    def get_q_vecs(self):
        if self._q_vecs is None:
            self._q_vecs = self._pad_geometry.q_vecs(self._beam)
            self._q_vecs.flags.writeable = False
        return self._q_vecs.copy()

    def get_q_mags_flat(self):
        if self._q_mags is None:
            self._q_mags = self._pad_geometry.q_mags(self._beam)
            self._q_mags.flags.writeable = False
        return self._q_mags.copy()

    def get_q_mags_list(self):
        return self._pad_geometry.split_data(self.get_q_mags_flat())

    def get_bragg_peaks(self):
        pass

    def set_bragg_peaks(self, bragg_peaks):
        pass
