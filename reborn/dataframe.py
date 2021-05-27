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

    _frame_id = None
    _pad_geometry = None
    _beam = None
    _raw_data_list = None
    _raw_data = None
    _processed_data_list = None
    _processed_data = None

    def __init__(self, beam=None, pad_geometry=None, event_id=None):
        self.set_pad_geometry(pad_geometry)
        self.set_beam(beam)
        self.set_event_id(event_id)

    def get_frame_id(self):
        r""" Unique identifier for this dataframe.  Most often this is an integer, but in some cases, such as the LCLS,
        it may be something else such as a tuple.  LCLS uses a tuple of integers: seconds, nanoseconds, and fiducial."""
        # FIXME: This should somehow make a copy if need be.  No need to copy integers.
        return self._frame_id

    def set_frame_id(self, frame_id):
        r""" See the corresponding get_ method."""
        self._frame_id = frame_id

    def get_beam(self):
        r""" Get the |Beam| instance."""
        return self._beam.copy()

    def set_beam(self, beam):
        r""" See the corresponding get_ method."""
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
        pads = detector.PADGeometryList(pads)
        pads.validate(raise_error=True)
        self._pad_geometry = pads.copy()

    def get_raw_data_list(self):
        r""" Get the raw data as a list of 2D arrays."""
        pass

    def get_raw_data_flat(self):
        r""" Get the raw data as a contiguous 1D array, with all PADs concatenated."""
        pass

    def set_raw_data(self, raw_data):
        r""" Set the raw data.  You may pass a list or a concatentated 1D array."""

    def get_processed_data_list(self):
        r""" See corresponding _raw_ method."""
        pass

    def get_processed_data_flat(self):
        r""" See corresponding _raw_ method."""
        pass

    def set_processed_data_flat(self):
        r""" See corresponding _raw_ method."""
        pass

    def get_q_vecs(self):
        return self._pad_geometry.q_vecs(self._beam)

    def get_bragg_peaks(self):
        pass

    def set_bragg_peaks(self, bragg_peaks):
        pass
