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

r""" Classes for handling dataframes. """
import numpy as np
from . import source, detector, utils


def warn(*args, **kwargs):
    utils.warn(':DataFrame:', *args, **kwargs)


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
    _dataset_id = 'Unknown dataset'
    # Cached arrays
    _q_mags = None
    _q_vecs = None
    _sa = None
    _pfac = None
    parameters = {}  # Should contain miscellaneous "parameters"; e.g. 'xrays_on', 'laser_on', etc.

    def __init__(self, raw_data=None, processed_data=None, mask=None, beam=None, pad_geometry=None, frame_id=0):
        if pad_geometry is not None:
            self.set_pad_geometry(pad_geometry)
        if beam is not None:
            self.set_beam(beam)
        if raw_data is not None:
            self.set_raw_data(raw_data)
        if mask is not None:
            self.set_mask(mask)
        if processed_data is not None:
            self.set_processed_data(processed_data)
        self.set_frame_id(frame_id)

    def validate(self):
        r""" Check that this dataframe is valid.  A valid dataframe must at minimum have a frame ID, a valid Beam
        instance, a valid PADGeometryList instance, and raw data.  """
        if self._frame_id is None:
            return False
        if self._beam is None:
            return False
        if self._pad_geometry is None:
            return False
        if self._raw_data is None:
            return False
        if self._beam.validate() is False:
            return False
        if self._pad_geometry.validate() is False:
            return False
        if not isinstance(self._raw_data, np.ndarray):
            return False
        return True

    def copy(self):
        r""" Makes a copy of the dataframe, including all internal data. """
        df = DataFrame()
        if self._pad_geometry is not None:
            df._pad_geometry = self._pad_geometry.copy()
        if self._beam is not None:
            df._beam = self._beam.copy()
        if self._raw_data is not None:
            df._raw_data = self._raw_data.copy()
        if self._processed_data is not None:
            df._processed_data = self._processed_data
        if self._pfac is not None:
            df._pfac = self._pfac.copy()
        if self._sa is not None:
            df._sa = self._sa.copy()
        if self._mask is not None:
            df._mask = self._mask.copy()
        return df

    @property
    def is_dark(self):
        if self._beam is None:
            return True
        return False

    def concat_data(self, data):
        if self._pad_geometry is None:
            warn('Your PADGeometry is not defined!  Define the geometry first.')
            return detector.concat_pad_data(data.copy())
        else:
            return self._pad_geometry.concat_data(data)

    def split_data(self, data):
        if self._pad_geometry is None:
            warn('Your PADGeometry is not defined!  Cannot split data.')
            return detector.split_pad_data(data.copy())
        else:
            return self._pad_geometry.split_data(data)

    @property
    def n_pads(self):
        r""" Number of PADs. """
        return len(self.get_processed_data_list())

    @property
    def raw_data(self):
        r""" Raw data (closest to the source). """
        return self.get_raw_data_flat()

    @property
    def processed_data(self):
        r""" Some modification of the raw data. """
        return self.get_processed_data_flat()

    @property
    def q_mags(self):
        r""" Concatenates the output of the corresponding function for each PADGeometry. """
        return self.get_q_mags_flat()

    @property
    def q_vecs(self):
        r""" Concatenates the output of the corresponding function for each PADGeometry. """
        return self.get_q_vecs()

    @property
    def solid_angles(self):
        r""" Concatenates the output of the corresponding function for each PADGeometry. """
        return self.get_solid_angles_flat()

    @property
    def polarization_factors(self):
        r""" Concatenates the output of the corresponding function for each PADGeometry. """
        return self.get_polarization_factors_flat()

    def clear_cache(self):
        r""" Deletes cached q_mags, q_vecs, solid_angles, polarization_factors"""
        self._q_mags = None
        self._q_vecs = None
        self._sa = None
        self._pfac = None

    def get_frame_index(self):
        r""" This is an integer index the is unique to this frame.  It is understood to be a context-dependent parameter
        that might, for example, be the index in a list of frames."""
        return self._frame_index

    def set_frame_index(self, index):
        r""" See corresponding get_frame_index method. """
        self._frame_index = int(index)

    def get_dataset_id(self):
        r""" Unique identifier for the parent dataset.

        For LCLS-I this would follow the "data source" convention: for example, "exp=cxil2316:run=56"
        """
        return self._dataset_id

    def set_dataset_id(self, val):
        r""" See the corresponding get_dataset_id method. """
        self._dataset_id = val

    def get_frame_id(self):
        r""" Unique identifier for this dataframe.  Most often this is an integer, but in some cases, such as the LCLS,
        it may be something else such as a tuple.  LCLS uses a tuple of integers: seconds, nanoseconds, and fiducial."""
        return self._frame_id

    def set_frame_id(self, frame_id):
        r""" See the corresponding get_frame_id method."""
        self._frame_id = frame_id

    def get_beam(self):
        r""" Get the |Beam| instance, which contains x-ray wavelength, beam direction, etc."""
        return self._beam.copy()

    def set_beam(self, beam):
        r""" See the corresponding get_beam method."""
        self.clear_cache()
        beam.validate()
        beam = beam.copy()
        self._beam = beam

    def get_pad_geometry(self):
        r""" Get the |PADGeometryList| of all PADs in the event.  A future version of this will likely accommodate
        multiple collections of PADs (e.g. when we have a SAXS detector and WAXS detector that are most reasonably
        analyzed separately.)."""
        return self._pad_geometry.copy()

    def set_pad_geometry(self, pads):
        r""" See the corresponding get_pad_geometry method. """
        self.clear_cache()
        # assert(isinstance(pads, detector.PADGeometryList))
        pads = detector.PADGeometryList(pads.copy())
        # pads.validate()
        self._pad_geometry = pads

    def get_raw_data_list(self):
        r""" Get the raw data as a list of 2D arrays."""
        return self.split_data(self.get_raw_data_flat())

    def get_raw_data_flat(self):
        r""" Get the raw data as a contiguous 1D array, with all PADs concatenated."""
        return self._raw_data.ravel()

    def set_raw_data(self, data):
        r""" Set the raw data.  You may pass a list or an |ndarray|.  Has the side effect of setting the 'writeable'
        flag of the array to False. """
        self._raw_data = self.concat_data(data)
        self._raw_data.flags.writeable = False
        self._processed_data = None

    def get_mask_list(self):
        r""" Get the mask as a list of 2D arrays."""
        return self.split_data(self.get_mask_flat())

    def get_mask_flat(self):
        r""" Get the mask as a contiguous 1D array, with all PADs concatenated."""
        if self._mask is None:
            self.set_mask(np.ones(self.get_raw_data_flat().shape, dtype=int))
        return self._mask.ravel()

    def set_mask(self, mask):
        r""" Set the mask.  You may pass a list or an |ndarray|."""
        if mask is None:
            self._mask = self._pad_geometry.ones(dtype=int)
        else:
            self._mask = self.concat_data(mask)
        self._mask.flags.writeable = False

    def get_processed_data_list(self):
        r""" See corresponding _raw_ method."""
        if self._processed_data is None:
            self._processed_data = self._raw_data.copy()
        return self.split_data(self.get_processed_data_flat())

    def get_processed_data_flat(self):
        r""" See corresponding _raw_ method."""
        if self._processed_data is None:
            self._processed_data = self._raw_data.copy()
            self._processed_data.flags.writeable = False
        return self._processed_data.ravel()

    def set_processed_data(self, data):
        r""" See corresponding _raw_ method."""
        d = self.concat_data(data)
        d.flags.writeable = False
        self._processed_data = d

    def clear_processed_data(self):
        r""" Clear the processed data.  After this operation, the get_processed_data method will return a copy of
        the raw data. """
        self._processed_data = None

    def get_q_vecs(self):
        r""" Get q vectors as an Nx3 array with all PADs concatenated. """
        if self._q_vecs is None:
            self._q_vecs = self._pad_geometry.q_vecs(self._beam)
            self._q_vecs.flags.writeable = False
        return self._q_vecs.copy()

    def get_q_mags_flat(self):
        r""" Get q magnitudes as a flat array. """
        if self._q_mags is None:
            self._q_mags = self._pad_geometry.q_mags(self._beam)
            self._q_mags.flags.writeable = False
        return self._q_mags.copy().ravel()

    def get_q_mags_list(self):
        r""" Get q magnitudes as a list of 2D arrays. """
        return self.split_data(self.get_q_mags_flat())

    def get_solid_angles_flat(self):
        r""" Get pixel solid angles as flat array. """
        if self._sa is None:
            self._sa = self._pad_geometry.solid_angles()
        return self._sa.copy().ravel()

    def get_polarization_factors_flat(self):
        r""" Get polarization factors as a flat array. """
        if self._pfac is None:
            self._pfac = self._pad_geometry.polarization_factors(beam=self._beam)
        return self._pfac.copy().ravel()

    # def save(self, filename):
    #     np.savez(filename, dataset_id=self._dataset_id,
    #                        frame_id=self._frame_id,
    #                        pad_geometry=self._pad_geometry,
    #                        beam=self._beam,
    #                        raw_data=self._raw_data,
    #                        processed_data=self._processed_data,
    #                        mask=self._mask)
    #
    # def load(self, filename):
    #     dat = np.load(filename)
    #     self.set_dataset_id(dat.get('dataset_id'))
    #     self.set_frame_id(dat.get('frame_id', 0))
    #     self.set_pad_geometry(dat.get('pad_geometry'))
    #     self.set_beam(dat.get('beam'))
    #     self.set_raw_data(dat.get('raw_data'))
    #     self.set_processed_data(dat.get('processed_data'))
    #     self.set_mask(dat.get('mask'))


    # def get_bragg_peaks(self):
    #     pass
    #
    # def set_bragg_peaks(self, bragg_peaks):
    #     pass

import pickle
def save_pickled_dataframe(df, filename):
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
def load_pickled_dataframe(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)