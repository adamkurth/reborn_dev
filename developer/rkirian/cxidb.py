#!/usr/bin/env python

import h5py
import datetime
import os


class CxidbWriter(object):

    def __init__(self, stack_size=100, base_file_name='cxidb_', file_start_number=1, append_format='_%06d.cxi',
                 overwrite=False, experimental_identifier='no identifier'):

        r"""

        This is a generic file writing.  It splits cxidb files into separate files, each of which it should be possible
        to load the whole thing into memory.

        Args:
            stack_size: Number of frames before making new file
            base_file_name: File name (and path if need be), onto which a file number will be appended.
            file_start_number: Number of first file, which will be appended to the first file.
            append_format: Formatting for appending file number.  Default is '_%06d.cxi'.
            overwrite: Declare if files should be overwritten.  Default is of course False.
        """

        self.base_file_name = base_file_name
        self.current_file_name = os.path.join(base_file_name, append_format % self.current_file_number)

        if overwrite:
            self.open_file_config = 'w'  # Wverwrite existing file
        else:
            self.open_file_config = 'w-'  # Fail if the file exists
        self.fid = h5py.File(self.current_file_name, self.open_file_config)
        self.fid.create_dataset("cxi_version", data=120)

        # populate the file with the classes tree
        entry_1 = self.fid.create_group("entry_1")
        entry_1.create_dataset("experimental_identifier", data=experimental_identifier)
        entry_1.create_dataset("start_time", data=datetime.datetime.utcnow().isoformat())
        # sample_1 = entry_1.create_group("sample_1")
        # sample_1.create_dataset("name", data="None")
        instrument_1 = entry_1.create_group("instrument_1")
        instrument_1.create_dataset("name", data="None")
        source_1 = instrument_1.create_group("source_1")
        source_1.create_dataset("energy", data=2.8893e-16) # in J
        source_1.create_dataset("pulse_width", data=70e-15) # in s

        detector_1 = instrument_1.create_group("detector_1")
        detector_1.create_dataset("distance", data=0.15) # in meters
        detector_1.create_dataset("data", data=sinc1)

        detector_2 = instrument_1.create_group("detector_2")
        detector_2.create_dataset("distance", data=0.65) # in meters
        detector_2.create_dataset("data", data=sinc2)

        # data_1 = entry_1.create_group("data_1")
        # data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
        #
        # data_2 = entry_1.create_group("data_2")
        # data_2["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_2/data')

    def write_frame(self, dataframe=None):

        pass


    def close(self):

        self.fid.close()
