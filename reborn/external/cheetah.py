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
Utilities for working with data created by Cheetah.  I make no promises that any of this will work as expected; Cheetahs
are wild animals.
"""

import h5py
import numpy as np
from . import crystfel
from ..fileio.getters import FrameGetter


# =======================================================================================================================
# Functions that are specific to the CSPAD and CrystFEL geom files
# =======================================================================================================================

def reshape_psana_cspad_array_to_cheetah_array(psana_array):
    r"""
    Transform  a native psana cspad numpy array of shape (32,185,388) into a "Cheetah array" of shape (1480, 1552).
    Conversion to Cheetah format requires a re-write of the data in memory, and each detector panel is no longer stored
    contiguously in memory.

    Arguments:
        psana_array (numpy array) :
            A numpy array of shape (32,185,388) produced by the psana module

    Returns:
        cheetah_array (numpy array) :
            A numpy array of shape (1480, 1552); same data as the psana array but mangled as done within Cheetah
    """

    imlist = []
    for i in range(0, 4):
        slist = []
        for j in range(0, 8):
            slist.append(psana_array[j + i * 8, :, :])
        imlist.append(np.concatenate(slist))
    cheetah_array = np.concatenate(imlist, axis=1)

    return cheetah_array


def cheetah_cspad_array_to_pad_list(psana_array, geom_dict):
    r"""
    This function is helpful if you have a CrystFEL geom file that refers to Cheetah output, but you wish to work with
    data in the native psana format.  First you should create a crystfel geometry dictionary using the function
    :func:`geometry_file_to_pad_geometry_list() <reborn.external.crystfel.geometry_file_to_pad_geometry_list>`.

    Arguments:
        psana_array (numpy array) :
            A numpy array of shape (32,185,388) produced by the psana module.
        geom_dict (dict) :
            A CrystFEL geometry dictionary produced by external.crystfel.geom_to_dict() .

    Returns:
        pad_list (list) :
            A list containing data from each pixel array
    """

    slab = reshape_psana_cspad_array_to_cheetah_array(psana_array)

    return crystfel.split_image(slab, geom_dict)


# def cheetah_remapped_cspad_array_to_pad_list(cheetah_array, geom_dict):
#
#     utils.depreciate('Dont use cheetah_remapped_cspad_array_to_pad_list() function.  Instead, use'
#                      'crystfel.split_image()')
#
#     return crystfel.split_image(cheetah_array, geom_dict)

# =======================================================================================================================
# Functions that are specific to the pnCCD and CrystFEL geom files
# =======================================================================================================================

def reshape_psana_pnccd_array_to_cheetah_array(psana_array):
    r"""
    Transform  a native psana pnccd numpy array of shape (???) into a "Cheetah array" of shape (1024,1024).
    Conversion to Cheetah format requires a re-write of the data in memory.  Panels might not be contiguous in memory.

    Arguments:
        psana_array (numpy array) :
            A numpy array of shape (???) produced by the psana module

    Returns:
        cheetah_array (numpy array) :
            A numpy array of shape (1024, 1024); same data as the psana array but mangled as done within Cheetah
    """

    slab = np.zeros((1024, 1024), dtype=psana_array.dtype)
    slab[0:512, 0:512] = psana_array[0,:,:]
    slab[512:1024, 0:512] = psana_array[1,::-1, ::-1]
    slab[512:1024, 512:1024] = psana_array[2,::-1, ::-1]
    slab[0:512, 512:1024] = psana_array[3,:,:]

    return slab


def cheetah_pnccd_array_to_pad_list(psana_array, geom_dict):
    r"""
    This function is helpful if you have a CrystFEL geom file that refers to Cheetah output, but you wish to work with
    data in the native psana format.  First you should create a crystfel geometry dictionary using the function
    :func:`geometry_file_to_pad_geometry_list() <reborn.external.crystfel.geometry_file_to_pad_geometry_list>`.

    Arguments:
        psana_array (numpy array) :
            A numpy array of shape (32,185,388) produced by the psana module.
        geom_dict (dict) :
            A CrystFEL geometry dictionary produced by external.crystfel.geom_to_dict() .

    Returns:
        pad_list (list) :
            A list containing data from each pixel array
    """

    slab = reshape_psana_pnccd_array_to_cheetah_array(psana_array)

    return crystfel.split_image(slab, geom_dict)


class CheetahFrameGetter(FrameGetter):

    r"""

    A frame getter that attempts to read the CXIDB variants that are written by Cheetah.

    """

    skip_peaks = False
    fs_min = 0
    fs_max = 0
    ss_min = 0
    ss_max = 0

    def __init__(self, cxi_file_name=None, geom_file_name=None):

        FrameGetter.__init__(self)
        self.geom_dict = crystfel.load_crystfel_geometry(geom_file_name)
        self.pad_geometry = crystfel.geometry_file_to_pad_geometry_list(geom_file_name)
        self.n_pads = len(self.pad_geometry)
        self.load_cxidb_file(cxi_file_name)
        self.current_frame = 0

        self.peaks = None
        
        try: 
            self.photon_energies = self.h5file['/LCLS/photon_energy_eV'][:] * 1.602e-19
        except:
            self.photon_energies = None

        try:
            self.encoder_values = self.h5file['/LCLS/detector_1/EncoderValue'][:]
        except:
            self.encoder_values = None 

    def load_cxidb_file(self, cxi_file_name):

        self.h5file = h5py.File(cxi_file_name, 'r')
        self.h5_data = self.h5file['/entry_1/data_1/data']
        self.n_frames = self.h5_data.shape[0]

    def get_peaks(self, h5file, frame_number):

        n_peaks = h5file['entry_1/result_1/nPeaks'][frame_number]

        if n_peaks <= 0:
            return None

        fs_pos_raw = h5file['entry_1/result_1/peakXPosRaw'][frame_number, 0:n_peaks]
        ss_pos_raw = h5file['entry_1/result_1/peakYPosRaw'][frame_number, 0:n_peaks]

        if self.peaks is None:
            fs_min = np.zeros(self.n_pads)
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
            indices = np.argwhere((fs_pos_raw > fs_min[i]) * (fs_pos_raw <= fs_max[i]) *
                                  (ss_pos_raw > ss_min[i]) * (ss_pos_raw <= ss_max[i]))
            if len(indices > 0):
                pad_numbers[indices] = i
                fs_pos[indices] = fs_pos_raw[indices] - fs_min[i]
                ss_pos[indices] = ss_pos_raw[indices] - ss_min[i]

        centroids = [None]*self.n_pads
        for i in range(self.n_pads):
            w = np.where(pad_numbers == i)[0]
            n = len(w)
            if n > 0:
                p = np.zeros([n, 2])
                p[:, 0] = fs_pos[w]
                p[:, 1] = ss_pos[w]
            else:
                continue
            centroids[i] = p

        peaks = {'n_peaks': n_peaks, 'centroids': centroids,
                 'pad_numbers': pad_numbers, 'fs_pos': fs_pos, 'ss_pos': ss_pos,
                 'peakXPosRaw': fs_pos_raw, 'peakYPosRaw': ss_pos_raw}

        return peaks

    def get_frame(self, frame_number=0):

        dat = np.array(self.h5_data[frame_number, :, :]).astype(np.double)
        pad_data = crystfel.split_image(dat, self.geom_dict)
        dat = {'pad_data': pad_data}

        if not self.skip_peaks:
            peaks = self.get_peaks(self.h5file, frame_number)
            dat['peaks'] = peaks

        if self.photon_energies is not None:
            dat['photon_energy'] = self.photon_energies[frame_number]

        if self.encoder_values is not None:
            dat['encoder_value'] = self.encoder_values[frame_number]

        return dat
