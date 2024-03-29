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
Utilities for working with CrystFEL files.
Most of the functions below wrap around the functions in the crystfel_utils module within the cfelpyutils package,
which is maintained by CFEL.
The crystfel_utils module is included in reborn so that you do not need to install it with pip.
"""
import os
import h5py
import numpy as np
import linecache
from .. import detector, utils, const
from ..fileio.getters import FrameGetter
from . import _crystfel_utils
import pkg_resources

example_stream_file_path = pkg_resources.resource_filename('reborn', 'data/misc/test.stream')
pnccd_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/pnccd_crystfel.geom')
epix10k_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/epix10k_crystfel.geom')
cspad_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/cspad_crystfel.geom')

# ------------------------------------------------
# Stream file delimiters
sta_geom    = "----- Begin geometry file -----"
sta_chunk   = "----- Begin chunk -----"
sta_crystal = "--- Begin crystal"

end_geom    = "----- End geometry file -----"
end_chunk   = "----- End chunk -----"
end_crystal = "--- End crystal"
# ------------------------------------------------


def load_crystfel_geometry(geometry_file):

    r"""
    Given a CrystFEL geometry file, create a python dictionary object.  The structure of this dictionary is defined
    by the cfelpyutils.crystfel_utils module, which is maintained by CFEL and included in reborn for convenience.

    Arguments:
        geometry_file (str): Path to geometry file

    Returns:
        Dict
    """

    return _crystfel_utils.load_crystfel_geometry(geometry_file)


def geometry_dict_to_pad_geometry_list(geometry_dict):

    r"""
    Given a CrystFEL geometry dictionary, create a list of `:class:<reborn.geometry.PADGeometry` objects.
    This will also append the name of the panel to the PADGeometry instance.

    Arguments:
        geometry_dict (str): Path to geometry file

    Returns: List of PADGeometry instances

    """

    geom = geometry_dict

    pads = []
    for panel_name in geometry_dict['panels'].keys():
        pad = detector.PADGeometry()
        pad.name = panel_name
        p = geom['panels'][panel_name]
        # for k in p.keys():
        #     print(k, p[k])
        # print('\n')
        dist = 0
        if p['coffset']:
            dist += p['coffset']
        if isinstance(p['clen'], str):
            pad.crystfel_clen = p['clen']
        else:
            if p['clen'] != -1:  # Why is clen sometimes -1?  Very annoying.
                dist += p['clen']
        pix = 1.0 / p['res']
        pad.fs_vec = np.array([p['fsx'], p['fsy'], p['fsz']]) * pix
        pad.n_fs = p['max_fs'] - p['min_fs'] + 1
        pad.ss_vec = np.array([p['ssx'], p['ssy'], p['ssz']]) * pix
        pad.n_ss = p['max_ss'] - p['min_ss'] + 1
        pad.t_vec = np.array([p['cnx'] * pix, p['cny'] * pix, dist])
        pad.crystfel_max_ss = p['max_ss']
        pad.crystfel_max_fs = p['max_fs']
        pad.crystfel_min_ss = p['min_ss']
        pad.crystfel_min_fs = p['min_fs']
        pad.crystfel_dim_structure = p.get('dim_structure', None)
        pads.append(pad)

    pads = detector.PADGeometryList(pads)
    maxfs = 0
    maxss = 0
    max3 = 0
    for p in pads:
        maxfs = max(maxfs, p.crystfel_max_fs)
        maxss = max(maxss, p.crystfel_max_ss)
        # This is a dirty hack... assumes that dim_structure is either ['%', int, 'ss', fs'] or [int, 'ss', 'fs'] or ['ss', 'fs'] 
        for d in p.crystfel_dim_structure:
            if isinstance(d, int):
                max3 = max(d, max3)
    for p in pads:
        if max3 <= 0:
            p.parent_data_shape = (maxss+1, maxfs+1)
            p.parent_data_slice = np.s_[p.crystfel_min_ss:p.crystfel_max_ss + 1, p.crystfel_min_fs:p.crystfel_max_fs + 1]    
        else:
            p.parent_data_shape = (max3+1, maxss+1, maxfs+1)
            d = 0
            for di in p.crystfel_dim_structure:
                if isinstance(di, int):
                    d = di
            p.parent_data_slice= np.s_[d, p.crystfel_min_ss:p.crystfel_max_ss + 1, p.crystfel_min_fs:p.crystfel_max_fs + 1]
    return pads


def fix_cspad_cheetah_indexing(pads):
    p = pads
    for i in range(32):
        s = slice(None, None, None)
        p[2*i].parent_data_slice = (i, s, slice(0, 194, None))
        p[2*i+1].parent_data_slice = (i, s, slice(194, 388, None))
        p[2*i].parent_data_shape = (32, 185, 388)
        p[2*i+1].parent_data_shape = (32, 185, 388)
    return p


def extract_geom_from_stream(stream_path, geom_path=None):

    r"""
    Extract a CrystFEL geometry file from a CrystFEL stream file.

    Args:
        stream_path:
        geom_path:

    Returns:
        None
    """

    # Open the files
    streamfile = open(stream_path, mode='r')
    geometryfile_to_write = open(geom_path, mode='w')

    # Start reading the stream file
    flag_done = 0
    for line in streamfile:  # Reading lines in streamfile
        if sta_geom in line:  # In geometery file
            for line2 in streamfile:  # Reading lines in chunk
                if end_geom in line2:
                    flag_done = 1  # Done writing the geometry file
                    break
                geometryfile_to_write.write(line2)
        if flag_done == 1:
            break

    # close the files
    streamfile.close()
    geometryfile_to_write.close()


def geometry_file_to_pad_geometry_list(geometry_file):
    r"""
    Given a CrystFEL geometry file, create a list of `:class:<reborn.geometry.PADGeometry` objects.  This will also
    append extra crystfel-specific items like fsx, max_fs, etc.

    Arguments:
        geometry_file (str): Path to geometry file

    Returns:
        List of PADGeometry instances
    """

    geometry_dict = load_crystfel_geometry(geometry_file)
    pad_list = geometry_dict_to_pad_geometry_list(geometry_dict)

    return detector.PADGeometryList(pad_list)


def split_data_block(data, geom_dict, frame_number=0):
    r"""
    Split a chunk of contiguous data (usually 3D) into a list of pad data.
    """
    split_data = []
    for panel_name in geom_dict['panels']:
        p = geom_dict['panels'][panel_name]
        d = p['dim_structure']
        if len(d) != data.ndim:
            print("The number of data dimensions does not match the geom specification")
            print("geom dim_structure", d)
            print("data shape", data.shape)
            raise ValueError("The number of data dimensions does not match the geom specification.")
        fs_idx = d.index('fs')
        ss_idx = d.index('ss')
        try:
            fn_idx = d.index('%')
        except ValueError:
            fn_idx = None
        p_idx = None
        try:  # Panel index identified by checking which type is an integer
            for i in range(len(d)):
                if type(d[i]) == int:
                    p_idx = i
        except ValueError:
            pass
        r = [None]*len(d)
        r[fs_idx] = [p['min_fs'], p['max_fs'] + 1]
        r[ss_idx] = [p['min_ss'], p['max_ss'] + 1]
        if fn_idx is not None:
            r[fn_idx] = [frame_number, frame_number + 1]
        if p_idx is not None:
            r[p_idx] = [int(d[p_idx]), int(d[p_idx]) + 1]
        if data.ndim == 4:
            im = data[r[0][0]:r[0][1], r[1][0]:r[1][1], r[2][0]:r[2][1], r[3][0]:r[3][1]]
        elif data.ndim == 3:
            im = data[r[0][0]:r[0][1], r[1][0]:r[1][1], r[2][0]:r[2][1]]
        else:
            im = data[r[0][0]:r[0][1], r[1][0]:r[1][1]]
        im = np.squeeze(np.array(im))
        if ss_idx > fs_idx:
            im = im.T.copy()
        split_data.append(im)
    return split_data


def split_image(data, geom_dict):
    r"""
    Split a 2D image into individual panels (useful for working with Cheetah output).
    If the input data is not a 2D image, then attempt to reshape it according to the
    expected shape as specified in the geometry dictionary.

    Arguments:
        data (numpy array) : Contiguous block of image data
        geom_dict (dict) : Geometry dictionary

    Returns:
        split_data (list) :
            List of individual PAD panel data
    """
    #if len(data.shape) != 2:
    n_fs = 0
    n_ss = 0
    for panel_name in geom_dict['panels']:
        p = geom_dict['panels'][panel_name]
        n_ss = max(n_ss, p['max_ss'] + 1)
        n_fs = max(n_fs, p['max_fs'] + 1)
    data = data.reshape(n_ss, n_fs)
    split_data = []
    for panel_name in geom_dict['panels']:
        p = geom_dict['panels'][panel_name]
        split_data.append(data[p['min_ss']:(p['max_ss'] + 1), p['min_fs']:(p['max_fs'] + 1)])
    return split_data


def unsplit_image(data, geom_dict):
    r"""
    Undo the action of split_image

    Arguments:
        data (list of |ndarray|) :  List of individual pads
        geom_dict (dict) : Geometry dictionary

    Returns:
        |ndarray|
    """
    n_fs = 0
    n_ss = 0
    for panel_name in geom_dict['panels']:
        p = geom_dict['panels'][panel_name]
        n_ss = max(n_ss, p['max_ss'] + 1)
        n_fs = max(n_fs, p['max_fs'] + 1)
    data_cont = np.zeros((n_ss, n_fs), dtype=data[0].dtype)
    for (i, panel_name) in enumerate(geom_dict['panels']):
        p = geom_dict['panels'][panel_name]
        data_cont[p['min_ss']:(p['max_ss'] + 1), p['min_fs']:(p['max_fs'] + 1)] = data[i]
    return data_cont


def write_geom_file_single_pad(file_path=None, beam=None, pad_geometry=None):
    r""" 
    Simple geom file writer.  Do not use this -- the file does not adhere 
    to the CrystFEL specifications, and we do not attempt to maintain compatibility...
    """
    pad = pad_geometry
    geom_file = os.path.join(file_path)
    fid = open(geom_file, 'w')
    fid.write("photon_energy = %g\n" % (beam.photon_energy / const.eV))
    fid.write("clen = %g\n" % pad.t_vec.flat[2])
    fid.write("res = %g\n" % (1 / pad.pixel_size()))
    fid.write("adu_per_eV = %g\n" % (1.0 / (beam.photon_energy / const.eV)))
    fid.write("0/min_ss = 0\n")
    fid.write("0/max_ss = %d\n" % (pad.n_ss - 1))
    fid.write("0/min_fs = 0\n")
    fid.write("0/max_fs = %d\n" % (pad.n_fs - 1))
    fid.write("0/corner_x = %g\n" % ((pad.t_vec.flat[0] - pad.pixel_size())/2.,))
    fid.write("0/corner_y = %g\n" % ((pad.t_vec.flat[1] - pad.pixel_size())/2,))
    fid.write("0/fs = x\n")
    fid.write("0/ss = y\n")
    fid.close()


def write_geom_file_from_template(pad_geometry=None, template_file=None, out_file=None):
    r"""
    Given a matching template, write a geom file with the geometry of a corresponding PADGeometryList
    """
    pads = pad_geometry
    f = open(template_file, 'r')
    out = open(out_file, 'w')
    in_panel = False
    for line in f:
        line = line.strip()
        if len(line) == 0:  # Blank line
            out.write('\n')
            continue
        if line[0] == ';':  # Comment
            out.write(line+'\n')
            continue
        key = line.split('=')[0].strip()
        val = line.split('=')[1].strip()
        if key == 'res':
            res = float(val);
            out.write(line+'\n')
            continue
        if '/' in key:  # Then this is a panel
            name = key.split('/')[0]
            key = key.split('/')[1]
            if key == 'res':
                res = float(val);
                out.write(line+'\n')
                continue
            if key == 'fs':
                v = utils.vec_norm(pads.get_by_name(name).fs_vec)
                out.write(name+'/'+key+' = '+'%+g'%v[0]+'x '+'%+g'%v[1]+'y '+'%+g'%v[2]+'z\n')
                continue
            if key == 'ss':
                v = utils.vec_norm(pads.get_by_name(name).ss_vec)
                out.write(name+'/'+key+' = '+'%+g'%v[0]+'x '+'%+g'%v[1]+'y '+'%+g'%v[2]+'z\n')
                continue
            # ==== Translation vector of this PAD ==============
            if key == 'corner_x':
                x = pads.get_by_name(name).t_vec[0]*res
                out.write(name+'/'+key+' = '+'%+g'%x+'\n')
                continue
            if key == 'corner_y':
                y = pads.get_by_name(name).t_vec[1]*res
                z = pads.get_by_name(name).t_vec[2]*res
                out.write(name+'/'+key+' = '+'%+g'%y+'\n')
                out.write(name+'/'+'clen = '+'%+g'%z+'\n')
                continue
            if (key == 'clen') or (key == 'coffset'):
                out.write(';'+line+'\n')
                continue
        if (key == 'detector_shift_x') or (key == 'detector_shift_y') or (key == 'clen'):
            out.write(';'+line+'\n')
            continue
        out.write(line+'\n')


def readStreamfile_get_total_number_of_frames(stream_file):
    r"""
    Get the total number of frames from a stream file.

    Arguments:
        stream_file (str): Path to the stream file

    Returns: int
    """
    f = open(stream_file, 'r') 
    count = 0
    for line in f:
        if sta_chunk in line:
            count += 1
    f.close()
    return count


def readStreamfile_get_nth_frame(streamfile_name, n):
    r"""
    Get the A matrix, CXI file path and CXI frame number from the nth frame in a stream file.
    """
    f = open(streamfile_name, 'r')
    count = 0
    for line in f:
        if sta_chunk in line:
            count += 1
        if count == n:
            break
    # Initialising 
    A = np.zeros((3,3))
    cxiFilepath = 0
    cxiFileFrameNumber = 0

    astar_exist = False

    for line in f:
        if "Image filename:" in line:
            cxiFilepath = line[16:-1]

        if "Event:" in line:
            cxiFileFrameNumber = int(line[9:])

        if "astar = " in line:
            A[0,0] = float(line[8:19])
            A[0,1] = float(line[19:30])
            A[0,2] = float(line[29:40])
            astar_exist = True

        if "bstar = " in line:
            A[1,0] = float(line[8:19])
            A[1,1] = float(line[19:30])
            A[1,2] = float(line[29:40])

        if "cstar = " in line:
            A[2,0] = float(line[8:19])
            A[2,1] = float(line[19:30])
            A[2,2] = float(line[29:40])

        if end_chunk in line:
            break

    # If frame does not contain a star, etc. set the A star matrix to None
    if astar_exist == False:
        A = None

    # close the file
    f.close()

    return A, cxiFilepath, cxiFileFrameNumber


class StreamfileFrameGetter(FrameGetter):
    r"""
    A frame getter that reads a CrystFEL stream file along with CXI files.  
    """

    # Initialise class variables
    h5_file_path = None
    h5_file = None
    current_frame = 0
    n_frames = 0
    cxi_path_replace = None  # Set to tuple of strings (remove, replace)

    def __init__(self, stream_file=None, geom_file=None, indexed_only=False):
        r"""
        Arguments:
            stream_file (str): Path to the stream file
            geom_file (str): Optional path to a geom file (else search the stream for geometry)
        """
        super().__init__()
        self.stream_file = stream_file
        with open(stream_file, 'r') as f: 
            self.begin_chunk_lines = []  # Cache where the stream chunks are
            for (n, line) in enumerate(f):
                if sta_chunk in line:
                    if indexed_only:
                        self.begin_chunk_lines.append(n)
                        self.n_frames += 1
                        dat = self.get_frame(frame_number=self.n_frames - 1, no_pad=True)
                        if dat['A_matrix'] is None:
                            self.n_frames -= 1
                            self.begin_chunk_lines.pop()
                            continue
                    else:
                        self.begin_chunk_lines.append(n)
                        self.n_frames += 1
        if geom_file is None:
            geom_file = stream_file  # The stream file should have geometry in it, so we try to use it
        self.geom_file = geom_file
        self.geom_dict = load_crystfel_geometry(geom_file)
        self.pad_geometry = geometry_file_to_pad_geometry_list(geom_file)

    def get_frame(self, frame_number=0, **kwargs):
        if 'no_pad' in kwargs:
            self.no_pad = kwargs['no_pad']
            del kwargs['no_pad']
        else:
            self.no_pad = False
        return super().get_frame(frame_number=frame_number)

    def get_data(self, frame_number=0):
        if frame_number >= self.n_frames:
            return None
        A = np.zeros((3,3))
        cxi_file_path = None
        cxi_frame_number = None
        photon_energy = None
        n = self.begin_chunk_lines[frame_number]
        dat = {}
        for i in range(1, int(1e6)):
            line = linecache.getline(self.stream_file, n + i)
            if end_chunk in line:
                break
            if "Image filename:" in line:
                cxi_file_path = line[16:-1]
                continue
            if "Event:" in line:
                cxi_frame_number = int(line[9:])
                continue
            if "photon_energy_eV" in line:
                photon_energy = float(line.split('=')[1])*1.602e-19
                continue
            if "astar = " in line:
                A[0,0] = float(line[8:19])
                A[0,1] = float(line[19:30])
                A[0,2] = float(line[29:40])
                continue
            if "bstar = " in line:
                A[1,0] = float(line[8:19])
                A[1,1] = float(line[19:30])
                A[1,2] = float(line[29:40])
                continue
            if "cstar = " in line:
                A[2,0] = float(line[8:19])
                A[2,1] = float(line[19:30])
                A[2,2] = float(line[29:40])
                continue
            line_split = line.split(':')
            if len(line_split) == 2:
                dat[line_split[0].strip()] = line_split[1].strip()
            line_split = line.split('=')
            if len(line_split) == 2:
                dat[line_split[0].strip()] = line_split[1].strip()
        if np.sum(A) == 0:
            A = None
        dat['A_matrix'] = A
        if self.cxi_path_replace is not None:
            cxi_file_path = cxi_file_path.replace(*self.cxi_path_replace)
        dat['cxi_file_path'] = cxi_file_path
        dat['cxi_frame_number'] = cxi_frame_number
        dat['photon_energy'] = photon_energy
        if not self.no_pad:
            # Extract data from the cxi file
            if self.h5_file_path != cxi_file_path:
                self.h5_file_path = cxi_file_path
                self.h5_file = h5py.File(self.h5_file_path, 'r')
            h5_data = self.h5_file['/entry_1/data_1/data']
            pad_data = np.array(h5_data[cxi_frame_number, :, :]).astype(np.double)
            pad_data = split_image(pad_data, self.geom_dict)
            dat['pad_data'] = pad_data
        return dat

    def __del__(self):
        linecache.clearcache()

