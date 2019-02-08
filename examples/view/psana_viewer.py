print("Loading python modules...")
import h5py
import re
import os
import sys
sys.path.append('../..')
import argparse
import psana
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyqtgraph as pg
import bornagain
from bornagain.viewers.qtviews import PADView
from bornagain.external.cheetah import cheetah_cspad_array_to_pad_list, cheetah_remapped_cspad_array_to_pad_list 
from bornagain.external import crystfel, lcls
from bornagain.fileio.getters import FrameGetter

print("Parsing arguments...")
parser = argparse.ArgumentParser(description='Simple diffraction viewer')
parser.add_argument('-r', dest='run', type=int, nargs=1, help='Run number', default=[-1])
parser.add_argument('-e', dest='exp', type=str, nargs=1, help='Experiment ID', default=[''])
parser.add_argument('-g', dest='geom', type=str, nargs=1, help='CrystFEL geomfile', default=['psana'])
parser.add_argument('-d', dest='det', type=str, nargs=1, help='Detector name', default=[''])
parser.add_argument('-m', dest='mask', type=str, nargs=1, help='Mask file (hdf5)', default=[''])
args = parser.parse_args()


def error(message):
    sys.stderr.write(message + '\n')
    sys.exit(1)


run = args.run[0]
experiment = args.exp[0]
geom_file = args.geom[0]
detector_id = args.det[0]
mask_file = args.mask[0]

if run == -1:
    error("You need to specify the run number.  Use the -r flag, or -h flag for more help.")

if experiment == "":
    print("You didn't specify an experiment.  I will try to guess...")
    srch = re.compile('cxi*|amo*')
    cwd = os.getcwd().split(os.path.sep)
    lst = list(filter(srch.match, cwd))
    if len(lst) != 2:
        error("Can't figure out what experiment you want to look at... please specify.")
    experiment = lst[1]
    print("Found experment ID based on your path...")
data_source = psana.DataSource("exp=%s:run=%d:idx" % (experiment, run))

if detector_id == "":
    print("You didn't specify a detector.  I will try to guess...")
    names = psana.DetNames('detectors')
    reg = re.compile(".*CsPad|pnccd.*|.*Epix")
    goodnames = []
    for lst in names:
        for name in lst:
            if reg.match(name) is not None:
                goodnames.append(reg.match(name).group(0))
    if len(goodnames) == 1:
        print("Found a detector to look at (there might be others...).")
        detector_id = goodnames[0]
    if len(goodnames) > 1:
        print("Found multiple detector IDs in this run:")
        for lst in names:
            for name in lst:
                if name:
                    print('\t' + name)
        error("Specify the detector you want to view with the -d flag.")
    if len(goodnames) < 1:
        print("Didn't find a good detector to look at.  Maybe you want to see one of these:")
        for lst in names:
            for name in lst:
                if name:
                    print('\t' + name)
        error("Specify the detector you want to view with the -d flag.")

data_source = psana.DataSource("exp=%s:run=%d:idx" % (experiment, run))
print(detector_id)
detector = lcls.AreaDetector(detector_id)

mask_data = None
if os.path.isfile(geom_file):
    # Everything needs to be handled differently with geom files... please make this go away...
    geom_dict = crystfel.load_crystfel_geometry(geom_file)
    pad_geometry = crystfel.geometry_file_to_pad_geometry_list(geom_file)
    if detector.type == 'cspad':
        def split_pads(psana_array):
            return cheetah_cspad_array_to_pad_list(psana_array, geom_dict)
    elif detector.type == 'pnccd':
        def split_pads(psana_array):
            slab = np.zeros((1024, 1024), dtype=psana_array.dtype)
            slab[0:512, 0:512] = psana_array[0]
            slab[512:1024, 0:512] = psana_array[1][::-1, ::-1]
            slab[512:1024, 512:1024] = psana_array[2][::-1, ::-1]
            slab[0:512, 512:1024] = psana_array[3]
            data_list = []
            for panel_name in geom_dict['panels'].keys():
                g = geom_dict['panels'][panel_name]
                d = slab[np.int(g['min_ss']):np.int(g['max_ss']), np.int(g['min_fs']):np.int(g['max_fs'])]
                data_list.append(d)
            return data_list
    elif detector.type == 'unknown':
        def split_pads(psana_array):
            if len(psana_array) != 2:
                error("I don't know how to transform this PAD data...")
            return [psana_array]
    detector._splitter = split_pads
    if mask_file == "":
        print("No mask specified")
    else:
        with h5py.File(mask_file, 'r') as fid:
            mask = np.array(fid['/data/data'])
            if detector_type == 'cspad' and mask.ndim == 2:
                # This is probably in the cheetah format
                mask_data = crystfel.split_image(mask, geom_dict)
            else:
                mask_data = None
elif geom_file == "psana":
    pad_geometry = detector.get_pad_geometry(data_source.runs().next())
else:
    error("Couldn't locate the geometry file %s" % (geom_file,))


class MyFrameGetter(FrameGetter):

    def __init__(self, detector=None, data_source=None):

        FrameGetter.__init__(self)
        
        # self.geom_file = geom_file
        # self.pad_geometry = pad_geometry
        self.data_source = data_source
        # self.run_number = run
        self.run = self.data_source.runs().next()
        self.times = self.run.times()
        self.n_frames = len(self.times)
        self.current_frame = 0      
        #print("Found %d frames in the xtc files for run %d." % (self.n_frames, self.run_number))

        self.detector = detector

    def get_frame(self, frame_number=1):

        self.current_frame = frame_number
        event = self.run.event(self.times[self.current_frame])
        return {'pad_data': self.detector.get_calib_split(event)}

print("Experiment ID: %s" % (experiment))
print("Run: %d" % (run))
print("Detector: %s" % (detector_id))
print("Geometry: %s" % (geom_file))

data_source = psana.DataSource("exp=%s:run=%d:idx" % (experiment, run))
frame_getter = MyFrameGetter(detector=detector, data_source=data_source)
padview = PADView(frame_getter=frame_getter, mask_data=mask_data, pad_geometry=pad_geometry)
padview.main_window.setWindowTitle('%s - run %d - %s' % (experiment, run, detector_id))
padview.start()
