print("Loading python modules... not my fault if this takes forever...")
# import h5py
import re
import os
import sys
# sys.path.append('../..')
import argparse
import psana
# import numpy as np
import reborn
from reborn.viewers.qtviews import PADView
from reborn.external import crystfel, lcls, cheetah
from reborn.fileio.getters import FrameGetter


def error(message):
    sys.stderr.write(message + '\n')
    sys.exit(1)


print("Parsing arguments...")
parser = argparse.ArgumentParser(description='Simple diffraction viewer')
parser.add_argument('-r', dest='run', type=int, help='Run number', default=-1)
parser.add_argument('-e', dest='exp', type=str, help='Experiment ID, for example "amok8916"', default='')
parser.add_argument('-g', dest='geom', type=str, help='CrystFEL geomfile', default='psana')
parser.add_argument('-d', dest='det', type=str, help='Detector name, for example "cspad"', default='')
# parser.add_argument('-m', dest='mask', type=str, nargs=1, help='Mask file (hdf5)', default='')
args = parser.parse_args()

run = args.run
if run == -1:
    error("You need to specify the run number.  Use the -r flag, or -h flag for more help.")

experiment = args.exp
if experiment == "":
    error("You need to specify an experiment ID.  Use the -e flag, or -h flag for more help.")

data_source = psana.DataSource("exp=%s:run=%d:idx" % (experiment, run))
print('Data source: %s' % (data_source,))

detector_id = args.det
if detector_id == "":
    print("You need to specify a detector ID.  Use the -e flag, or -h flag for more help.")
    print("Here are some possible detectors:")
    detnames = psana.DetNames('detectors')
    reg = re.compile(".*CsPad|pnccd.*|.*Epix")
    goodnames = []
    for lst in detnames:
        print("")
        for name in lst:
            sys.stdout.write('%20s' % (name,))
    error("")

detector = lcls.AreaDetector(detector_id)
print('Using detector ID: %s' % (detector_id,))

# mask_file = args.mask
mask_data = None

geom_file = args.geom
if os.path.isfile(geom_file):
    # We assume that geom file is CrystFEL format, and that it relies on Cheetah's data layout conventions.
    geom_dict = crystfel.load_crystfel_geometry(geom_file)
    pad_geometry = crystfel.geometry_file_to_pad_geometry_list(geom_file)
    if detector.type == 'cspad':
        def split_pads(psana_array):
            return cheetah.cheetah_cspad_array_to_pad_list(psana_array, geom_dict)
    elif detector.type == 'pnccd':
        def split_pads(psana_array):
            return cheetah.cheetah_pnccd_array_to_pad_list(psana_array, geom_dict)
    elif detector.type == 'unknown':
        def split_pads(psana_array):
            if len(psana_array) != 2:
                error("I don't know how to transform this PAD data...")
            return [psana_array]
    detector._splitter = split_pads
    # if mask_file == "":
    #     print("No mask specified")
    # else:
    #     with h5py.File(mask_file, 'r') as fid:
    #         mask = np.array(fid['/data/data'])
    #         if detector_type == 'cspad' and mask.ndim == 2:
    #             # This is probably in the cheetah format
    #             mask_data = crystfel.split_image(mask, geom_dict)
    #         else:
    #             mask_data = None
else:
    pad_geometry = detector.get_pad_geometry(data_source.runs().next())


class MyFrameGetter(FrameGetter):
    def __init__(self, detector=None, data_source=None):
        FrameGetter.__init__(self)
        self.data_source = data_source
        self.run = self.data_source.runs().next()
        self.times = self.run.times()
        self.n_frames = len(self.times)
        self.current_frame = 0
        self.detector = detector
    def get_frame(self, frame_number=1):
        self.current_frame = frame_number
        event = self.run.event(self.times[self.current_frame])
        pad_data = self.detector.get_calib_split(event)
        return {'pad_data': pad_data}

print("Experiment ID: %s" % (experiment))
print("Run: %d" % (run))
print("Detector: %s" % (detector_id))
print("Geometry: %s" % (geom_file))

data_source = psana.DataSource("exp=%s:run=%d:idx" % (experiment, run))
frame_getter = MyFrameGetter(detector=detector, data_source=data_source)
padview = PADView(frame_getter=frame_getter, mask_data=mask_data, pad_geometry=pad_geometry)
padview.main_window.setWindowTitle('%s - run %d - %s' % (experiment, run, detector_id))
padview.start()
