import sys
import time
import numpy as np
from reborn import source, detector, utils, dataframe
from reborn.fileio.getters import FrameGetter
from reborn.target import crystal
from reborn.viewers.qtviews import PADView
from reborn.fortran import crystal_f
np.random.seed(0)
beam = source.Beam(wavelength=1.5e-10)
geom = detector.jungfrau4m_pad_geometry_list(detector_distance=0.3)
# print(geom)
# sys.exit()
cryst = crystal.CrystalStructure('2LYZ')
q_vecs = geom.q_vecs(beam=beam)
A = np.zeros((3, 3))
a0 = 1e-9
A[0, 0] = 1/a0
A[1, 1] = 1/a0
A[2, 2] = 1/a0
F = np.random.rand(10, 10, 10)
var_siz = (20e-3/a0)**2
var_mos = (20e-3)**2 * 0
var_wav = (5e-3)**2 * 0
var_div = (5e-3)**2 * 0
B = (100e-3*a0)**2 * 0
Iout = np.empty(q_vecs.shape[0])
kin = beam.k_in
neighbors = 1
class Getter(FrameGetter):
    df = dataframe.DataFrame(pad_geometry=geom, beam=beam)
    def __init__(self):
        super().__init__()
        self.n_frames = 1e6
    def get_data(self, frame_number=0):
        R = np.identity(3)
        R = utils.random_rotation()
        print(np.dot(R, R.T))
        t = time.time()
        crystal_f.gaussian_crystal(q_vecs.T, kin.T, A.T, F.T, R.T, var_siz, var_mos, var_wav, var_div, B,
                                   neighbors, Iout.T)
        print(f"{time.time()-t} seconds")
        self.df.set_raw_data(Iout)
        return self.df
fg = Getter()
pv = PADView(frame_getter=fg, levels=[-0.01, .2])
pv.start()
