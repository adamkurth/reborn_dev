import sys
import time
import numpy as np
from reborn import source, detector, utils, dataframe, const
from reborn.fileio.getters import FrameGetter
from reborn.simulate import solutions
from reborn.target import crystal
from reborn.viewers.qtviews import PADView
from reborn.fortran import crystal_f
np.random.seed(0)
# A few parameters for the simulation:
beam = source.Beam(photon_energy=9000*const.eV, pulse_energy=5e-3)
geom = detector.jungfrau4m_pad_geometry_list(detector_distance=0.15)
cryst = crystal.CrystalStructure('2LYZ')
# Some stuff that is needed for the crystal simulation.  The Fortran code for crystal sim is under development and
# the intensities are not yet normalized appropriately.
q_vecs = geom.q_vecs(beam=beam)
A = np.zeros((3, 3))
a0 = 1e-9  # Make a fake unit cell
A[0, 0] = 1/a0
A[1, 1] = 1/a0
A[2, 2] = 1/a0
F = np.random.rand(10, 10, 10)  # Fake structure factors
var_siz = (40e-3/a0)**2
var_mos = (20e-3)**2 * 0
var_wav = (5e-3)**2 * 0
var_div = (5e-3)**2 * 0
B = (100e-3*a0)**2 * 0
Iout = np.empty(q_vecs.shape[0])
kin = beam.k_in
neighbors = 1
# Pre-compute water background (from Hura and Clark papers)
water = solutions.get_pad_solution_intensity(pad_geometry=geom, beam=beam, liquid='water', thickness=4e-6,
                                             poisson=False)
water = geom.concat_data(water)
# Make a "FrameGetter" so we can flip through simulations
class Getter(FrameGetter):
    df = dataframe.DataFrame(pad_geometry=geom, beam=beam)
    def __init__(self):
        super().__init__()
        self.n_frames = 1e6
    def get_data(self, frame_number=0):
        R = utils.random_rotation()  # np.identity(3)
        t = time.time()
        crystal_f.gaussian_crystal(q_vecs.T, kin.T, A.T, F.T, R.T, var_siz, var_mos, var_wav, var_div, B,
                                   neighbors, Iout.T)
        print(f"{time.time()-t} seconds")
        self.df.set_raw_data(np.random.poisson(Iout*1000+water))
        return self.df
# Here's the framegetter instance
getter = Getter()
# Framegetters return "DataFrame" instances that have various info about detector, beam and diffraction data
dat = getter.get_frame(0)
# There are three different ways to structure the diffraction data.  Most often I work with all the PADs concatenated
# together for vectorized operations.
raw = dat.get_raw_data_flat()
print(raw.shape)
# You can also reshape to the "native" format, e.g. as it comes from psana in the case of Jungfrau
raw = geom.reshape(raw)
print(raw.shape)
# Often times you need to work on individual 2D arrays, so we can split into a list of PADs
raw = geom.reshape(raw)
print(raw.shape)
# Here is a viewer to flip through frames
# pv = PADView(frame_getter=getter)  #, levels=[-0.01, .2])
# pv.start()