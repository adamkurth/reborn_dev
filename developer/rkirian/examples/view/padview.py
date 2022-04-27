import sys, time
import numpy as np
import pyqtgraph as pg
from reborn.external import crystfel
from reborn.simulate import solutions
from reborn.viewers.qtviews import PADView
from reborn import detector, dataframe, source
from reborn.fileio.getters import FrameGetter
from reborn.const import eV
import pandas
# np.random.seed(0)
pdb = '1LYZ'
geom = detector.cspad_pad_geometry_list(detector_distance=0.1)
# print(geom)
# geom = crystfel.geometry_file_to_pad_geometry_list('../lcls/cxix53120/calib/jungfrau.geom')
# geom.translate([0, 0, 0.1])
beam = source.Beam(photon_energy=9500*eV, pulse_energy=1e-5, diameter_fwhm=100e-9)
class MyFrameGetter(FrameGetter):
    def __init__(self):
        super().__init__()
        self.n_frames = 1000
        self.init_params = {}
        self.pandas_dataframe = pandas.DataFrame({'Frame #': np.arange(self.n_frames)})
        # self.simulator = examples.PDBMoleculeSimulator(pdb_file=pdb, pad_geometry=geom, beam=beam)
    def get_data(self, frame_number=0):
        np.random.seed(frame_number)
        # g = geom.copy()
        # g.translate([0, 0, 0.05])
        I = solutions.get_pad_solution_intensity(pad_geometry=geom, beam=beam, thickness=500e-6, poisson=True)
        # profiler = detector.RadialProfiler(beam=beam, pad_geometry=geom)
        # p = profiler.get_mean_profile(I)
        # pg.plot(profiler.bin_centers, p)
        # I = geom.q_mags(beam)
        # I = self.simulator.next()
        df = dataframe.DataFrame()
        df.set_beam(beam)
        df.set_pad_geometry(geom)
        df.set_raw_data(I)
        # df.set_dataset_id(pdb)
        return df
def processor(dat):
    # print('='*30, 'update mask')
    # m = dat.get_mask_list()
    # m[0][:, :] = 0
    # dat.set_mask(m)
    return dat

frame_getter = MyFrameGetter()
# import pandas
# frame_getter.pandas_dataframe = pandas.DataFrame({'1': np.arange(1000)*2, '2': np.sin(np.arange(
#     1000)/100)})
# frame_getter.view(debug_level=1)
pv = PADView(frame_getter=frame_getter, debug_level=1, dataframe_preprocessor=processor)
# pv.save_screenshot('/home/rkirian/Downloads/test.jpg')
# pv.run_plugin('view_pandas_table')
# pv.run_plugin('scattering_profile')
pv.run_plugin('shift_detector')
# pv.run_plugin('levels')
# pv.add_rings(q_mags=3.567e10)
# pv.show_all_geom_info()
# pv.show_pad_frames()
# x = (np.random.rand(1000, 2)-0.5)*1000
# pv.add_scatter_plot(x[:, 0], x[:, 1], pen=pg.mkPen('g'), brush=None, width=2, pxMode=False, size=10)
# pv.show_coordinate_axes()
# pv.show_grid()
# pv.show_pad_labels()
# dr = np.pi/180
# pv.add_rings(q_mags=[2e10]) #, d_spacings=[50e-10], pens=pg.mkPen([255, 0, 0], width=2))
# pv.run_plugin('scattering_profile')
pv.start()
