import sys, time
import numpy as np
import pyqtgraph as pg
from reborn.simulate import solutions
from reborn.viewers.qtviews import PADView
from reborn import detector, dataframe, source
from reborn.fileio.getters import FrameGetter
from reborn.const import eV
import pandas
class MyFrameGetter(FrameGetter):
    def __init__(self):
        super().__init__()
        self.geom = detector.epix100_pad_geometry_list(detector_distance=0.01)
        self.beam = source.Beam(photon_energy=9500 * eV, pulse_energy=1e-5, diameter_fwhm=100e-9)
        self.n_frames = 100
        self.pandas_dataframe = pandas.DataFrame({'Frame #': np.arange(self.n_frames)})
        self.df = None
        self.intensity = None
    def get_data(self, frame_number=0):
        np.random.seed(frame_number)
        if self.df is None:
            print('Simulating frame')
            tic = time.time()
            self.intensity = solutions.get_pad_solution_intensity(pad_geometry=self.geom, beam=self.beam, thickness=500e-6, poisson=False)
            print(time.time()-tic, 'seconds')
            df = dataframe.DataFrame()
            df.set_beam(self.beam)
            df.set_pad_geometry(self.geom)
            self.df = df
        df = self.df
        df.set_raw_data(np.random.poisson(np.double(self.intensity) * (1 + 0.5 * np.random.rand())))
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
pv = PADView(frame_getter=frame_getter, debug_level=1) #, dataframe_preprocessor=processor)
# pv.edit_ring_radii()
# pv = PADView(data=frame_getter.get_first_frame().get_raw_data_list(), debug_level=2)
# pv = PADView(data={'pad_data': frame_getter.get_first_frame().get_raw_data_list()}, debug_level=2)
# pv.save_screenshot('/home/rkirian/Downloads/test.jpg')
# pv.run_plugin('view_pandas_table')
# pv.run_plugin('snr_mask')
# pv.run_plugin('widgets/display_editor')
# pv.run_plugin('shift_detector')
pv.run_plugin('run_stats')
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
