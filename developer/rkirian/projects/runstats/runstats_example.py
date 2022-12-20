import sys
import shutil
import numpy as np
import reborn
from reborn.simulate.solutions import get_pad_solution_intensity
from reborn.analysis import runstats, saxs
from reborn.const import eV
geom = reborn.detector.cspad_2x2_pad_geometry_list(detector_distance=0.1)
beam = reborn.source.Beam(photon_energy=9500*eV, diameter_fwhm=1e-6, pulse_energy=1e-3)
mask = geom.edge_mask()
def delete_checkpoints():
    shutil.rmtree("logs", ignore_errors=True)
    shutil.rmtree("checkpoints", ignore_errors=True)
class Getter(reborn.fileio.getters.FrameGetter):
    def __init__(self, geom=None, beam=None, mask=None, n_frames=1, **kwargs):
        self.geom = geom
        self.beam = beam
        self.mask = mask
        self.n_frames = n_frames
        self.none_frames = np.zeros(self.n_frames)
        pat = get_pad_solution_intensity(pad_geometry=self.geom, thickness=3e-6, beam=self.beam, poisson=False)
        self.pat = geom.concat_data(pat)
        self.fh = open('testfile', 'w+')
        d = dict(onefish='a', twofish='b')
        self.keys = d.keys()
    def get_data(self, frame_number=0):
        if self.none_frames[frame_number]:
            return None
        pat = np.random.poisson(self.pat)
        pat[0] *= 0
        pat[0] += 1
        df = reborn.dataframe.DataFrame(raw_data=pat, mask=self.mask, pad_geometry=self.geom, beam=self.beam)
        return df



# Test single process
n_processes = 1
n_frames = 100
delete_checkpoints()
framegetter = Getter(geom=geom, mask=mask, beam=beam, n_frames=n_frames)
histogram_params = dict(bin_min=0, bin_max=20, n_bins=21)
# Test with single processor
config = dict(log_file="logs/padstats", checkpoint_file='checkpoints/padstats', checkpoint_interval=21,
              message_prefix='Test Run -', reduce_from_checkpoints=True, histogram_params=histogram_params)
# stats = runstats.padstats(framegetter=framegetter, config=config)
# assert stats['sum'].flat[0] == n_frames
# assert np.sum(np.abs(np.sum(stats['histogram'], axis=1) - n_frames)) == 0
# # runstats.view_padstats(stats, histogram=True)
#
# # Test reloading of checkpoints from the above
# print('='*50)
# stats = runstats.padstats(framegetter=framegetter, config=config)
# # runstats.view_padstats(stats, histogram=True)
# assert stats['sum'].flat[0] == n_frames
# delete_checkpoints()
# print('='*50)
#
# # Test start/stop/step
# stats = runstats.padstats(framegetter=framegetter, start=1, stop=6, step=2, config=config)
# # runstats.view_padstats(stats, histogram=True)
# assert stats['sum'].flat[0] == 3
# print('='*50)
# delete_checkpoints()
#
# # Test multiple processes
# stats = runstats.padstats(framegetter=framegetter, n_processes=4, config=config)
# # runstats.view_padstats(stats, histogram=True)
# delete_checkpoints()
# print('='*50)
#
# # Test fewer frames than processes
# framegetter = Getter(geom=geom, mask=mask, beam=beam, n_frames=3)
# stats = runstats.padstats(framegetter=framegetter, n_processes=4, config=config)
# # runstats.view_padstats(stats, histogram=True)
# delete_checkpoints()


# Test input args that cannot be serialized
f = open('tst', 'w')
k = dict(a='a').keys()
framegetter = Getter(geom=geom, mask=mask, beam=beam, n_frames=3, slices=np.s_[1, 2, 3], a=f, k=k)
stats = runstats.padstats(framegetter=framegetter, n_processes=4, config=config)
# runstats.view_padstats(stats, histogram=True)
delete_checkpoints()


sys.exit()
#
# n_processes = 5
# shutil.rmtree("logs", ignore_errors=True)
# shutil.rmtree("checkpoints", ignore_errors=True)
# framegetter = dict(framegetter=Getter, kwargs=dict(geom=geom, mask=mask, beam=beam))
# config = dict(log_file="logs/padstats", checkpoint_file='checkpoints/padstats', checkpoint_interval=5,
#               message_prefix='Prefix:', reduce_from_checkpoints=True, debug=True)
# # config = None
# histparams = dict(bin_min=-30, bin_max=100, n_bins=100)
# stats = runstats.padstats(framegetter, n_processes=n_processes, parallel=True, verbose=1, config=config,
#                           histogram_params=histparams)
# runstats.view_padstats(stats, histogram=True)
# stats = runstats.padstats(framegetter, n_processes=n_processes, parallel=True, verbose=1, config=config)
# runstats.view_padstats(stats, histogram=True)
