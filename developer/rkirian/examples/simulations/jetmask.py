import numpy as np
import reborn
from reborn.analysis.masking import StreakMasker
from reborn.viewers.qtviews import PADView
from reborn.simulate.solutions import get_pad_solution_intensity
from reborn.const import eV
geom = reborn.detector.jungfrau4m_pad_geometry_list(detector_distance=0.1)
for g in geom:
    g.do_cache = True
geom.do_cache = True
beam = reborn.source.Beam(photon_energy=9500*eV, diameter_fwhm=1e-6, pulse_energy=1e-3)
mask = geom.edge_mask()
class StreakGetter(reborn.fileio.getters.FrameGetter):
    def __init__(self, geom, beam, mask):
        self.geom = geom
        self.beam = beam
        self.mask = mask
        self.pattern = get_pad_solution_intensity(pad_geometry=geom, thickness=3e-6, beam=beam, poisson=False)
        self.pattern = geom.concat_data(self.pattern)
        self.masker = StreakMasker(geom, beam, 100, [0, 2e10])
        super().__init__()
        self.n_frames = 1e6
    def get_data(self, frame_number=0):
        pat = self.pattern.copy()
        for i in range(int(np.random.rand()*3)):
            angle = (90 - (np.random.rand()-0.5)*30) * np.pi / 180  # Angle of the streak
            pixel_vecs = self.geom.s_vecs()  # Unit vectors from origin to detector pixels
            streak_vec = self.beam.e1_vec * np.cos(angle) + self.beam.e2_vec * np.sin(angle)
            phi = 90 * np.pi / 180 - np.arccos(np.abs(np.dot(pixel_vecs, streak_vec)))
            theta = np.arccos(np.abs(np.dot(pixel_vecs, beam.beam_vec)))
            streak = np.random.rand() * np.random.poisson(100 * np.exp(-1000000 * phi ** 2 - 100 * theta ** 2))
            pat += streak
        pat = np.random.poisson(pat)
        smask = self.masker.get_mask(pat, self.mask)
        df = reborn.dataframe.DataFrame(raw_data=pat, mask=smask*self.mask, pad_geometry=self.geom, beam=self.beam)
        return df
getter = StreakGetter(geom, beam, mask)
padview = PADView(getter, debug=1)
padview.set_mask_color([0, 128, 0, 50])
padview.start()
