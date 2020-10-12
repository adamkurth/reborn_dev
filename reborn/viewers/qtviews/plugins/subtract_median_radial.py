import numpy as np
from reborn import detector, source


def plugin(self):
    r""" Plugin for PADView. """
    mask = self.mask_data
    data = self.get_pad_display_data()
    pads = self.pad_geometry
    if self.beam is None:
        photon_energy = 9000*1.602e-19
        self.beam = source.Beam(photon_energy=photon_energy)
    q_mags = detector.concat_pad_data([p.q_mags(beam=self.beam) for p in pads])
    q_max = np.max(q_mags)
    n_bins = int(np.sqrt(q_mags.size)/2.0)
    profiler = detector.RadialProfiler(pad_geometry=pads, beam=self.beam, q_range=(0, q_max), n_bins=n_bins)
    mprof = profiler.get_median_profile(data, mask=mask)
    mprofq = profiler.bin_centers
    mpat = np.interp(q_mags, mprofq, mprof)
    mpat = detector.concat_pad_data(mpat)
    data = detector.concat_pad_data(data)
    data -= mpat
    data = detector.split_pad_data(pads, data)
    # for i in range(len(pad_data)):
    #     pad_data[i] -= np.median(pad_data[i], axis=1).reshape((pad_data[i].shape[0], 1))
    if self.processed_data is None:
        self.processed_data = {}
    self.processed_data['pad_data'] = data
    self.update_pads()
    self.set_levels()