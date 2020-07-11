import numpy as np
from reborn import detector, source


def plugin(self):
    r""" Plugin for PADView. """
    self.debug('plugin(subtract_median_ss)')
    mask = self.mask_data
    pad_data = self.get_pad_display_data()
    pad_geometry = self.pad_geometry
    if self.beam is None:
        photon_energy = self.get_float(label="Enter photon energy in keV", text="9")*1.602e-19*1e3
        print('photon_energy', photon_energy)
        self.beam = source.Beam(photon_energy=photon_energy)
    q_mags = detector.concat_pad_data([p.q_mags(beam=self.beam) for p in pad_geometry])
    q_max = np.max(q_mags)
    n_bins = int(len(np.sqrt(q_mags))/2.0)
    profiler = detector.RadialProfiler(pad_geometry=pad_geometry, beam=self.beam, q_range=(0, q_max), n_bins=n_bins)
    print('Getting median profile...')
    median = profiler.get_median_profile(pad_data, mask=mask)
    print('Done!')
    median_q_mags = profiler.bin_centers
    for i in range(len(pad_data)):
        pad_data[i] -= np.median(pad_data[i], axis=1).reshape((pad_data[i].shape[0], 1))
    if self.processed_data is None:
        self.processed_data = {}
    self.processed_data['pad_data'] = pad_data
    self.update_pads()