from time import time
import reborn


class Plugin():
    def __init__(self, padview):
        self.padview = padview
        self.profiler = reborn.detector.RadialProfiler(pad_geometry=padview.pad_geometry, beam=padview.beam)
        self.action()
    def action(self):
        padview = self.padview
        padview.debug('Calculating mean profile...', 1)
        t = time()
        data = self.profiler.subtract_profile(padview.get_pad_display_data(), mask=padview.mask_data, statistic='mean')
        padview.debug('Done (%g seconds)' % (time()-t), 1)
        padview.set_pad_display_data(data, auto_levels=True, update_display=True)
