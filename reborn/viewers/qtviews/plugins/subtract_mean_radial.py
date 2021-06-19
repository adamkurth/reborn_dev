from time import time
import reborn


class Plugin():
    def __init__(self, padview):
        self.padview = padview
        self.profiler = reborn.detector.RadialProfiler(pad_geometry=padview.dataframe.get_pad_geometry(),
                                                       beam=padview.dataframe.get_beam())
        self.action()
    def action(self):
        padview = self.padview
        padview.debug('Calculating mean profile...', 1)
        t = time()
        data = self.profiler.subtract_profile(padview.get_pad_display_data(),
                                              mask=padview.dataframe.get_mask_list(), statistic='mean')
        padview.debug('Done (%g seconds)' % (time()-t), 1)
        padview.set_pad_display_data(data, auto_levels=True, update_display=True)
