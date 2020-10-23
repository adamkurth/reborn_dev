from reborn import detector


def plugin(self):
    r""" Plugin for PADView. """
    data = detector.concat_pad_data(self.get_pad_display_data())
    sang = detector.concat_pad_data([p.solid_angles() for p in self.pad_geometry])  # These are solid angles
    data /= sang*1e6  # FIXME: Why is this factor needed?  Why doesn't pyqtgraph display the data correctly without it?
    self.set_pad_display_data(data, auto_levels=True, update_display=True)
