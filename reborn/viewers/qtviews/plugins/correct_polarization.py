from reborn import detector
def plugin(padview):
    r""" Plugin for PADView. """
    data = detector.concat_pad_data(padview.get_pad_display_data())
    data /= detector.concat_pad_data([p.polarization_factors(beam=padview.dataframe.get_beam()) for p in
                                      padview.dataframe.get_pad_geometry()])
    padview.set_pad_display_data(data, auto_levels=True, update_display=True)
