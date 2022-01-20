import numpy as np
from reborn import detector, source, dataframe, temp_dir
from reborn.analysis import runstats
from reborn.fileio.getters import ListFrameGetter


def test_padstats():
    geom = detector.cspad_2x2_pad_geometry_list()
    geom = geom.binned(10)
    beam = source.Beam(wavelength=1e-10)
    dataframes = []
    for i in range(3):
        dat = geom.zeros() + i
        df = dataframe.DataFrame(raw_data=dat, pad_geometry=geom, beam=beam)
        dataframes.append(df)
    fg = ListFrameGetter(dataframes)
    stats = runstats.padstats(framegetter=fg)
    assert(isinstance(stats, dict))
    assert(stats['sum'].flat[0] == 3)
    # runstats.view_padstats(stats)
    filepath = temp_dir + '/stats.npz'
    runstats.save_padstats(stats, filepath)
    stats2 = runstats.load_padstats(filepath)
    assert(isinstance(stats, dict))
    assert(stats['sum'].flat[0] == stats2['sum'].flat[0])
