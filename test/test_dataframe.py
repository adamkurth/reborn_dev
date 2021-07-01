import numpy as np
from reborn import detector, source, dataframe


def test_01():
    df = dataframe.DataFrame()
    beam = source.Beam(photon_energy=10*1.602e-19)
    geom = detector.pnccd_pad_geometry_list()
    raw = geom.solid_angles()
    df.set_raw_data(raw)
    df.set_pad_geometry(geom)
    df.set_beam(beam)
    proc = df.get_processed_data_list()
    sa = df.solid_angles
    pfac = df.polarization_factors
    q_mags = df.q_mags
    q_vecs = df.q_vecs
    assert(isinstance(proc, list))
    assert(isinstance(pfac, np.ndarray))
    assert (isinstance(q_mags, np.ndarray))
    assert (isinstance(q_vecs, np.ndarray))
    assert(sa is not None)
    df2 = df.copy()
    geom2 = df2.get_pad_geometry()
    assert(geom2 == geom)  # These are copies, with same parameters
    for g in geom2:  # Change detector distance in one of them
        g.t_vec[2] = 0
    assert(geom2 != geom)  # Now they are not equal
    geom3 = df2.get_pad_geometry()  # This should not be affected by the det. dist. change
    assert(geom == geom3)  # Indeed...


