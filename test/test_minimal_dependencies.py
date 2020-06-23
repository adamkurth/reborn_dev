r"""
This should work with the following conda setup:

> conda create --name reborn_minimal python=3 pytest scipy h5py
"""


def test_imports():

    import reborn
    assert reborn is not None


def test_detector():
    from reborn import detector
    p = detector.PADGeometry()
    assert p is not None


def test_source():
    from reborn import source
    b = source.Beam()
    assert b is not None


def test_utils():
    from reborn import utils
    assert utils is not None