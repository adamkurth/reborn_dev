from reborn.source import Beam
import os

def test_json():
    fname = 'test.beam'
    b1 = Beam()
    b1.save_json(fname)
    b2 = Beam()
    b2.load_json(fname)
    os.remove(fname)
    d1 = b1.to_dict()
    d2 = b2.to_dict()
    for k in list(d1.keys()):
        assert(d1[k] == d2[k])
