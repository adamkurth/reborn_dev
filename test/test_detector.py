import sys
import numpy as np
sys.path.append('..')
import bornagain as ba

def test_set_data(main=False):
    
    pl = ba.detector.PanelList()
    p = ba.detector.Panel()
    p.simple_setup(100,101,100e-6,1,1.5e-10)
    p.data = np.ones([p.nS,p.nF])
    pl.append(p)
    p = ba.detector.Panel()
    p.simple_setup(102,103,100e-6,1,1.5e-10)
    p.data = np.zeros([p.nS,p.nF])
    pl.append(p)
    
    a = pl.data
    pl.data = a
    
    if main:
        print(pl.beam)
    q = pl.Q
    q = pl[0].Q
    
    assert(np.max(np.abs(a - pl.data)) == 0)
    
if __name__ == "__main__":
    
    main = True
    test_set_data(main)