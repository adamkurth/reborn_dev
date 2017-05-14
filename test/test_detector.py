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
    
def test_beam(main=False):
    
    p = ba.detector.Panel()
    wavelength = 1
    p.beam.wavelength = wavelength
    if main:
        print(p.beam)
    assert(p.beam.wavelength == wavelength) 
    
    pl = ba.detector.PanelList()
    pl.wavelength = 1 
    assert(pl.beam.wavelength == wavelength)
    
    if main:
        print(pl.beam)
        print(p._beam)

def test_reshape(main=False):
    
    p = ba.detector.Panel()
    nF = 3
    nS = 4
    p.simple_setup(nF=nF,nS=nS,pixel_size=1,distance=1,wavelength=1)
    d = np.arange(0,nF*nS)
    p.data = d.copy()
    if main:
        print(d)
    d = p.reshape(d)
    if main:
        print(d.shape)
        print(d)
        print(p.data)
    assert(d.shape[1] == nF)
    assert(d.shape[0] == nS)

def test_panel_simple_setup(main=False):
    
    p = ba.detector.Panel()
    nF = 3
    nS = 4
    p.simple_setup(nF=nF,nS=nS,pixel_size=1,distance=1,wavelength=1)
    if main:
        print(p.V[0,:],p.V[-1,:])
    assert(p.V[0,0] == -nF/2.0 + 0.5)
    assert(p.V[-1,0] == nF/2.0 - 0.5)
    assert(p.V[0,1] == -nS/2.0 + 0.5)
    assert(p.V[-1,1] == nS/2.0 - 0.5)
    
# def test_panellist_simple_setup(main=False):
#     
#     pl = ba.detector.PanelList()
#     pl.simple_setup(nF=nF,nS=nS,pixel_size=1,distance=1,wavelength=1)
#     assert(pl[0].V[0,0] == -nF/2.0 + 0.5)
#     assert(pl[0].V[-1,0] == nF/2.0 - 0.5)
#     assert(pl[0].V[0,1] == -nS/2.0 + 0.5)
#     assert(pl[0].V[-1,1] == nS/2.0 - 0.5)
#     assert(pl.V[0,0] == -nF/2.0 + 0.5)
#     assert(pl.V[-1,0] == nF/2.0 - 0.5)
#     assert(pl.V[0,1] == -nS/2.0 + 0.5)
#     assert(pl.V[-1,1] == nS/2.0 - 0.5)

def test_data(main=False):
    
    p = ba.detector.Panel()
    nF = 3
    nS = 4
    p.simple_setup(nF=nF,nS=nS,pixel_size=1,distance=1,wavelength=1)
    d = np.arange(0,nF*nS)
    p.data = d
    
    
if __name__ == "__main__":
    
    main = True
    test_set_data(main)
    test_simple_setup(main)
    test_beam(main)
    test_reshape(main)
    