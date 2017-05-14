import sys
import numpy as np
sys.path.append('..')
import bornagain as ba
from bornagain.utils import vec_check

def test_panel(main=False):
    
    nF=3
    nS=4
    pixel_size=100e-6
    distance=1.5
    wavelength=1.5e-10
    T=(1.5,0,distance)
    F=[1.,0,0]
    S=(0,1.,0)
    
    p = ba.detector.Panel()
    
    # Check that any 3-tuple can be used to set vectors
    p.T = T
    p.F = F
    p.S = S
    p.nF = nF
    p.nS = nS
    p.beam.wavelength = wavelength
    
    # Check creation and clearing of geometry cache
    def cache_cleared():
        assert(p._v is None)
        assert(p._sa is None)
        assert(p._pf is None)
        assert(p._k is None)
        assert(p._ps is None)
        assert(p._rsbb is None)
        assert(p._gh is None)
    def create_cache():
        V = p.V
        sa = p.solid_angle
        pf = p.polarization_factor
        q = p.Q
        ps = p.pixel_size
        rsbb = p.real_space_bounding_box
    create_cache()
    p.T = T #  This should clear the cache
    cache_cleared()
    create_cache()
    p.F = F #  This should clear the cache
    cache_cleared()
    create_cache()
    p.S = S #  This should clear the cache
    cache_cleared()
    create_cache()
    p.pixel_size = pixel_size #  This should clear the cache
    cache_cleared()
    
    p.simple_setup(nF=nF,nS=nS,pixel_size=1,distance=distance,wavelength=wavelength)
    assert(p.V[0,0] == -nF/2.0 + 0.5)
    assert(p.V[-1,0] == nF/2.0 - 0.5)
    assert(p.V[0,1] == -nS/2.0 + 0.5)
    assert(p.V[-1,1] == nS/2.0 - 0.5)

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
    
    q = pl.Q
    q = pl[0].Q
    
    assert(np.max(np.abs(a - pl.data)) == 0)
    
def test_beam(main=False):
    
    p = ba.detector.Panel()
    wavelength = 1
    p.beam.wavelength = wavelength
    
    assert(p.beam.wavelength == wavelength) 
    
    pl = ba.detector.PanelList()
    pl.wavelength = 1 
    assert(pl.beam.wavelength == wavelength)
    

def test_reshape(main=False):
    
    p = ba.detector.Panel()
    nF = 3
    nS = 4
    p.simple_setup(nF=nF,nS=nS,pixel_size=1,distance=1,wavelength=1)
    d = np.arange(0,nF*nS)
    p.data = d.copy()
    d = p.reshape(d)
    assert(d.shape[1] == nF)
    assert(d.shape[0] == nS)


    
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

    
    
if __name__ == "__main__":
    
    main = True
    test_set_data(main)
    test_panel(main)
    test_panel_simple_setup(main)
    test_beam(main)
    test_reshape(main)
    