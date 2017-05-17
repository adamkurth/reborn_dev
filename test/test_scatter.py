import sys
import numpy as np
sys.path.append('..')
import bornagain as ba

def test_all():
    
    pl = ba.detector.PanelList()
    pl.beam.wavelength = 1.5e-10
    p = ba.detector.Panel()
    p.simple_setup(100,101,1000e-6,0.1,1.5e-10)
    p.data = np.ones([p.nS,p.nF])
    p.mask = np.ones([p.nS,p.nF])
    p.mask[0,0] = 0
    pl.append(p)
    p = ba.detector.Panel()
    p.simple_setup(102,103,100e-6,1,1.5e-10)
    p.data = np.zeros([p.nS,p.nF])
    p.mask = np.ones([p.nS,p.nF])
    p.mask[0,0] = 0
    pl.append(p)
    
    rp = ba.scatter.RadialProfile()
    nQBins = 100
    qRange = np.array([0.1,3])*1e10
    rp.make_plan(pl,mask=pl.mask,nBins=nQBins,qRange=qRange)
    
    profile = rp.get_profile(pl)
    
    assert(np.max(profile) == 1)
    assert(np.min(profile) == 0)
    
    