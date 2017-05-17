import sys
import numpy as np
sys.path.append('..')
import bornagain as ba

def test_all():
    
    pl = ba.detector.PanelList()
    pl.beam.wavelength = 1.5e-10
    
    p0 = ba.detector.Panel()
    p0.simple_setup(100,101,1000e-6,0.1,1.5e-10)
    pl.append(p0)
    
    p1 = ba.detector.Panel()
    p1.simple_setup(102,103,100e-6,1,1.5e-10)
    pl.append(p1)
    
    dat = pl.ones()
    mask = pl.ones()
    
    rp = ba.scatter.RadialProfile()
    nQBins = 100
    qRange = np.array([0.1,3])*1e10
    rp.make_plan(pl,mask=mask,nBins=nQBins,qRange=qRange)
    
    profile = rp.get_profile(dat, pl)
    
    assert(np.max(profile) == 1)
    assert(np.min(profile) == 0)
    
    