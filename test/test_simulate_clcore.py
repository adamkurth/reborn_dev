import sys

import numpy as np
import matplotlib.pyplot as plt
import pytest

sys.path.append('..')
try:
    from bornagain.simulate import clcore
    import bornagain as ba
    havecl = True
except ImportError:
    havecl = False

@pytest.mark.skipif(havecl is False, reason="requires pyopencl module")
def test_two_atoms(main=False):

    # Check if pyopencl is available
    assert(havecl == True)

    r = np.zeros([2,3])
    r[1,0] = 1e-10
    
    f = np.ones([2])
    
    R = np.eye(3)
    
    q = np.zeros([2,3])
    
    A = clcore.phase_factor_qrf(q, r, f, R)
    
    I = np.abs(A)**2
    
    assert(np.max(I) == 4.0)
    
    return True
    
@pytest.mark.skipif(havecl is False, reason="requires pyopencl module")
def test_equivalence_pad_qrf(main=False):
    
    nF = np.int(50)
    nS = np.int(101)
    pix = np.float32(100e-6)
    w = np.float32(1.5e-10)
    F = np.array([1,0,0],dtype=np.float32)*pix
    S = np.array([0,1,0],dtype=np.float32)*pix
    B = np.array([0,0,1],dtype=np.float32)
    T = np.array([0,0,0.1],dtype=np.float32)
    R = np.eye(3,dtype=np.float32)
    
    p = ba.detector.Panel()
    p.nF = nF
    p.nS = nS
    p.F = F
    p.S = S
    p.beam.B = B
    p.beam.wavelength = w
    p.T = T
    
    # Scattering vectors generated from panel object
    q = np.float32(p.Q)
    
    # Atomic coordinates
    N = 5
    r = np.zeros([N,3],dtype=np.float32)
    r[1,0] = 1000e-10
    r[2,1] = 1000e-10
    
    # Scattering factors
    f = np.ones([N],dtype=np.float32)
    
    # Compute diffraction amplitudes
    A1 = clcore.phase_factor_qrf(q,r,f)
    A2 = clcore.phase_factor_pad(r,f,T,F,S,B,nF,nS,w,R)
    A1 = A1.astype(np.complex64)
    A2 = A2.astype(np.complex64)
    dif = np.max(np.abs(A1-A2))
    
    if main:
        print('Max difference: %g' % (dif))
        print('Showing qrf and pad')
        A1 = np.reshape(np.abs(A1)**2,[nS,nF])
        A2 = np.reshape(np.abs(A2)**2,[nS,nF])
        plt.imshow(A1,cmap='gray',interpolation='nearest')
        plt.show()
        plt.imshow(A2,cmap='gray',interpolation='nearest')
        plt.show()
        plt.imshow(A1-A2,cmap='gray',interpolation='nearest')
        plt.show()
    
    # This is a pretty weak tolerance...
    assert(dif <= 1e-4)
    
if __name__ == '__main__':
    
    print('Running as main')
    main = True
    test_two_atoms(main)
    test_equivalence_pad_qrf(main)
    