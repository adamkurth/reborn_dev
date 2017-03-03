import numpy as np
try:
    import clcore
    havecl = True
except ImportError:
    havecl = False

def test_two_atoms():

    # Check if pyopencl is available
    assert(havecl == True)

    r = np.zeros([2,3])
    r[1,0] = 1e-10
    
    f = np.ones([2])
    
    q = np.zeros([2,3])
    
    A = clcore.phase_factor_qrf(q, r, f)
    
    I = np.abs(A)**2
    
    assert(np.max(I) == 4.0)
    
    return True
    
