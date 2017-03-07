import sys

import numpy as np

sys.path.append("..")
import bornagain as ba

def test_random_rotation_matrix(main=False):
    
    R = ba.utils.random_rotation_matrix()
    d = np.linalg.det(R)
    
    if main:
        print("A random rotation matrix R:")
        print(R)
        print("Determinant of R:")
        print(d)
    assert(np.abs(d - 1) < 1e-15)
    
    
if __name__ == "__main__":
    
    main = True
    test_random_rotation_matrix(main)