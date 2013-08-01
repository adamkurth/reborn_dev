'''
Created on Jul 27, 2013

@author: kirian
'''

import numpy as np

class source(object):
    
    def __init__(self):
        
        self.wavelength = 0
        self.B = np.array([0,0,1])

    def print_(self):

        print("wavelength : %g" % self.wavelength)
        print("B : [%g, %g, %g]" % (self.B[0],self.B[1],self.B[2]))
        
    def sanityCheck(self):
        
        if self.wavelength == 0:
            return False
        return True


