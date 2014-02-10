import unittest
from pydiffract import detector, source
import numpy as np

def generateSimplePanelList():

    """ Generate a simple panel list consisting of a 200x200 array, but split into
    two panels that are just beside one another.  No beam information will be set 
    here."""

    pa = detector.panelList()
    p = detector.panel()
    p.nF = 200
    p.nS = 100
    p.T = np.array([-(100 - 0.5), 0.5, 100])
    p.data = np.random.random([100, 200])
    pa.append(p)
    p = detector.panel()
    p.nF = 200
    p.nS = 100
    p.T = np.array([-(100 - 0.5), -(100 - 0.5), 100])
    p.data = np.random.random([100, 200])
    pa.append(p)

    return pa


class detectorTests(unittest.TestCase):

    """ All tests relevant to detector classes. """

    def testPanelListCreation(self):

        """ Check that panel lists are created as expected. """

        pa = generateSimplePanelList()

        # Check for he right number of panels and total pixels
        self.failUnless(pa.nPanels == 2)
        self.failUnless(pa.nPix == 40000)

class sourceTests(unittest.TestCase):

    """ All tests relevant to source classes. """

    def testBeamCreation(self):

        """ Check initializations of beam class. """

        beam = source.beam()

        # Check that there is never a "default" wavelength specified.
        self.failUnless(beam.wavelength is None)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
