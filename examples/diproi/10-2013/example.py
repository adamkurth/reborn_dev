import numpy as np
import matplotlib.pyplot as plt
from pydiffract import dataio, detector

filePath = 'A13_IMG_53908109.h5'

# setup data reader
readerPlan = dataio.diproiPlan()
readerPlan.dataField = 'image/ccd1'
reader = dataio.diproiReader()
reader.setPlan(readerPlan)
getter = dataio.frameGetter()
getter.reader = reader

# setup detector geometry, panel list
panels = detector.panelList()
panels.simpleSetup(2048, 2048, 13.5e-6, 5e-2)
panels.wavelength = 32e-9



