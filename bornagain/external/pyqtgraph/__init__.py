"""
This contains the entire pyqtgraph module along with a few extra functions. 

The pyqtgraph module was cloned from here:
https://github.com/pyqtgraph/pyqtgraph-core


Here are the extra utilities:

keep_open    Keeps a window open (e.g. when window vanishes upon exiting script)

Utilities for operating on the colormap of multiple images at once
MultiHistogramLUTWidget
MultiHistogramLUTItem
"""

from .extras import *
from .pyqtgraph_core import *
