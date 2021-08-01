r"""
This is the documentation for the reborn package.
"""
import os
import tempfile
from . import utils
from . import source
from . import detector
from . import dataframe
from . import simulate
from . import target
from . import fileio

temp_dir = os.path.join(tempfile.gettempdir(), 'reborn')
