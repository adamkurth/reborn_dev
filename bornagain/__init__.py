from . import detector
from . import scatter
# from . import simulate
from . import source
# from . import target
from . import units
from . import utils

from .detector import SimplePAD
from .target.crystal import Molecule


CONFIG_OPTIONS = {'warn_depreciated': True,
                  'force_depreciated': False}


def set_global(opt, value):
    global CONFIG_OPTIONS
    if opt not in CONFIG_OPTIONS:
        raise KeyError('Unknown configuration option "%s"' % opt)
    if opt == 'warn_depreciated' and value not in (True, False):
        raise ValueError('warn_depreciated must be either True or False')
    CONFIG_OPTIONS[opt] = value


def get_global(opt):
    """Return the value of a single global configuration option.
    """
    return CONFIG_OPTIONS[opt]