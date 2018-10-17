r"""
This is the documentation for the bornagain package.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from . import detector
from . import scatter
# from . import simulate
from . import source
# from . import target
from . import units
from . import utils

from .detector import SimplePAD


CONFIG_OPTIONS = {'warn_depreciated': True,
                  'force_depreciated': False,
                  'verbose': False,
                  'debug': 0}


def set_global(option, value):

    r"""

    Set global configurations.

    Args:
        option: One of the following:
    ==================================  ================================================================
    Option                              Effect
    ==================================  ================================================================
    'warn_depreciated' (True/False):    Print warnings when depreciated classes etc. are used.
    'force_depreciated' (True/False):   Like 'warn_depreciated' but raise RunTimeError instead of warn.
    ==================================  ================================================================

        value: The value that you wish to associate with one of the above options.

    Returns: Nothing
    """

    global CONFIG_OPTIONS
    if option not in CONFIG_OPTIONS:
        raise KeyError('Unknown configuration option "%s"' % option)
    if option == 'warn_depreciated' and value not in (True, False):
        raise ValueError('warn_depreciated must be either True or False')
    CONFIG_OPTIONS[option] = value


def get_global(option):

    r"""

    Get global configurations.

    Args:
        option: See :func:`set_global` for the relevant options.

    Returns: The value associated with the option.
    """

    return CONFIG_OPTIONS[option]
