r"""
This is the documentation for the bornagain package.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from . import detector
# from . import simulate
from . import source
from . import target
from . import utils

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
    'verbose' (True/False):             Attempt to print useful information
    'debug' (integer):                  This is intended for developers.  If the argument is 0,
                                        bornagain will run as usual.  The other numbers will have some
                                        special meaning that is not yet defined.
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


def docs():

    r"""

    Open the bornagain documentation in a web browser (if available).

    """

    import os
    import pkg_resources

    docs_file_path = pkg_resources.resource_filename('bornagain', '')
    docs_file_path = os.path.join(docs_file_path, '..', 'doc', 'html', 'index.html')

    if os.path.exists(docs_file_path):
        docs_file_path = 'file://' + docs_file_path
    else:
        docs_file_path = 'https://rkirian.gitlab.io/bornagain'

    try:
        import webbrowser
    except ImportError:
        print("Can't open docs because you need to install the webbrowser Python package.")
        print("If using conda, perhaps you could run 'conda install webbrowser'")
        print('You can otherwise point your webbrowser to https://rkirian.gitlab.io/bornagain')
        return

    webbrowser.open('file://' + docs_file_path)
