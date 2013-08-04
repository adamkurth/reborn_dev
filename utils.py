"""

Some useful utility functions for pydiffract

"""

import sys


def warn(message):

    """ Simple warning message """

    sys.stdout.write("WARNING: %s\n" % message)


def error(message):

    """ Simple error message (to be replaced later...) """

    sys.stderr.write("ERROR: %s\n" % message)
