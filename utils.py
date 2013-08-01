
import sys
from numpy import sum
from math import sqrt

def warn(message):
    sys.stdout.write("WARNING: %s\n" % message)

def error(message):
    sys.stderr.write("ERROR: %s\n" % message)

def norm2(vec):
    return sqrt(sum(vec**2))

