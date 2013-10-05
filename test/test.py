#!/usr/bin/env python

from pydiffract import convert
import numpy as np
from timeit import timeit
import cProfile
import pstats
from pylab import *


def test_all(pa):

    test_V(pa)
    pa.deleteGeometryData()
    test_K(pa)
    pa.deleteGeometryData()
    test_dat(pa)
    test_copy(pa)
    test_bounding_box(pa, verbose=False)
    test_solid_angle(pa)
    test_assemble(pa, verbose=True)

def test_V(pa):
    print("Checking real-space vector calculation...")
    V = pa.V
    print("Pass")

def test_K(pa):
    print("Checking reciprocal-space vector calculation...")
    K = pa.K
    print("Pass")

def test_dat(pa):
    print("Checking basic intensity data operations...")
    dat = pa.data
    dat += 1
    pa.data = dat
    print("Pass")

def test_copy(pa):
    print("Checking panelList copy function...")
    pa2 = pa.copy()
    print("Pass")

def test_bounding_box(pa, verbose=False):
    print("Checking real-space bounding box calculation...")
    r = pa[0].getRealSpaceBoundingBox()
    if verbose == True:
        print('r')
        print(r / pa[0].pixSize)
    ra = pa.realSpaceBoundingBox
    if verbose == True:
        print('ra')
        print(ra / pa[0].pixSize)
    print("Pass")

def test_solid_angle(pa, verbose=False):
    print("Checking solid angle calculation...")
    sa = pa[0].solidAngle
    if verbose == True:
        print(sa.min())
        print(sa.max())
    print("Pass")

def test_assemble(pa, verbose=False):
    print("Checking simple 2d assembly...")
    adat = pa.simpleRealSpaceProjection
    if verbose == True:
        adat[adat < 0] = 0
        imshow(log(adat + 100), interpolation='nearest', cmap='gray')
        show()
    print("Pass")


[pa, reader] = convert.crystfelToPanelList("examples/example1.geom")
reader.getShot(pa, "examples/example1.h5")

test_all(pa)

# adat = pa.simpleRealSpaceProjection
#
#
# func = 'test_assemble(pa)'
# N = 1
# for i in range(N):
#
#     cProfile.run(func, filename='test.cprof')
#     stats = pstats.Stats("test.cprof")
#     stats.strip_dirs().sort_stats('time').print_stats()
