#!/usr/bin/env python

from pydiffract import convert
import numpy as np
from timeit import timeit
import cProfile
import pstats

# print("")

[pa, reader] = convert.crystfelToPanelList("examples/example1.geom")

# print(pa[0])

reader.getShot(pa, "examples/example1.h5")

# print(pa[0].data)


def test_V(pa):

    V = pa.V

def test_K(pa):

    K = pa.K

def test_dat(pa):

    dat = pa.data

    dat += 1

    pa.data = dat

def test_copy(pa):

    pa2 = pa.copy()


def test_bounding_box(pa):

    r = pa.getRealSpaceBoundingBox()
    print(r)

def test_assemble(pa):

    adat = pa.assembledData

def test_solid_angle(pa):

    sa = pa[0].solidAngle
    print(sa.min())
    print(sa.max())

N = pa[0].N
print(N)





func = 'test_solid_angle(pa)'
N = 1
for i in range(N):

    cProfile.run(func, filename='test.cprof')
    stats = pstats.Stats("test.cprof")
    stats.strip_dirs().sort_stats('time').print_stats()
