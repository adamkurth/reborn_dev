#!/usr/bin/env python

from bornagain import dataio, utils
import numpy as np
# from timeit import timeit
# import cProfile
# import pstats
import matplotlib.pyplot as plt
#from pylab import *


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
    test_polarization_factor(pa, verbose=True)
    test_print(pa, verbose=False)

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
    print("Checking PanelList copy function...")
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
    adat = pa.assembledData
    if verbose == True:
        adat[adat < 0] = 0
        plt.imshow(np.log(adat + 100), interpolation='nearest', cmap='gray')
        plt.show()
    print("Pass")

def test_polarization_factor(pa, verbose=False):
    print("Checking polarization factor... fixme")
    pf = pa.polarizationFactor
    if verbose == True:
        print(pf)

def test_print(pa, verbose=True):

    print(pa)

pl = dataio.h5Reader()
pl.loadCrystfel(geomFile="example1.geom")
pl.fileList = ["example1.h5"]
pl.getFrame(0)

p = pl.radialProfile()
plt.plot(p)
plt.show()
test_all(pl)




# adat = pa.assembledData()
#
#
# func = 'test_assemble(pa)'
# N = 1
# for i in range(N):
#
#     cProfile.run(func, filename='test.cprof')
#     stats = pstats.Stats("test.cprof")
#     stats.strip_dirs().sort_stats('time').print_stats()
