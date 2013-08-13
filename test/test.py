#!/usr/bin/env python

from pydiffract import convert
# import h5py


print("")

[pa, reader] = convert.crystfelToPanelList("examples/example1.geom")

print(pa[0])

reader.getShot(pa, "examples/example1.h5")

print(pa[0].data)

# pa.computeRealSpaceGeometry()

V = pa.V
K = pa.K

dat = pa.data

dat += 1

pa.data = dat

print(pa[0].data.dtype)

pa2 = pa.copy()

print(pa[0].F[0])


# pa.check()
