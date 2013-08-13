#!/usr/bin/env python

from pydiffract import convert
import h5py



print("")

pa = convert.crystfel_to_panel_list("examples/example1.geom")
print(pa[0])

pa.read("examples/example1.h5")

# print(pa[0].data)

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
