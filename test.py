#!/usr/bin/env python

import convert

print("")

pa = convert.crystfel_to_panel_list("examples/example1.geom")
print(pa[0])

pa.read("examples/example1.h5")

print(pa[0].data)

worked = pa.computeRealSpaceGeometry()

print(worked)

pa.consolidateData()


# pa.check()
