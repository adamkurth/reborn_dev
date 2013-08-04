#!/usr/bin/env python

import convert

pa = convert.crystfel_to_panel_list("examples/example1.geom")
print(pa[0])

pa.read("examples/example1.h5")

print(pa[0].I)

# pa.check()