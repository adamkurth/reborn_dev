#!/usr/bin/env python

import convert

pa = convert.crystfelToPanelArray("examples/example1.geom")

pa.read("examples/example1.h5")

print(pa)

p = pa.panels[0]

print(p.I)

p.check()