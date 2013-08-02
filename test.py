#!/usr/bin/env python

import convert

p = convert.crystfelToPanelArray("examples/example1.geom")

p.read("examples/example1.h5")

print(p.panels[1].I)

#print(p)


