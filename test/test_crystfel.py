from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
sys.path.append('..')

from bornagain.external import crystfel


def test_crystfel():

    geom_dict = crystfel.geometry_file_to_pad_geometry_list('../examples/data/crystfel/geom/cxin5016-oy-v1.geom')

    assert(isinstance(geom_dict, list))
    assert (len(geom_dict) == 64)

