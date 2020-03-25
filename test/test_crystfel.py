from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
sys.path.append('..')


try:
    import cfelpyutils
    has_cfelpyutils = True
except:
    has_cfelpyutils = False


def test_crystfel():

    if has_cfelpyutils:
        from reborn.external import crystfel
        geom_dict = crystfel.geometry_file_to_pad_geometry_list('../examples/data/crystfel/geom/cxin5016-oy-v1.geom')

        assert(isinstance(geom_dict, list))
        assert (len(geom_dict) == 64)

