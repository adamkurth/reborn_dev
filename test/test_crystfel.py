from reborn.data import cspad_geom_file
from reborn.external import crystfel
 

def test_crystfel():

    geom_dict = crystfel.geometry_file_to_pad_geometry_list(cspad_geom_file)
    assert(isinstance(geom_dict, list))
    assert (len(geom_dict) == 64)
