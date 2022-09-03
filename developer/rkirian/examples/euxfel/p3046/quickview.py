import sys
from reborn.external import crystfel, euxfel
geom = crystfel.geometry_file_to_pad_geometry_list('geometry/agipd_september_2022_v03.geom')
run_id = 32
if len(sys.argv) == 2:
    run_id = int(sys.argv[1])
fg = euxfel.EuXFELFrameGetter(experiment_id=3046, run_id=run_id, geom=geom)
fg.view()
