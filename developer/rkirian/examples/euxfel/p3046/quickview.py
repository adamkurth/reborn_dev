import sys
from reborn.external import crystfel, euxfel
geom_file = 'geometry/p3046_manual_refined_geoass_run10.geom'
geom = crystfel.geometry_file_to_pad_geometry_list(geom_file)
experiment_id = 3046
run_id = 32
if len(sys.argv) == 2:
    run_id = int(sys.argv[1])
fg = euxfel.EuXFELFrameGetter(experiment_id=experiment_id, run_id=run_id, geom=geom)
fg.view(debug_level=2)
