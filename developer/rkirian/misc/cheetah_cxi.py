import sys
sys.path.append('../..')
from reborn.viewers.qtviews import PADView
from reborn.external.cheetah import CheetahFrameGetter

geom_file_name = 'data/cxin5016-oy-v1.geom'
cxi_file_name = 'data/cxilu5617-r0149-c00.cxi'
frame_getter = CheetahFrameGetter(cxi_file_name, geom_file_name)
pad_geometry = frame_getter.pad_geometry
padview = PADView(frame_getter=frame_getter, pad_geometry=pad_geometry)
padview.show_fast_scan_directions()
padview.add_circle_roi()
padview.add_rectangle_roi()
padview.start()
