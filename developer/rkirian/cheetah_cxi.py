import sys
sys.path.append('../..')
from bornagain.viewers.qtviews import PADView
from bornagain.external.cheetah import CheetahFrameGetter

geom_file_name = 'data/cxin5016-oy-v1.geom'
cxi_file_name = 'data/cxilu5617-r0149-c00.cxi'
padview = PADView()
padview.crystfel_geom_file_name = geom_file_name
padview.main_window.setWindowTitle(cxi_file_name)
padview.frame_getter = CheetahFrameGetter(cxi_file_name, geom_file_name)
padview.pad_geometry = padview.frame_getter.pad_geometry
padview.show_frame(frame_number=0)
padview.start()
