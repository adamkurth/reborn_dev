import sys
sys.path.append('../..')
from bornagain.viewers.qtviews import PADView

geom_file_name = 'data/cxin5016-oy-v1.geom'
cxi_file_name = 'data/cxilu5617-r0149-c00.cxi'
padview = PADView()
padview.load_cheetah_cxi_file(cxi_file_name, geom_file_name)
padview.start()
